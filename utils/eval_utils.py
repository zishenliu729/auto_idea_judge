import random
from llm import create_client, extract_json_between_markers, get_response_from_llm
from llm_withtools import convert_msg_history
from utils.swe_log_parsers import MAP_REPO_TO_PARSER


# ============================================================
# eval_utils.py
# 评估相关的工具函数，供 swe_bench/harness.py 和 coding_agent.py 调用。
#
# 核心职责：
#   1. parse_eval_output：将测试日志解析为 {测试名: 状态} 映射
#   2. msg_history_to_report：从 LLM 消息历史中提取最近一次测试报告
#   3. get_report_score：计算测试通过率（passed/total）
#   4. score_tie_breaker：用 o1 模型打破多个等分方案的平局
# ============================================================


def parse_eval_output(instance_id, eval_output):
    """
    将原始测试日志（eval_output）解析为结构化的测试结果映射表。

    关键转换步骤：instance_id → repo 名称 → 对应的 log_parser
      - SWE-bench 的 instance_id 格式如 "scikit-learn__scikit-learn-12421"
        （仓库名用 '__' 分隔，issue 编号用 '-' 分隔）
      - 转换：先把 '__' 替换成 '/'，再去掉最后一个 '-' 后面的数字部分，
        得到标准 "owner/repo" 格式（如 "scikit-learn/scikit-learn"）
      - 特殊处理：instance_id == 'dgm' 时直接用 'dgm'（DGM 自测模式）

    Args:
        instance_id (str): SWE-bench 任务的唯一标识符，如 "django__django-14999"，
                           或特殊值 'dgm'。
        eval_output (str): 测试框架的原始日志输出字符串。

    Returns:
        dict[str, str]: {测试用例名: 状态} 映射（状态为 PASSED/FAILED/ERROR/SKIPPED）；
                        出现任何异常时返回空 dict {}（静默失败，避免中断整体评估）。
    """
    try:
        if instance_id == 'dgm':
            repo = 'dgm'
        else:
            # 例："scikit-learn__scikit-learn-12421"
            # → replace('__', '/') → "scikit-learn/scikit-learn-12421"
            # → split('-')[:-1]    → ["scikit-learn/scikit-learn"]
            # → "-".join(...)      → "scikit-learn/scikit-learn"
            repo = "-".join(instance_id.replace("__", "/").split("-")[:-1])

        # 从 MAP_REPO_TO_PARSER 中查找对应仓库的解析函数
        log_parser = MAP_REPO_TO_PARSER[repo]
        return log_parser(eval_output)

    except Exception as e:
        # 未知仓库、解析失败等情况统一返回空 dict，不中断上层逻辑
        return {}


def msg_history_to_report(instance_id, msg_history, model=None):
    """
    从 LLM 的消息历史中提取最新的测试报告。

    工作原理：coding_agent 在运行测试时，测试输出通过工具调用结果（Tool Result）
    返回到消息历史中。此函数从最新的消息往回扫描（reversed），
    找到第一条包含 "Tool Result:" 的 user 消息，将其内容作为测试日志解析。

    为什么从后往前扫描：测试可能运行多次（初次运行、修复后重跑），
    只需要最后一次的结果，逆序扫描效率更高。

    Args:
        instance_id (str): SWE-bench 任务 ID，用于确定使用哪个 log_parser。
        msg_history (list): LLM 消息历史，格式因模型而异（通过 convert_msg_history 统一化）。
        model (str | None): 模型名称，用于 convert_msg_history 做格式兼容转换。

    Returns:
        dict[str, str]: {测试用例名: 状态} 映射；
                        找不到有效报告时返回空 dict {}。
    """
    # 将各模型格式的消息历史统一转换为通用格式（确保 role/content 字段存在）
    msg_history = convert_msg_history(msg_history, model=model)

    # 从最新消息往前找包含工具结果的 user 消息
    for msg in reversed(msg_history):
        if msg['role'] == 'user':
            # 取消息的第一个 content 块的文本内容
            content = msg['content'][0]['text']
            if 'Tool Result:' in content:
                report = parse_eval_output(instance_id, content)
                # 如果解析出了非空报告才返回（避免返回空报告误导评分）
                if report:
                    return report
    return {}


def get_report_score(test_report):
    """
    计算测试报告的通过率（PASSED 数 / 总测试数）。

    用途：coding_agent 用此分数评估自己对某个 issue 的修复质量，
    进化框架用此分数判断 child 是否优于 parent。

    Args:
        test_report (dict[str, str]): {测试用例名: 状态} 映射，
                                       由 msg_history_to_report 返回。

    Returns:
        float: 通过率 [0.0, 1.0]；测试集为空时返回 0（避免除以零）。
    """
    passed_count = sum([1 for v in test_report.values() if v == 'PASSED'])
    total_count = len(test_report)
    return passed_count / total_count if total_count > 0 else 0


def score_tie_breaker(problem_statement, code_diffs, test_reports, best_score_indices=[], logging=print):
    """
    当多个候选方案得分相同时，调用 o1 模型作为裁判，从中选出最优方案。

    为什么需要 tie-breaker：
      coding_agent 可能对同一个 issue 尝试多次（多轮 agentic 运行），
      每次生成不同的 patch，但测试通过率相同。此时需要更深层的判断——
      o1 能理解代码语义，综合考虑代码质量、可维护性等因素做出裁决。

    实现细节：
      - 把所有并列最优方案（best_score_indices 指定）的 diff 和测试报告组织成 prompt
      - 要求 o1 以 JSON 格式返回每个方案的评分（scores 列表）
      - 若多个方案评分相同，在其中随机选一个（random.choice，避免总选第一个）
      - 任何异常（API 失败、JSON 解析失败等）都回退到 best_score_indices[0]

    Args:
        problem_statement (str): issue 的问题描述文本。
        code_diffs (list[str]): 所有候选方案的 patch diff 字符串列表。
        test_reports (list[dict]): 对应的测试报告列表。
        best_score_indices (list[int]): 并列最优的方案下标列表；
                                        为空时认为所有方案都是候选。
        logging (callable): 日志输出函数，默认 print。

    Returns:
        int: 被选中方案在 code_diffs 列表中的下标。
    """
    # 未指定并列最优时，所有方案都参与裁决
    best_score_indices = list(range(len(code_diffs))) if not best_score_indices else best_score_indices
    # 默认回退值：取并列最优中的第一个
    best_score_index = best_score_indices[0]
    try:
        # 使用推理能力更强的 o1 做裁判（claude-3-5-sonnet 善于工具调用，o1 善于深度推理）
        client = create_client('o1-2024-12-17')
        # 构建每个候选方案的描述块（含 diff 和测试报告）
        proposed_solutions = [f'# Proposed solution {i+1}\n\n<code_diff_{i+1}>\n{code_diffs[index]}\n</code_diff{i+1}>\n<test_report_{i+1}>\n{test_reports[index]}\n</test_report_{i+1}>' for i, index in enumerate(best_score_indices)]
        proposed_solutions = '\n\n'.join(proposed_solutions)
        prompt = f"""Given the following problem statement, proposed solutions, and test reports, provide a summary of the differences between the code diffs and an evaluation of the proposed solutions.

<problem_description>
{problem_statement}
</problem_description>

{proposed_solutions}

Respond precisely in the following format including the JSON start and end markers:

```json
<JSON>
```

In <JSON>, provide a JSON response with the following fields:
- "difference_summary": Summary of the differences between the code diffs.
- "reasoning": Explanation of the reasoning behind the evaluation.
- "scores": List of numerical scores for each proposed solution.

Your response will be automatically parsed, so ensure that the string response is precisely in the correct format. Do NOT include `<JSON>` tag in your output.
"""
        response, msg_history = get_response_from_llm(
            msg=prompt,
            client=client[0],
            model=client[1],
            system_message='You are an excellent software engineer who has been asked to evaluate the proposed solutions to a problem statement.',
            print_debug=True,
            msg_history=None,
        )
        logging(repr(response))
        # 从响应中提取 JSON 块（extract_json_between_markers 处理 ```json ... ``` 包装）
        response_json = extract_json_between_markers(response)
        llm_scores = response_json['scores']
        # 若 o1 给多个方案同分，随机选其一（防止总选第一个带来偏差）
        llm_best_score_index = random.choice([i for i, score in enumerate(llm_scores) if score == max(llm_scores)])
        best_score_index = best_score_indices[llm_best_score_index]
    except Exception as e:
        # API 调用失败、JSON 解析出错等情况下静默回退到第一个并列最优
        logging(f"Error in score_tie_breaker: {e}")
    return best_score_index
