from prompts.self_improvement_prompt import find_selfimprove_eval_logs, get_current_code, process_selfimprove_eval_logs
from utils.common_utils import read_file


# ============================================================
# prompts/diagnose_improvement_prompt.py
# 用于评估某次自改进是否有效的诊断提示词。
#
# 使用时机：在 self_improve_step.py 中，coding_agent 完成自改进后（生成了 model_patch），
# 调用此模块让 o1 模型比对"改进前"和"改进后"的 agent 运行日志，
# 判断 model_patch 对 agent 性能的影响（提升/无影响/回归），并给出数值评分。
#
# 与 self_improvement_prompt.py 的关系：
#   - self_improvement_prompt.py：诊断"问题是什么"（生成改进方案）
#   - diagnose_improvement_prompt.py：评估"改进是否有效"（改进后回顾分析）
# ============================================================


# -------- System prompt --------
diagnose_improvement_system_message = """Here is the relevant code for the LLM Coding agent with the model patch applied.

# LLM Coding Agent Code
The current code of LLM coding agent.
----- LLM Coding Agent Code Start -----
{code}
----- LLM Coding Agent Code End -----

# LLM Coding Agent Code Patch
The edits of LLM coding agent from the previous improvement.
----- LLM Coding Agent Code Patch Start -----
{model_patch_text}
----- LLM Coding Agent Code Patch End -----

# Test Patch
SWE-bench's official private tests to detect whether the issue is solved.
----- Test Patch Start -----
{test_patch}
----- Test Patch End -----

# Answer Patch
SWE-bench's official answer patch to the issue.
----- Answer Patch Start -----
{answer_patch}
----- Answer Patch End -----

# Your Task
Your task is to identify:
1. If the model patch has improved the agent's coding capabilities.
2. If the model patch has introduced any new issues or regressions.

Give a detailed analysis of the agent's performance before and after applying the model patch. Focus on the impact of the patch on the agent's problem-solving capabilities and general performance.
"""
# system_message 包含：当前 agent 代码 + 本次 model_patch 内容 + 官方测试/答案
# 这样 o1 既能理解 patch 做了什么改动，又有 ground truth 对比


# -------- User prompt --------
diagnose_improvement_prompt = """Here are the logs for the coding agent, before and after applying the model patch, trying to solve the GitHub issues. It will be VERY LONG. Think very hard on the impact of the model patch on the agent's performance.
Note: ignore errors with "create_message_with_backoff" and all API rate limit errors.

# Agent Running Log Before Patch
The coding agent's log before improvement
----- Log Before Patch Start -----
{md_log}
----- Log Before Patch End -----

# Predicted Patch Before Patch
The predicted patch from agent before improvement to try to solve issue.
----- Predicted Patch Before Patch Start -----
{prev_predicted_patch}
----- Predicted Patch Before Patch End -----

# Issue Test Results Before Patch
The test results before improvement from SWE-bench using the above official private tests.
----- Issue Test Results Before Patch Start -----
{eval_log}
----- Issue Test Results Before Patch End -----

# Agent Running Log After Patch
The coding agent's log after improvement
----- Log After Patch Start -----
{new_md_log}
----- Log After Patch End -----

# Predicted Patch After Patch
The predicted patch from agent after improvement to try to solve issue.
----- Predicted Patch After Patch Start -----
{new_predicted_patch}
----- Predicted Patch After Patch End -----

# Issue Test Results After Patch
The test results after improvement from SWE-bench using the above official private tests.
----- Issue Test Results After Patch Start -----
{new_eval_log}
----- Issue Test Results After Patch End -----


# Instruction

Respond precisely in the following format including the JSON start and end markers:

```json
<JSON>
```

In <JSON>, provide a JSON response with the following fields:
- "impact": Analyze the impact of the model patch on the agent's performance. Focus on how the patch has affected the agent's problem-solving capabilities and general performance. This should be a long and thorough analysis.
- "improvements": Identify any improvements introduced by the model patch that enhance the agent's capabilities.
- "regressions": Identify any new issues or regressions introduced by the model patch that affect the agent's effectiveness.
- "score": Provide an overall score for the model patch's impact on the agent's performance. This should be a numerical value between -2 and 2, where -2 indicates a significant negative impact, 0 indicates no impact, and 2 indicates a significant positive impact.

Your response will be automatically parsed, so ensure that the string response is precisely in the correct format. Do NOT include the `<JSON>` tag in your output.
Focus on analyzing the impact of the model patch on the agent's performance, identifying improvements and regressions, and providing an overall score for the patch's impact.
Your thinking should be thorough. Please think very deeply."""
# user_prompt 包含改进前后各一份：运行日志、预测 patch、测试结果
# 评分范围 -2 到 2（有意义的数值范围，而非简单的 0/1），
# 便于 DGM_outer.py 中按评分进行进化选择


def get_diagnose_improvement_prompt(
        entry_id, parent_commit, root_dir, model_patch_file, out_dir, run_id, dataset,
        patch_files=[],
    ):
    """
    构建评估自改进效果的完整 prompt（system + user）。

    数据收集逻辑：
      - "改进前"（parent_commit 的结果）：通过 find_selfimprove_eval_logs() 获取
        parent_commit 版本在该 entry 上的运行日志、预测 patch、测试结果
      - "改进后"（run_id 的结果）：获取当前版本在同一 entry 上的结果
      - model_patch_text：读取 model_patch_file 的内容（本次自改进生成的 diff）
      - 官方 ground truth：从 dataset 中取 entry_id 对应的 answer_patch 和 test_patch

    所有内容组合成 system_message 和 user_prompt，传给 o1 进行深度分析。

    Note: 当前实现签名接受 3 个返回值（md_log, eval_log, predicted_patch），
    与 find_selfimprove_eval_logs 实际返回 4 个值（多了 eval_results）不完全匹配，
    这是历史遗留 bug，实际调用时需要注意。

    Args:
        entry_id (str): SWE-bench 任务 ID（如 "django__django-14999"）。
        parent_commit (str): 父代版本的 commit hash（改进前）。
        root_dir (str): DGM 根目录。
        model_patch_file (str): 本次自改进生成的 model_patch.diff 路径。
        out_dir (str): 评估结果输出目录。
        run_id (str): 当前自改进的 run_id（改进后）。
        dataset: SWE-bench 数据集（支持迭代查找的集合）。
        patch_files (list[str]): 父代之前的所有 patch 文件路径列表。

    Returns:
        tuple[str, str]: (system_message, user_prompt)，可直接传给 get_response_from_llm()。
    """
    # 收集改进前的运行数据
    md_logs, eval_logs, predicted_patches = find_selfimprove_eval_logs(entry_id, out_dir, commit_id=parent_commit)
    md_log, eval_log, predicted_patch = process_selfimprove_eval_logs(md_logs, eval_logs, predicted_patches)

    # 读取当前 agent 代码（仅 coding_agent.py 和 tools/，排除其余文件）
    code_files = ['coding_agent.py', 'tools/']
    code_text = get_current_code(root_dir, code_files, patch_files=patch_files)
    # 读取本次自改进的 patch 内容（用于 system_message 展示改动）
    model_patch_text = read_file(model_patch_file)

    # 收集改进后的运行数据
    new_md_logs, new_eval_logs, new_predicted_patches = find_selfimprove_eval_logs(entry_id, out_dir, commit_id=run_id)
    new_md_log, new_eval_log, new_predicted_patch = process_selfimprove_eval_logs(new_md_logs, new_eval_logs, new_predicted_patches)

    # 从 dataset 获取 ground truth（官方答案和测试）
    entry = next((e for e in dataset if e['instance_id'] == entry_id), None)
    answer_patch = entry['patch']
    test_patch = entry['test_patch']

    return diagnose_improvement_system_message.format(
        code=code_text,
        model_patch_text=model_patch_text,
        answer_patch=answer_patch,
        test_patch=test_patch
    ), \
        diagnose_improvement_prompt.format(
            md_log=md_log, eval_log=eval_log,
            new_md_log=new_md_log, new_eval_log=new_eval_log,
            prev_predicted_patch=predicted_patch, new_predicted_patch=new_predicted_patch
        )
