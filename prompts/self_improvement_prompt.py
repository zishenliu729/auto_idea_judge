import os
import json

from utils.common_utils import load_json_file, read_file


# ============================================================
# prompts/self_improvement_prompt.py
# 自改进诊断提示词模块——DGM 中"识别问题、生成改进方案"的核心提示词库。
#
# 工作流程（在 self_improve_step.py 中被调用）：
#   1. o1 模型接收 system_message（含当前 agent 代码）+ user prompt（含运行日志）
#   2. 返回 JSON，包含 log_summarization / potential_improvements /
#      improvement_proposal / implementation_suggestion / problem_description
#   3. problem_description 被格式化为 coding_agent 的任务指令（problem_statement）
#   4. coding_agent 按指令修改自己的代码，生成 model_patch
#
# 与 diagnose_improvement_prompt.py 的关系：
#   - 本文件：诊断"问题是什么"（改进前分析，生成改进方案）
#   - diagnose_improvement_prompt.py：评估"改进是否有效"（改进后回顾分析）
# ============================================================


# ============================================================
# Agent 架构说明文本（用于 system_message，让 o1 了解 agent 结构）
# ============================================================

coding_agent_summary = """# Coding Agent Summary

- **Main File**: `coding_agent.py`
  - Primary Class: `AgenticSystem`
  - The `forward()` function is the central entry point.
  - Prompts are located either within the `forward()` function or in the `prompts/` directory.
- **Tools**: `tools/`
  - The `tools/` directory contains various tools that LLMs can use to perform specific tasks.
  - Each tool must have a `tool_info()` function that returns a JSON object containing 'name', 'description', and 'input_schema'. The 'input_schema' should be a JSON object containing 'type', 'properties', and 'required'.
  - Each tool must have a `tool_function()` function that takes the arguments defined in input_schema, performs the tool's task, and returns a string.
  - See other tools for reference.
- **Utilities**: `utils/`
  - The `utils/` directory contains utility functions used across the codebase.

- **Additional Details**:
  - The agent is very good at automatically utilizing the right available tools at the right time. So do not have an agentic flow that explicitly forces a tool's usage.
  - Common tools, such as file editing and bash commands, are easy for the agent to recognize and use appropriately. However, more complex and niche tools may require explicit instructions in the prompt.
  - Tools should be designed to be as general as possible, ensuring they work across any GitHub repository. Avoid hardcoding repository-specific details or behaviors (e.g., paths).
  - Do not use 'while True' loops in the agent's code. This can cause the agent to get stuck and not respond.
  - Verify the implementation details of helper functions prior to usage to ensure proper integration and expected behavior.
  - Do not install additional packages or dependencies directly. Update `requirements.txt` if new dependencies are required and install them using `pip install -r requirements.txt`.
\n\n"""

# Polyglot 版本的 agent 说明（增加了多语言支持和工具 schema 格式要求）
coding_agent_summary_polyglot = """# Coding Agent Summary

- **Main File**: `coding_agent.py`
  - Primary Class: `AgenticSystem`
  - The `forward()` function is the central entry point.
  - Prompts are located either within the `forward()` function or in the `prompts/` directory.
- **Tools**: `tools/`
  - The `tools/` directory contains various tools that LLMs can use to perform specific tasks.
  - Each tool must have a `tool_info()` function that returns a JSON object containing 'name', 'description', and 'input_schema'. The 'input_schema' should be a JSON object containing 'type', 'properties', and 'required'.
  - Each tool must have a `tool_function()` function that takes the arguments defined in input_schema, performs the tool's task, and returns a string.
  - See other tools for reference.
- **Utilities**: `utils/`
  - The `utils/` directory contains utility functions used across the codebase.

- **Additional Details**:
  - The coding agent trying to solve a programming task. A task is in one programming language, but the coding agent needs to deal with different languages including C++, Go, Java, JavaScript, Python, and Rust.
  - The agent is very good at automatically utilizing the right available tools at the right time. So do not have an agentic flow that explicitly forces a tool's usage.
  - Be detailed in the prompt about what steps (e.g. implementing tests, refining solutions, etc.) you would like the agent to execute.
  - Common tools, such as file editing and bash commands, are easy for the agent to recognize and use appropriately. However, more complex and niche tools may require explicit instructions in the prompt.
  - Tools should be designed to be as general as possible, ensuring they work across any task. Avoid hardcoding task-specific details or behaviors (e.g., paths or solutions).
  - DO NOT use 'while True' loops in the agent's code IN ANY CASE!! This can cause the agent to get stuck and not respond.
  - Verify the implementation details of helper functions prior to usage to ensure proper integration and expected behavior.
  - **DO NOT create parsing errors tools or functions, collecting raw error messages and letting the agent analyze them will be more efficient.**
\n\n
""" + """
### DOC: tool function schema

Carefully consider whether to add/enhance the current tool or edit the workflow in forward()

Pay special attention to making sure that "required" and "type" are always at the correct level of nesting. For example, "required" should be at the same level as "properties", not inside it.
Make sure that every property, no matter how short, has a type and description correctly nested inside it.
Other arguments than you have seen are not permitted. For example, in "edit_line_ranges" with "type": "array", arguments like "minItems" and "maxItems" are not permitted.
\n\n
"""


# ============================================================
# 诊断 system_message：包含当前 agent 代码，让 o1 了解待改进的代码
# ============================================================

diagnose_system_message = """Here is the implementation of the coding agent.

# Coding Agent Implementation
----- Coding Agent Implementation Start -----
{code}
----- Coding Agent Implementation End -----

Your task is to identify ONE detailed plan that would improve the agent's coding ability. The improvement should not be specific to any particular GitHub issue or repository.
"""
# 注意：system_message 限定"只提一个改进计划"，避免 o1 提出过于分散的改进

# 不同场景的问题描述前缀（区分 SWE-bench 和 Polyglot 任务）
swe_issue_prompt = "Here is the log for the coding agent trying to solve the GitHub issues but failed."
polyglot_issue_prompt = "Here is the log for the coding agent trying to solve a programming task. A task is in one programming language, but the coding agent needs to deal with different languages including C++, Go, Java, JavaScript, Python, and Rust."


# ============================================================
# 标准诊断 prompt（用于有具体失败案例的场景）
# ============================================================

diagnose_prompt = """
# Agent Running Log
----- Agent Running Log Start -----
{md_log}
----- Agent Running Log End -----

# GitHub Issue
The GitHub issue that the agent is trying to solve.
----- GitHub Issue Start -----
{github_issue}
----- GitHub Issue End -----

# Predicted Patch
The agent's predicted patch to solve the issue.
----- Predicted Patch Start -----
{predicted_patch}
----- Predicted Patch End -----

# Private Test Patch
SWE-bench's official private tests to detect whether the issue is solved. This is not available to the agent during evaluation. The agent should try to implement its own tests.
----- Private Test Patch Start -----
{test_patch}
----- Private Test Patch End -----

# Issue Test Results
The test results from SWE-bench using the above official private tests.
----- Issue Test Results Start -----
{eval_log}
----- Issue Test Results End -----

Respond precisely in the following format including the JSON start and end markers:

```json
<JSON>
```

In <JSON>, provide a JSON response with the following fields:
- "log_summarization": Analyze the above logs and summarize how the agent tried to solve the GitHub issue. Note which tools and how they are used, the agent's problem-solving approach, and any issues encountered.
- "potential_improvements": Identify potential improvements to the coding agent that could enhance its coding capabilities. Focus on the agent's general coding abilities (e.g., better or new tools usable across any repository) rather than issue-specific fixes (e.g., tools only usable in one framework). All necessary dependencies and environment setup have already been handled, so do not focus on these aspects.
- "improvement_proposal": Choose ONE high-impact improvement from the identified potential improvements and describe it in detail. This should be a focused and comprehensive plan to enhance the agent's overall coding ability.
- "implementation_suggestion": Referring to the coding agent's summary and implementation, think critically about what feature or tool could be added or improved to best implement the proposed improvement. If the proposed feature can be implemented by modifying the existing tools, describe the modifications needed, instead of suggesting a new tool.
- "problem_description": Phrase the improvement proposal and implementation suggestion as a GitHub issue description. It should clearly describe the feature so that a software engineer viewing the issue and the repository can implement it.

Your response will be automatically parsed, so ensure that the string response is precisely in the correct format. Do NOT include the `<JSON>` tag in your output."""


# ============================================================
# 特殊场景 prompt 变体（针对已知问题类型的专项诊断）
# ============================================================

# 变体 1：agent 生成了空 patch（没有任何代码修改）
# 让 o1 专注于分析"为什么 agent 没有做任何编辑"并提出解决方案
diagnose_prompt_emptypatches = """There are some empty patches when attempting to solve GitHub issues. Since the coding agent is stochastic, it may not always produce a patch. Handle cases where the coding agent fails to generate a patch or generates one that only modifies the test cases without editing the primary source code. For example, the simplest solution is to ask the agent to try again.

Respond precisely in the following format including the JSON start and end markers:

```json
<JSON>
```

In <JSON>, provide a JSON response with the following fields:
- "potential_improvements": Identify potential improvements to the coding agent's system. All necessary dependencies and environment setup have already been handled, so do not focus on these aspects.
- "improvement_proposal": Choose ONE high-impact improvement from the identified potential improvements and describe it in detail. This should be a focused and comprehensive plan to enhance the agent's overall coding ability.
- "implementation_suggestion": Referring to the coding agent's summary and implementation, think critically about what feature could be added or improved to best implement the proposed improvement.
- "problem_description": Phrase the improvement proposal and implementation suggestion as a GitHub issue description. It should clearly describe the feature so that a software engineer viewing the issue and the repository can implement it.

Your response will be automatically parsed, so ensure that the string response is precisely in the correct format. Do NOT include the `<JSON>` tag in your output."""

# 变体 2：agent 随机性导致结果不稳定
# 引导 o1 提出多次采样、选最优、利用历史上下文等方案
diagnose_prompt_stochasticity = """Since the coding agent is stochastic, it may not produce the correct patch for the given problem statement on the first try. Take into account the agent's stochastic nature and provide a solution to handle such cases. For example, one solution could be to ask the agent to try multiple times and select the best patch. The file `utils/eval_utils.py` contains helper functions to evaluate the generated patches. Giving previous attempts as context to the agent may also help.

Respond precisely in the following format including the JSON start and end markers:

```json
<JSON>
```

In <JSON>, provide a JSON response with the following fields:
- "potential_improvements": Identify potential improvements to the coding agent's system. All necessary dependencies and environment setup have already been handled, so do not focus on these aspects.
- "improvement_proposal": Choose ONE high-impact improvement from the identified potential improvements and describe it in detail. This should be a focused and comprehensive plan to enhance the agent's overall coding ability.
- "implementation_suggestion": Referring to the coding agent's summary and implementation, think critically about what feature could be added or improved to best implement the proposed improvement.
- "problem_description": Phrase the improvement proposal and implementation suggestion as a GitHub issue description. It should clearly describe the feature so that a software engineer viewing the issue and the repository can implement it.

Your response will be automatically parsed, so ensure that the string response is precisely in the correct format. Do NOT include the `<JSON>` tag in your output."""

# 变体 3：上下文长度超限（含具体错误信息）
# 告诉 o1 具体的错误类型，引导其提出上下文截断/压缩方案
# 注意：只处理"已抛出异常"的情况，不要无谓压缩未超限的上下文
diagnose_prompt_contextlength = """While the coding agent is attempting to solve GitHub issues, it encounters an error due to the input being too long for the requested model. This error is likely due to the context length exceeding the model's maximum input size. Handle cases where the input is too long for the model. The coding agent is mainly using the file `llm_withtools.py`. LLMs typically have a context window of 200k tokens. Handle context length only if the context window limit is reached and caught as an exception; otherwise, it is okay to leave it as is.

<error_message>
Error in get_response_withtools: Error code: 400 - {'message': 'Input is too long for requested model.'}
</error_message>

Respond precisely in the following format including the JSON start and end markers:

```json
<JSON>
```

In <JSON>, provide a JSON response with the following fields:
- "potential_improvements": Identify potential improvements to the coding agent's system. All necessary dependencies and environment setup have already been handled, so do not focus on these aspects.
- "improvement_proposal": Choose ONE high-impact improvement from the identified potential improvements and describe it in detail. This should be a focused and comprehensive plan to enhance the agent's overall coding ability.
- "implementation_suggestion": Referring to the coding agent's summary and implementation, think critically about what feature could be added or improved to best implement the proposed improvement.
- "problem_description": Phrase the improvement proposal and implementation suggestion as a GitHub issue description. It should clearly describe the feature and details so that a software engineer viewing the issue and the repository can implement it.

Your response will be automatically parsed, so ensure that the string response is precisely in the correct format. Do NOT include the `<JSON>` tag in your output."""


# ============================================================
# 问题描述模板（将 o1 的改进方案转化为 coding_agent 的任务指令）
# ============================================================

# {implementation_suggestion}：技术实现建议（告诉 agent 应该修改哪些文件/如何实现）
# {problem_description}：GitHub issue 风格的描述（给 agent 提供任务背景和目标）
problem_description_prompt = """# To Implement\n\n{implementation_suggestion}\n\n{problem_description}"""


def get_problem_description_prompt(response_json, is_polyglot=False):
    """
    将 o1 的诊断 JSON 响应转换为 coding_agent 的任务指令（problem_statement）。

    最终 problem_statement = agent架构说明 + 改进方案 + GitHub issue 描述，
    直接传给 coding_agent.py 的 forward()，让 agent 知道"要改什么"以及"如何改"。

    Args:
        response_json (dict): o1 返回的 JSON，含 implementation_suggestion 和 problem_description。
        is_polyglot (bool): 是否为 Polyglot 模式（影响使用哪个 agent 架构说明）。

    Returns:
        str: 完整的任务指令文本（problem_statement）。
    """
    if is_polyglot:
        return coding_agent_summary_polyglot + problem_description_prompt.format(
            implementation_suggestion=response_json["implementation_suggestion"],
            problem_description=response_json["problem_description"]
        )
    else:
        return coding_agent_summary + problem_description_prompt.format(
            implementation_suggestion=response_json["implementation_suggestion"],
            problem_description=response_json["problem_description"]
        )


def read_mdlog_file(filepath, filter=True):
    """
    读取 agent 运行日志文件，可选过滤掉噪声行。

    过滤目标："Error in get_response_withtools" 开头的行——
    这些是 API 调用错误，在诊断时属于噪声，不影响对 agent 行为的分析。
    诊断提示词中也明确要求"ignore errors with 'create_message_with_backoff'"。

    Args:
        filepath (str): .md 格式的日志文件路径。
        filter (bool): 是否过滤噪声行，默认 True。

    Returns:
        str: 日志内容（过滤后或原始）。
    """
    if not filter:
        return read_file(filepath)

    filter_content = [
        'Error in get_response_withtools',
    ]
    filtered_lines = []
    with open(filepath, 'r') as f:
        for line in f:
            if not any(line.startswith(fc) for fc in filter_content):
                filtered_lines.append(line.rstrip('\n'))
    return "\n".join(filtered_lines).strip()


def find_selfimprove_eval_logs(entry, out_dir, commit_id='initial', filter=True):
    """
    从指定 commit 版本的预测目录中，收集指定 entry 的所有运行数据。

    目录结构假设（由 harness.py 生成）：
      out_dir/{commit_id}/predictions/{pred_folder}/{entry}.md    ← agent 运行日志
      out_dir/{commit_id}/predictions/{pred_folder}/{entry}.json  ← 预测结果（含 model_patch）
      out_dir/{commit_id}/predictions/{pred_folder}/{entry}_eval.md ← 评估日志（优先使用）
      out_dir/{commit_id}/logs/run_evaluation/{f}/{f}/{entry}/report.json ← SWE-bench 评估结果

    注意 {f}/{f}/ 的双重目录：这是 SWE-bench 的 run_evaluation.py 遗留的目录命名方式，
    与 swe_bench/report.py 中的 run_id 命名保持一致。

    如果一个 entry 被评估了多次（num_evals > 1），这里会收集所有次的结果；
    process_selfimprove_eval_logs 只取第一次的结果。

    Args:
        entry (str): SWE-bench 任务 ID（如 "django__django-14999"）。
        out_dir (str): 评估输出的根目录（out_dir_base，非具体 run_id 目录）。
        commit_id (str): agent 代码版本的 commit hash 或 'initial'。
        filter (bool): 是否过滤日志噪声行。

    Returns:
        tuple: (md_logs, eval_logs, predicted_patches, eval_results)，每个都是 list。
    """
    predictions_dir = os.path.join(out_dir, commit_id, 'predictions')
    all_preds_folders = [f for f in os.listdir(predictions_dir) if os.path.isdir(os.path.join(predictions_dir, f))]

    prediction_log_files = [os.path.join(predictions_dir, f, f"{entry}.md") for f in all_preds_folders]
    prediction_json_files = [os.path.join(predictions_dir, f, f"{entry}.json") for f in all_preds_folders]
    # 过滤掉不存在的文件（某些 entry 可能在某次评估中未被处理）
    prediction_log_files = [f for f in prediction_log_files if os.path.exists(f)]
    prediction_json_files = [f for f in prediction_json_files if os.path.exists(f)]
    # 优先查找 _eval.md 格式（较新版本的评估结果格式）
    try_eval_logs = [os.path.join(predictions_dir, f, f"{entry}_eval.md") for f in all_preds_folders]
    try_eval_logs = [f for f in try_eval_logs if os.path.exists(f)]

    md_logs = []
    for file in prediction_log_files:
        md_logs.append(read_mdlog_file(file, filter=filter))

    predicted_patches = []
    eval_results = []
    for json_file in prediction_json_files:
        prediction_data = load_json_file(json_file)
        predicted_patch = prediction_data.get("model_patch", "")
        predicted_patches.append(predicted_patch)
        eval_result = prediction_data.get("eval_result", "")
        eval_results.append(eval_result)

    if not try_eval_logs:
        # 兜底：从 SWE-bench 标准报告目录读取评估结果
        # NOTE: {f}/{f}/ 是 SWE-bench report.py 的历史命名惯例
        eval_log_files = [
            os.path.join(out_dir, commit_id, f'logs/run_evaluation/', f, f, entry, 'report.json')
            for f in all_preds_folders
        ]
        eval_log_files = [f for f in eval_log_files if os.path.exists(f)]
        eval_logs = []
        for file in eval_log_files:
            eval_json = load_json_file(file)
            eval_logs.append(get_eval_log_text(eval_json))
    else:
        eval_logs = []
        for file in try_eval_logs:
            print(file)
            eval_logs.append(read_file(file))

    return md_logs, eval_logs, predicted_patches, eval_results


def process_selfimprove_eval_logs(md_logs, eval_logs, predicted_patches, eval_results):
    """
    从多次评估的结果中选取第一次，并截断过长的日志。

    为什么只用第一次：
      简单起见，只取第一次的结果作为代表。
      实际上多次评估结果可能不同（agent 的随机性），
      但诊断时用一次典型的运行日志足够。

    日志截断：
      250,000 字符约等于 50k-60k tokens，已经足够 o1 理解 agent 的行为模式。
      超长日志不仅浪费 tokens，还可能超过 o1 的上下文窗口。

    Args:
        md_logs (list[str]): 所有次 agent 运行日志。
        eval_logs (list[str]): 所有次评估结果。
        predicted_patches (list[str]): 所有次预测 patch。
        eval_results (list[str]): 所有次评估结果状态（如 "empty_patch"）。

    Returns:
        tuple: (md_log, eval_log, predicted_patch, eval_result) 各取第一个。
    """
    md_log = md_logs[0] if md_logs else "No logs available."
    eval_log = eval_logs[0] if eval_logs else "No test results available. Assume all tests failed."
    predicted_patch = predicted_patches[0] if predicted_patches else "No predicted patch available. Assume the agent failed."

    # 截断超长日志（>250k 字符 ≈ 50k+ tokens）
    if len(md_log) > 250000:
        md_log = md_log[:250000] + "\n<log clipped>"

    eval_result = eval_results[0] if eval_results else "No evaluation result available. Assume the agent failed."
    return md_log, eval_log, predicted_patch, eval_result


# ============================================================
# Polyglot 专用 prompt 变体（与 SWE-bench 版本类似，但侧重点不同）
# ============================================================

# Polyglot 空 patch 诊断：要求分析具体日志（比 SWE 版本更详细）
diagnose_prompt_emptypatches_polyglot = """There are some empty patches when attempting to solve GitHub issues. Since the coding agent is stochastic, it may not always produce a patch. Handle cases where the coding agent fails to generate a patch or generates one that only modifies the test cases without editing the primary source code. For example, the simplest solution is to change the prompt to specifically make sure it called the edit tool.

Please analyze the log below to identify why no code edits were made.

# Agent Running Log
----- Agent Running Log Start -----
{md_log}
----- Agent Running Log End -----

Respond precisely in the following format including the JSON start and end markers:

```json
<JSON>
```

In <JSON>, provide a JSON response with the following fields:
- "potential_improvements": Identify potential improvements to the coding agent's system. All necessary dependencies and environment setup have already been handled, so do not focus on these aspects.
- "improvement_proposal": Choose ONE high-impact improvement from the identified potential improvements and describe it in detail. This should be a focused and comprehensive plan to enhance the agent's overall coding ability.
- "implementation_suggestion": Referring to the coding agent's summary and implementation, think critically about what feature could be added or improved to best implement the proposed improvement.
- "problem_description": Phrase the improvement proposal and implementation suggestion as a GitHub issue description. It should clearly describe the feature so that a software engineer viewing the issue and the repository can implement it.

Your response will be automatically parsed, so ensure that the string response is precisely in the correct format. Do NOT include the `<JSON>` tag in your output."""

# Polyglot 随机性诊断：强调测试的重要性（因为 Polyglot 没有官方测试供 agent 参考）
diagnose_prompt_stochasticity_polyglot = """Since the coding agent is stochastic, it may not produce the correct patch for the given problem statement on the first try.
Take into account the agent's stochastic nature and provide a solution to handle such cases.
For example, one solution could be to ask the agent to try multiple times and select the best patch according to the test results. Scale the reflection times is also a good idea.
Giving previous attempts and test results as context to the agent may also help.
The tests for tasks are not provided in the repo, and the agent needs workflow design to implement them.
Agent's own tests that validate the solution may not cover all the cases that will be checked by the private tests during official scoring. So the quality of implemented tests are crucial.

Please analyze the log below to identify how we can improve the testing process and multiple solution attempts.

# Agent Running Log
----- Agent Running Log Start -----
{md_log}
----- Agent Running Log End -----

Respond precisely in the following format including the JSON start and end markers:

```json
<JSON>
```

In <JSON>, provide a JSON response with the following fields:
- "potential_improvements": Identify potential improvements to the coding agent's system. All necessary dependencies and environment setup have already been handled, so do not focus on these aspects.
- "improvement_proposal": Choose ONE high-impact improvement from the identified potential improvements and describe it in detail. This should be a focused and comprehensive plan to enhance the agent's overall coding ability.
- "implementation_suggestion": Referring to the coding agent's summary and implementation, think critically about what feature could be added or improved to best implement the proposed improvement.
- "problem_description": Phrase the improvement proposal and implementation suggestion as a GitHub issue description. It should clearly describe the feature so that a software engineer viewing the issue and the repository can implement it.

Your response will be automatically parsed, so ensure that the string response is precisely in the correct format. Do NOT include the `<JSON>` tag in your output."""


def get_diagnose_prompt_swe(entry_id, commit, root_dir, out_dir, dataset, patch_files=[]):
    """
    为 SWE-bench 任务构建诊断 prompt（system_message + user_prompt）。

    根据 entry_id 选择 prompt 变体：
      - 'solve_empty_patches'：agent 没有生成 patch 的专项诊断
      - 'solve_stochasticity'：agent 随机性导致结果不稳定的诊断
      - 'solve_contextlength'：上下文长度超限的诊断
      - 其他：标准诊断（含具体失败案例的运行日志）

    system_message 包含当前 agent 代码（coding_agent.py + tools/ + utils/ 的内容），
    但排除与进化逻辑相关的工具文件（避免 o1 被这些复杂代码分散注意力）。

    Args:
        entry_id (str): SWE-bench 任务 ID，或特殊值 ('solve_empty_patches' 等)。
        commit (str): agent 代码版本的 commit hash（用于查找对应版本的评估日志）。
        root_dir (str): DGM 根目录（/dgm/）。
        out_dir (str): 评估结果输出目录。
        dataset: SWE-bench 数据集（支持迭代查找）。
        patch_files (list[str]): 父代 patch 文件路径列表（用于在代码读取时叠加）。

    Returns:
        tuple[str, str]: (system_message, user_prompt) 可直接传给 get_response_from_llm。
    """
    if entry_id == 'solve_empty_patches':
        diagnose_prompt_out = diagnose_prompt_emptypatches
    elif entry_id == 'solve_stochasticity':
        diagnose_prompt_out = diagnose_prompt_stochasticity
    elif entry_id == 'solve_contextlength':
        diagnose_prompt_out = diagnose_prompt_contextlength
    else:
        md_logs, eval_logs, predicted_patches, eval_results = find_selfimprove_eval_logs(entry_id, out_dir, commit_id=commit)
        md_log, eval_log, predicted_patch, eval_result = process_selfimprove_eval_logs(md_logs, eval_logs, predicted_patches, eval_results)
        entry = next((e for e in dataset if e['instance_id'] == entry_id), None)
        answer_patch = entry['patch']
        test_patch = entry['test_patch']
        github_issue = entry['problem_statement']
        diagnose_prompt_out = swe_issue_prompt + diagnose_prompt.format(
            md_log=md_log, eval_log=eval_log, predicted_patch=predicted_patch,
            answer_patch=answer_patch, test_patch=test_patch, github_issue=github_issue
        )

    # system_message 包含 agent 源码（排除进化/评估辅助代码）
    code_files = ['coding_agent.py', 'tools/', 'utils/']
    exclude_files = [
        'utils/evo_utils.py',         # 进化算法工具（与 agent 改进无关）
        'utils/docker_utils.py',      # Docker 操作工具（与 agent 改进无关）
        'utils/swe_log_parsers.py',   # 日志解析器（与 agent 改进无关）
        'prompts/self_improvement_prompt.py',  # 本文件自身（避免循环）
    ]
    code_text = get_current_code(root_dir, code_files, patch_files=patch_files, exclude_files=exclude_files)
    diagnose_system_message_out = coding_agent_summary + diagnose_system_message.format(code=code_text)

    return diagnose_system_message_out, diagnose_prompt_out


def get_diagnose_prompt_polyglot(entry_id, commit, root_dir, out_dir, dataset, patch_files=[]):
    """
    为 Polyglot 任务构建诊断 prompt（system_message + user_prompt）。

    与 SWE-bench 版本的差异：
      1. 不区分特殊 entry_id（所有 Polyglot 任务都有具体日志）
      2. 使用 coding_agent_summary_polyglot（含多语言说明和工具 schema DOC）
      3. 25% 概率随机选择随机性诊断 prompt（鼓励多样化改进方向）
      4. 通过 eval_result 判断是否需要空 patch 诊断
      5. 兜底：标准诊断 prompt（含日志、预测 patch 和 ground truth）

    注意：随机选择的实现用了 random.random() < 0.25，
    这个"随机硬币"在 DGM 的每次改进尝试中独立投掷，
    使得不同代次的改进方向有所不同，避免陷入局部最优。

    Args: （参数同 get_diagnose_prompt_swe）

    Returns:
        tuple[str, str]: (system_message, user_prompt)。
    """
    md_logs, eval_logs, predicted_patches, eval_results = find_selfimprove_eval_logs(entry_id, out_dir, commit_id=commit)
    md_log, eval_log, predicted_patch, eval_result = process_selfimprove_eval_logs(md_logs, eval_logs, predicted_patches, eval_results)

    entry = next((e for e in dataset if e['instance_id'] == entry_id), None)
    assert entry, f"Could not find entry with id {entry_id} in dataset."
    is_polyglot = 'language' in entry
    # Polyglot 任务用 reference_answers/tests（而非 SWE-bench 的 patch/test_patch）
    answer_patch = entry['patch'] if not is_polyglot else entry['reference_answers']
    test_patch = entry['test_patch'] if not is_polyglot else entry['reference_tests']
    github_issue = entry['problem_statement']

    code_files = ['coding_agent.py', 'tools/', 'utils/']
    exclude_files = [
        'utils/evo_utils.py',
        'utils/docker_utils.py',
        'utils/swe_log_parsers.py',
        'utils/eval_utils.py',        # Polyglot 不用 SWE 评估工具
        'prompts/self_improvement_prompt.py',
    ]
    code_text = get_current_code(root_dir, code_files, patch_files=patch_files, exclude_files=exclude_files, is_polyglot=is_polyglot)

    import random

    # 25% 概率：直接使用随机性诊断（鼓励改进多次尝试能力，而非仅关注单次失败）
    if random.random() < 0.25:
        return coding_agent_summary_polyglot + diagnose_system_message.format(code=code_text), \
               diagnose_prompt_stochasticity_polyglot.format(md_log=md_log)
    # 空 patch：专项诊断（分析 agent 为何未做任何修改）
    if 'empty_patch' in eval_result:
        return coding_agent_summary_polyglot + diagnose_system_message.format(code=code_text), \
               diagnose_prompt_emptypatches_polyglot.format(md_log=md_log)
    # 标准情况：完整日志 + 预测 patch + ground truth
    return coding_agent_summary_polyglot + diagnose_system_message.format(code=code_text), \
           polyglot_issue_prompt + diagnose_prompt.format(
               md_log=md_log, eval_log=eval_log, predicted_patch=predicted_patch,
               answer_patch=answer_patch, test_patch=test_patch, github_issue=github_issue
           )


def get_eval_log_text(eval_json, test_status=None):
    """
    将 SWE-bench 的 report.json 格式化为人类可读的评估摘要文本。

    report.json 结构：
      {instance_id: {tests_status: {FAIL_TO_PASS: {success: [...], failure: [...]},
                                    PASS_TO_PASS: {success: [...], failure: [...]}}}}

    FAIL_TO_PASS：应该由 patch 修复的测试（衡量"是否解决了问题"）
    PASS_TO_PASS：原本就能通过的测试（衡量"是否引入了回归"）

    格式化为带 ✓/✗ 的可读文本，方便 o1 快速理解评估结果。

    Args:
        eval_json (dict): report.json 的内容。
        test_status: 预留参数（当前未使用）。

    Returns:
        str: 格式化的评估结果文本，或 "No test results available."。
    """
    if not test_status:
        first_key = next(iter(eval_json))
        tests_status = eval_json[first_key].get('tests_status', {})

    result_parts = []

    result_parts.append("## New tests for the issue")
    result_parts.append("These test whether the coding agent fixed the requested issue.")
    fail_to_pass = tests_status.get('FAIL_TO_PASS', {})
    if fail_to_pass.get('success'):
        result_parts.append(f"Successfully fixed {len(fail_to_pass['success'])}:")
        for test in fail_to_pass['success']:
            result_parts.append(f"  ✓ {test}")
    if fail_to_pass.get('failure'):
        result_parts.append(f"Failed to fix {len(fail_to_pass['failure'])} tests:")
        for test in fail_to_pass['failure']:
            result_parts.append(f"  ✗ {test}")
    else:
        result_parts.append(f"Pass All New Tests!")

    result_parts.append("## Previous tests from the repo")
    result_parts.append("These test whether the modification that coding agent made break the previous tests")
    pass_to_pass = tests_status.get('PASS_TO_PASS', {})
    if pass_to_pass.get('success'):
        result_parts.append(f"\nMaintained {len(pass_to_pass['success'])} passing tests")
    if pass_to_pass.get('failure'):
        result_parts.append(f"Regression in {len(pass_to_pass['failure'])} previously passing tests:")
        for test in pass_to_pass['failure']:
            result_parts.append(f"  ✗ {test}")
    else:
        result_parts.append(f"Pass All Previous Tests!")

    return "\n".join(result_parts) if result_parts else "No test results available. Assume all tests failed."


def get_current_code(current_dir, code_files, patch_files=None, exclude_files=None, is_polyglot=False):
    """
    读取指定文件/目录的 Python 代码内容，拼接为单一字符串（用于 system_message）。

    功能：
      - 支持单个文件和目录（递归读取 .py 文件）
      - 支持 exclude_files 黑名单（排除不相关的工具文件）
      - Polyglot 模式下将 coding_agent.py 替换为 coding_agent_polyglot.py
      - 可选在末尾附加 patch 文件内容（让 o1 了解当前版本相对初始版本的变化）

    代码文本格式：
      每个文件前缀 "# {rel_path}" 注释，方便 o1 定位代码位置。
      patch 文件格式："# Patch {i+1}: {rel_path}"。

    Args:
        current_dir (str): 解析相对路径的根目录（/dgm/）。
        code_files (list[str]): 要包含的文件或目录路径列表（相对于 current_dir）。
        patch_files (list[str] | None): 要附加的 patch 文件路径列表。
        exclude_files (list[str] | None): 要排除的文件路径列表（相对于 current_dir）。
        is_polyglot (bool): 是否为 Polyglot 模式。

    Returns:
        str: 所有代码内容拼接的文本。
    """
    if patch_files is None:
        patch_files = []
    if exclude_files is None:
        exclude_files = []

    exclude_set = set(exclude_files)
    code_text = []

    for file_path in code_files:
        full_path = os.path.join(current_dir, file_path)

        if file_path in exclude_set:
            continue

        if os.path.isfile(full_path):
            rel_path = os.path.relpath(full_path, current_dir)
            if rel_path not in exclude_set:
                # Polyglot 模式：用 polyglot 版本替换 SWE-bench 版本
                if is_polyglot and 'coding_agent.py' in file_path:
                    full_path = full_path.replace('coding_agent.py', f'coding_agent_polyglot.py')

                code_text.append(f"# {rel_path}")
                code_text.append(read_file(full_path))

        elif os.path.isdir(full_path):
            # 递归读取目录下所有 .py 文件
            for root, _, files in os.walk(full_path):
                for f in files:
                    if f.endswith('.py'):
                        file_full_path = os.path.join(root, f)
                        rel_path = os.path.relpath(file_full_path, current_dir)
                        if rel_path not in exclude_set:
                            code_text.append(f"# {rel_path}")
                            code_text.append(read_file(file_full_path))

    # 附加 patch 文件（让 o1 了解相对初始版本的累积变化）
    for i, patch_file in enumerate(patch_files):
        rel_path = os.path.relpath(patch_file, current_dir)
        if rel_path not in exclude_set:
            code_text.append(f"# Patch {i+1}: {rel_path}")
            code_text.append(read_file(patch_file))

    return "\n".join(code_text)
