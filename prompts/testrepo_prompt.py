

# ============================================================
# prompts/testrepo_prompt.py
# 为 coding_agent 生成"如何在当前仓库中运行测试"的提示词片段。
#
# 核心问题：不同评测场景下，测试命令格式各不相同：
#   - SWE-bench 仓库（swerepo=True）：从 eval_script 解析实际测试命令
#   - Polyglot 编程任务（polyglot=True）：eval_script 直接就是运行命令
#   - DGM 自测（两者均 False）：固定使用 pytest 测试 tools/ 和 utils/
#
# 这个提示词片段会被嵌入到 coding_agent 的系统提示中，
# 告诉 agent 如何运行测试来验证其修改的正确性。
# ============================================================


def get_test_command(eval_script):
    """
    从 SWE-bench 的 eval_script 中解析出实际的测试命令（不含具体文件参数）。

    eval_script 是 SWE-bench 的评估脚本，通常最后几行是测试执行命令，
    倒数第二行是我们需要的基础测试命令（最后一行通常是 exit 或 set -e 等）。

    处理逻辑：
      1. 取 eval_script 的倒数第二行作为测试命令
      2. 检查最后一个参数是否是带点号的路径（如 tests/some_folder/some_file.py）
         - 若是（且不以 .py 结尾），说明是目录路径，需要给 agent 提供格式提示
         - 将路径格式从 tests/some_folder/some_file 转换为 some_folder.some_file
         （pytest 支持两种格式，但 "." 分隔格式更通用）
      3. 移除最后所有含 "." 的参数（这些是具体的测试文件路径，
         agent 应自行指定，不应写死在提示中）

    Args:
        eval_script (str): SWE-bench 评估脚本的完整内容。

    Returns:
        tuple[str, str]:
            - test_command: 基础测试命令字符串（不含具体文件）
            - test_hint: 提示 agent 如何指定测试文件的说明（可能为空字符串）
    """
    test_hint = ''
    # eval_script 的倒数第二行是实际的测试执行命令
    lines = eval_script.strip().split('\n')
    test_command = lines[-2].strip()
    # 检查命令末尾是否有带 "." 的参数（模块路径格式）
    parts = test_command.split()
    if '.' in parts[-1] and not parts[-1].endswith('.py'):
        # 带点路径格式：提示 agent 用 some_folder.some_file 格式指定测试
        test_hint = 'If the target test file path is tests/some_folder/some_file.py, then <specific test files> should be `some_folder.some_file`.'
    # 移除所有含 "." 的尾部参数（可能是具体测试模块路径）
    while parts and '.' in parts[-1]:
        parts.pop()
    # 重新拼接去掉文件参数后的命令
    test_command = ' '.join(parts)
    return f'cd /testbed/ && {test_command} <specific test files>', test_hint


def get_test_description(eval_script='', swerepo=False, polyglot=False):
    """
    生成测试运行说明的提示词片段。

    三种场景的处理逻辑：
      1. swerepo=True：SWE-bench 仓库任务
         - 从 eval_script 中提取测试命令
         - 提示词告诉 agent 必须严格使用指定的命令选项
      2. polyglot=True：Polyglot 多语言编程任务
         - eval_script 本身就是完整的测试命令（支持 C++/Go/Java/JS/Python/Rust）
         - 直接在 prompt 中以代码块展示
      3. 两者均 False：DGM 自测模式（agent 在修改自身代码后自测）
         - 固定使用 pytest 测试 /dgm/ 仓库中的 tools/ 和 utils/ 目录
         - 明确禁止测试 agentic_system.forward()（避免无限递归运行 agent）

    Args:
        eval_script (str): SWE-bench 评估脚本或 Polyglot 测试命令；DGM 自测时为空字符串。
        swerepo (bool): 是否为 SWE-bench 仓库任务。
        polyglot (bool): 是否为 Polyglot 多语言任务。

    Returns:
        str: 测试说明提示词片段（已 strip 首尾空白）。
    """
    # swerepo 和 polyglot 是互斥的（assert 保证调用正确性）
    assert not (swerepo and polyglot), "swerepo and polyglot cannot both be True"
    if swerepo:
        swe_prompt = '''The tests in the repository can be run with the bash command `{test_command}`. If no specific test files are provided, all tests will be run. The given command-line options must be used EXACTLY as specified. Do not use any other command-line options. {test_hint}'''
        test_command, test_hint = get_test_command(eval_script)
        description = swe_prompt.format(test_command=test_command, test_hint=test_hint)
    elif polyglot:
        # Polyglot 任务的测试命令直接以代码块展示（可能是 Makefile 命令或 shell 脚本）
        description = f"In the repository folder, the tests can be run with the following bash command(s):\n\n```{eval_script}```\n"
    else:
        # DGM 自测：仅测试工具和工具函数，禁止运行 agentic_system.forward()
        description = 'The tests in the repository can be run with the bash command `cd /dgm/ && pytest -rA <specific test files>`. If no specific test files are provided, all tests will be run. The given command-line options must be used EXACTLY as specified. Do not use any other command-line options. ONLY test tools and utils. NEVER try to test or run agentic_system.forward().'

    return description.strip()
