
import argparse
import subprocess
import logging
from logging.handlers import RotatingFileHandler
import os
import threading

from llm_withtools import CLAUDE_MODEL, OPENAI_MODEL, chat_with_agent
from utils.eval_utils import get_report_score, msg_history_to_report, score_tie_breaker
from utils.git_utils import diff_versus_commit, reset_to_commit, apply_patch


# ============================================================
# coding_agent.py
# SWE-bench 版本的 coding agent 核心实现。
#
# 架构：
#   - AgenticSystem：封装单次 issue 修复的完整流程
#     - forward()：主入口，给 LLM agent 提问，让其修改仓库代码
#     - get_regression_tests()：询问 agent 找出回归测试
#     - run_regression_tests()：让 agent 运行回归测试验证修复
#   - 线程安全日志：每个并行评估线程拥有独立的 logger（写入各自的 chat_history.md）
#
# 运行场景：
#   1. SWE-bench 问题修复（self_improve=False）：instance_id 为具体 issue ID
#   2. DGM 自改进（self_improve=True）：instance_id 固定为 'dgm'，修改自身代码
#
# 调用链：
#   swe_bench/harness.py → coding_agent.main()（在 Docker 容器内运行）
#   → AgenticSystem.forward() → chat_with_agent()（调用 LLM with tools）
#   → diff_versus_commit() 保存 model_patch.diff
# ============================================================


# 线程本地存储：每个并行评估线程各持有一个 logger，互不干扰
thread_local = threading.local()


def get_thread_logger():
    """
    获取当前线程对应的 logger 实例。

    Returns:
        logging.Logger | None: 当前线程的 logger，未初始化时返回 None。
    """
    return getattr(thread_local, 'logger', None)


def set_thread_logger(logger):
    """将 logger 存入当前线程的本地存储。"""
    thread_local.logger = logger


def setup_logger(log_file='./chat_history.md', level=logging.INFO):
    """
    为当前线程创建文件日志器（轮转文件，防止日志无限增长）。

    为什么用 RotatingFileHandler：
      chat_history.md 记录了完整的 agent 对话历史，包含所有工具调用和响应，
      对于长时间运行的 agent 可能非常大。轮转限制在 10MB/文件，最多 5 个备份。

    格式选择：file_formatter 只记录 %(message)s（不含时间戳和级别），
    因为 chat_history.md 本身就是对话记录的 markdown 格式，加时间戳会破坏可读性。

    Args:
        log_file (str): 日志文件路径（通常为 run_dir/<instance_id>/chat_history.md）。
        level (int): 日志级别，默认 INFO。

    Returns:
        logging.Logger: 配置好的线程私有 logger 实例。
    """
    # 用线程 ID 区分不同线程的 logger（避免并行运行时 logger 冲突）
    logger = logging.getLogger(f'AgenticSystem-{threading.get_ident()}')
    logger.setLevel(level)

    # 清除旧 handler（防止重复调用时日志被重复写入）
    logger.handlers = []

    # 日志格式仅保留消息本身（chat_history.md 是对话记录，不需要日志元数据）
    file_formatter = logging.Formatter('%(message)s')

    # 确保日志目录存在
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    # 轮转文件：单文件最大 10MB，保留最多 5 个备份（防止磁盘用尽）
    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)

    logger.addHandler(file_handler)

    # 存入线程本地存储，后续通过 safe_log() 调用
    set_thread_logger(logger)

    return logger


def safe_log(message, level=logging.INFO):
    """
    线程安全的日志输出。

    优先使用线程私有 logger；logger 未初始化时降级到 print（不抛异常）。
    此函数作为 chat_with_agent() 的 logging 回调传入，
    确保每个 agent 对话的输出写入正确的文件。

    Args:
        message: 日志消息（任意可 str() 的对象）。
        level (int): 日志级别，默认 INFO。
    """
    logger = get_thread_logger()
    if logger:
        logger.log(level, message)
    else:
        print(f"Warning: No logger found for thread {threading.get_ident()}")


class AgenticSystem:
    """
    封装单次 issue 修复（或自改进）流程的 agent 系统。

    核心设计：
      - forward() 是对 LLM 的单轮指令（send instruction → agent 自主使用工具修改代码 → 结束）
      - 不做显式的"修改→测试→修改"循环（agent 自己决定何时调用测试工具）
      - self_improve 模式下，instance_id 固定为 'dgm'，agent 修改的是自身代码（/dgm/ 目录）
    """

    def __init__(
            self,
            problem_statement,
            git_tempdir,
            base_commit,
            chat_history_file='./chat_history.md',
            test_description=None,
            self_improve=False,
            instance_id=None,
        ):
        """
        初始化 AgenticSystem。

        Args:
            problem_statement (str): 问题描述（SWE-bench 的 problem_statement 或自改进任务描述）。
            git_tempdir (str): 目标代码仓库在 Docker 容器内的路径（如 /testbed/）。
            base_commit (str): 基准 commit hash，用于计算 diff（model_patch）。
            chat_history_file (str): agent 对话日志的输出文件路径。
            test_description (str | None): 测试运行说明（由 testrepo_prompt.py 生成）。
            self_improve (bool): 是否为自改进模式（修改 /dgm/ 而非 /testbed/）。
            instance_id (str | None): SWE-bench 任务 ID；自改进时固定为 'dgm'。
        """
        self.problem_statement = problem_statement
        self.git_tempdir = git_tempdir
        self.base_commit = base_commit
        self.chat_history_file = chat_history_file
        self.test_description = test_description
        self.self_improve = self_improve
        # 自改进时 instance_id 固定为 'dgm'，用于 parse_eval_output 选择正确的日志解析器
        self.instance_id = instance_id if not self_improve else 'dgm'
        # 默认使用 Claude（工具调用能力强），o1/o3 用于 tie-breaker 等需要深度推理的场景
        self.code_model = CLAUDE_MODEL

        # 初始化线程私有 logger，写入 chat_history_file
        self.logger = setup_logger(chat_history_file)

        # 每次初始化清空日志文件（新的 agent 运行从头记录）
        with open(chat_history_file, 'w') as f:
            f.write('')

    def get_current_edits(self):
        """
        获取当前工作区相对于 base_commit 的完整 diff（包含未跟踪文件）。

        Returns:
            str: unified diff 格式的字符串，可用于生成 model_patch.diff。
        """
        diff = str(diff_versus_commit(self.git_tempdir, self.base_commit))
        return diff

    def get_regression_tests(self):
        """
        让 agent 自动识别仓库中的回归测试集（应在修复前后都通过的测试）。

        prompt 结构：
          1. 告诉 agent 仓库位置（git_tempdir）
          2. 提供问题描述（problem_statement）
          3. 提供测试运行说明（test_description）
          4. 要求 agent 列出回归测试的位置、内容和运行方式

        Returns:
            str: agent 的回归测试摘要文本（从消息历史最后一条提取）。
        """
        instruction = f"""I have uploaded a Python code repository in the directory {self.git_tempdir}.

<problem_description>
{self.problem_statement}
</problem_description>

<test_description>
{self.test_description}
</test_description>

Your task is to identify regression tests in the {self.git_tempdir} directory that should pass both before and after addressing the <problem_description>. I have already taken care of the required dependencies.
At the end, please provide a summary that includes where the regression tests are located, what they are testing, and how they can be executed.
"""

        new_msg_history = chat_with_agent(instruction, model=self.code_model, msg_history=[], logging=safe_log)
        regression_tests_summary = new_msg_history[-1]
        try:
            # 从 Claude 格式的消息中提取文本内容
            regression_tests_summary = regression_tests_summary['content'][-1]['text']
        except:
            pass
        return regression_tests_summary

    def run_regression_tests(self, regression_tests_summary):
        """
        让 agent 基于之前识别的回归测试，验证当前修改是否引入了回归。

        与 get_regression_tests() 不同，此函数在 agent 修改代码后调用，
        把当前 diff 和回归测试摘要一起提供给 agent，让其运行测试并报告结果。

        测试报告由 msg_history_to_report() 从消息历史中提取
        （扫描包含 "Tool Result:" 的最新 user 消息，解析为 {测试名: 状态} 映射）。

        Args:
            regression_tests_summary (str): get_regression_tests() 返回的测试摘要。

        Returns:
            dict[str, str]: {测试用例名: 状态} 映射（PASSED/FAILED/ERROR/SKIPPED）。
        """
        code_diff = self.get_current_edits()
        instruction = f"""I have uploaded a Python code repository in the directory {self.git_tempdir}. There is an attempt to address the problem statement. Please review the changes and run the regression tests.

<problem_description>
{self.problem_statement}
</problem_description>

<attempted_solution>
{code_diff}
</attempted_solution>

<test_description>
{self.test_description}
</test_description>

<regression_tests_summary>
{regression_tests_summary}
</regression_tests_summary>

Your task is to run the regression tests in the {self.git_tempdir} directory to ensure that the changes made to the code address the <problem_description>.
"""
        new_msg_history = chat_with_agent(instruction, model=self.code_model, msg_history=[], logging=safe_log)
        test_report = msg_history_to_report(self.instance_id, new_msg_history, model=self.code_model)
        return test_report

    def forward(self):
        """
        agent 的主运行入口：向 LLM 发送修复任务，agent 自主使用工具完成修改。

        单轮 instruction 设计：
          - 告诉 agent 仓库位置（git_tempdir）和问题描述（problem_statement）
          - 告诉 agent 如何运行测试（test_description）
          - 不做显式的修改→测试→修改循环；agent 自行决定是否运行测试

        修改结果通过 diff_versus_commit() 在 forward() 外部获取，
        保存为 model_patch.diff（由 main() 负责）。
        """
        instruction = f"""I have uploaded a Python code repository in the directory {self.git_tempdir}. Help solve the following problem.

<problem_description>
{self.problem_statement}
</problem_description>

<test_description>
{self.test_description}
</test_description>

Your task is to make changes to the files in the {self.git_tempdir} directory to address the <problem_description>. I have already taken care of the required dependencies.
"""
        new_msg_history = chat_with_agent(instruction, model=self.code_model, msg_history=[], logging=safe_log)


def main():
    """
    命令行入口：解析参数、运行 AgenticSystem、保存 model_patch.diff。

    此函数在 Docker 容器内被 swe_bench/harness.py 通过 subprocess 调用，
    接收 problem_statement、git_dir、base_commit 等参数，
    完成修复后将 diff 写入 outdir/model_patch.diff。
    """
    parser = argparse.ArgumentParser(description='Process repository with an agentic system.')
    parser.add_argument('--problem_statement', required=True, help='The problem statement to process')
    parser.add_argument('--git_dir', required=True, help='Path to git repository directory')
    parser.add_argument('--base_commit', required=True, help='Base commit hash to compare against')
    parser.add_argument('--chat_history_file', required=True, help='Path to chat history file')
    parser.add_argument('--outdir', required=False, default="/dgm/", help='Output directory')
    parser.add_argument('--test_description', default=None, required=False, help='Description of how to test the repository')
    parser.add_argument('--self_improve', default=False, action='store_true', help='Whether to self-improve the repository or solving swe')
    parser.add_argument('--instance_id', default=None, help='Instance ID for SWE issue')
    args = parser.parse_args()

    # 初始化并运行 agent
    agentic_system = AgenticSystem(
        problem_statement=args.problem_statement,
        git_tempdir=args.git_dir,
        base_commit=args.base_commit,
        chat_history_file=args.chat_history_file,
        test_description=args.test_description,
        self_improve=args.self_improve,
        instance_id=args.instance_id,
    )

    # agent 运行（修改仓库代码）
    agentic_system.forward()

    # 保存修改结果为 model_patch.diff（用于后续评估和进化历史记录）
    model_patch = diff_versus_commit(args.git_dir, args.base_commit)
    model_patch_outfile = os.path.join(args.outdir, 'model_patch.diff') if args.outdir else 'model_patch.diff'
    with open(model_patch_outfile, 'w') as f:
        f.write(model_patch)


if __name__ == "__main__":
    main()
