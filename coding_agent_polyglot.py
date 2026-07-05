import argparse
import subprocess
import logging
from logging.handlers import RotatingFileHandler
import os
import threading

from llm_withtools import CLAUDE_MODEL, OPENAI_MODEL, chat_with_agent
from utils.git_utils import diff_versus_commit, reset_to_commit, apply_patch


# ============================================================
# coding_agent_polyglot.py
# Polyglot 多语言版本的 coding agent（与 coding_agent.py 的 SWE-bench 版本对应）。
#
# 主要差异（相比 coding_agent.py）：
#   1. 支持多种编程语言任务（C++/Go/Java/JavaScript/Python/Rust），
#      通过 language 参数区分
#   2. 预定义各语言的测试命令（TEST_COMMANDS 字典），
#      用于在 Docker 容器内运行对应语言的测试套件
#   3. self_improve 模式下使用 OPENAI_MODEL（o3 系列），
#      而 SWE-bench 版本始终使用 CLAUDE_MODEL
#   4. get_current_edits() 返回的格式不同：
#      此版本返回格式化的 user 消息 dict，SWE-bench 版本返回原始 diff 字符串
#
# 使用场景：Polyglot 编程竞赛式任务（给定问题+测试，实现多语言解法），
# 而非真实 GitHub 项目的 bug 修复（SWE-bench）。
# ============================================================


# 各语言的标准测试命令（在 Docker 容器内运行）
# 注意：每条命令是一个 list（传给 subprocess/container.exec_run 时无需 shell 解析）

NPM_TEST_COMMANDS = [
    # JavaScript/Node.js：先建立 node_modules 软链（避免重复安装），再运行测试
    ["sh", "-c", "set -e"],
    ["sh", "-c", "[ ! -e node_modules ] && ln -s /npm-install/node_modules ."],
    ["sh", "-c", "[ ! -e package-lock.json ] && ln -s /npm-install/package-lock.json ."],
    ["sed", "-i", "s/\\bxtest(/test(/g", "*.spec.js"],  # 启用被跳过的测试（xtest → test）
    ["npm", "run", "test"]
]

CPP_TEST_COMMANDS = [
    # C++：cmake 配置（启用所有测试）+ make 编译
    ["sh", "-c", "set -e"],
    ["sh", "-c", "[ ! -d \"build\" ] && mkdir build"],
    ["sh", "-c", "cd build"],
    ["cmake", "-DEXERCISM_RUN_ALL_TESTS=1", "-G", "Unix Makefiles", ".."],
    ["make"],
    ["sh", "-c", "cd ../"]
]

TEST_COMMANDS = {
    "python": [["pytest", "-rA", "--tb=long"]],
    "rust": [["cargo", "test", "--", "--include-ignored"]],   # 包含被 #[ignore] 标记的测试
    "go": [["go", "test", "./..."]],                           # 递归测试所有子包
    "javascript": NPM_TEST_COMMANDS,
    "cpp": CPP_TEST_COMMANDS,
    "java": [["./gradlew", "test"]],                           # Gradle 构建系统
}


# 线程本地存储：并行评估时每个线程独立的 logger
thread_local = threading.local()


def get_thread_logger():
    """获取当前线程对应的 logger 实例。"""
    return getattr(thread_local, 'logger', None)


def set_thread_logger(logger):
    """将 logger 存入当前线程的本地存储。"""
    thread_local.logger = logger


def setup_logger(log_file='./chat_history.md', level=logging.INFO):
    """
    为当前线程创建轮转文件日志器（与 coding_agent.py 中的实现相同）。

    Args:
        log_file (str): 日志文件路径。
        level (int): 日志级别。

    Returns:
        logging.Logger: 配置好的线程私有 logger 实例。
    """
    logger = logging.getLogger(f'AgenticSystem-{threading.get_ident()}')
    logger.setLevel(level)
    logger.handlers = []

    file_formatter = logging.Formatter('%(message)s')

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)

    logger.addHandler(file_handler)
    set_thread_logger(logger)

    return logger


def safe_log(message, level=logging.INFO):
    """线程安全日志输出，无 logger 时降级到 print。"""
    logger = get_thread_logger()
    if logger:
        logger.log(level, message)
    else:
        print(f"Warning: No logger found for thread {threading.get_ident()}")


class AgenticSystem:
    """
    Polyglot 多语言任务的 agent 系统。

    与 SWE-bench 版本的核心区别：
      - 接受 language 参数（决定测试命令）
      - self_improve 时使用 OPENAI_MODEL（o3 具有更强的多语言推理能力）
      - get_current_edits() 返回格式化的消息 dict（可直接追加到 msg_history）
    """

    def __init__(
            self,
            problem_statement,
            git_tempdir,
            base_commit,
            chat_history_file='./chat_history.md',
            test_description=None,
            self_improve=False,
            language='python'
        ):
        """
        初始化 Polyglot AgenticSystem。

        Args:
            problem_statement (str): 编程任务描述。
            git_tempdir (str): 任务仓库在容器内的路径。
            base_commit (str): 基准 commit hash（用于计算 diff）。
            chat_history_file (str): agent 对话日志输出路径。
            test_description (str | None): 测试运行说明。
            self_improve (bool): 是否为自改进模式。
            language (str): 任务的编程语言（影响测试命令选择）。
        """
        self.problem_statement = problem_statement
        self.git_tempdir = git_tempdir
        self.base_commit = base_commit
        self.chat_history_file = chat_history_file
        self.test_description = test_description
        self.self_improve = self_improve
        self.language = language

        # 关键区别：self_improve 时用 o3（推理强），普通任务用 Claude（工具调用强）
        self.code_model = CLAUDE_MODEL if not self_improve else OPENAI_MODEL

        self.logger = setup_logger(chat_history_file)

        with open(chat_history_file, 'w') as f:
            f.write('')

    def get_current_edits(self):
        """
        获取当前 diff 并封装为 user 消息格式（可直接追加到 msg_history）。

        与 SWE-bench 版本的区别：
          SWE-bench 版返回原始 diff 字符串；
          此版本返回格式化的 user 消息 dict，
          便于后续将 diff 作为对话上下文传给 agent。

        Returns:
            list[dict]: 包含单条 user 消息（展示当前 diff）的消息历史列表。
        """
        diff = str(diff_versus_commit(self.git_tempdir, self.base_commit))
        new_msg_history = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"# Current Repo Edits\n{diff}",
                    }
                ],
            }
        ]
        return new_msg_history

    def forward(self):
        """
        Polyglot agent 的主运行入口：向 LLM 发送编程任务，agent 自主完成实现。

        与 SWE-bench 版本相比，prompt 更简洁（不需要 test_description 字段），
        聚焦于"分析问题 → 修改代码"的两步流程。

        多语言支持：agent 通过 tools 中的 bash 工具可以执行任何语言的编译/测试命令，
        语言相关的测试命令通过 test_description 传入（在 polyglot/harness.py 中生成）。
        """
        task = f"""I have uploaded a code repository in the directory {self.git_tempdir}. Help solve the following problem.

<problem_description>
{self.problem_statement}
</problem_description>

Your task is to make changes to the files in the {self.git_tempdir} directory to address the <problem_description>. I have already taken care of the required dependencies.
"""
        instruction = f"{task}\n\nPlease analyze the problem description carefully. Then make edits to the code files to complete the instruction."
        init_edit = chat_with_agent(instruction, model=self.code_model, msg_history=[], logging=safe_log)


def main():
    """
    命令行入口：在 Docker 容器内被 polyglot/harness.py 调用。

    多了 --language 参数（区分 C++/Go/Java/JS/Python/Rust），
    其余逻辑与 coding_agent.py 的 main() 相同。
    """
    parser = argparse.ArgumentParser(description='Process repository with an agentic system.')
    parser.add_argument('--problem_statement', required=True, help='The problem statement to process')
    parser.add_argument('--git_dir', required=True, help='Path to git repository directory')
    parser.add_argument('--base_commit', required=True, help='Base commit hash to compare against')
    parser.add_argument('--chat_history_file', required=True, help='Path to chat history file')
    parser.add_argument('--outdir', required=False, default="/dgm/", help='Output directory')
    parser.add_argument('--test_description', default=None, required=False, help='Description of how to test the repository')
    parser.add_argument('--self_improve', default=False, action='store_true', help='Whether to self-improve the repository or solving swe')
    parser.add_argument('--language', required=False, default="python", choices=['cpp', 'java', 'python', 'go', 'rust', 'javascript'], help='Task\'s programming language')
    args = parser.parse_args()

    agentic_system = AgenticSystem(
        problem_statement=args.problem_statement,
        git_tempdir=args.git_dir,
        base_commit=args.base_commit,
        chat_history_file=args.chat_history_file,
        test_description=args.test_description,
        self_improve=args.self_improve,
        language=args.language,
    )

    agentic_system.forward()

    # 保存修改结果为 model_patch.diff
    model_patch = diff_versus_commit(args.git_dir, args.base_commit)
    model_patch_outfile = os.path.join(args.outdir, 'model_patch.diff') if args.outdir else 'model_patch.diff'
    with open(model_patch_outfile, 'w') as f:
        f.write(model_patch)


if __name__ == "__main__":
    main()
