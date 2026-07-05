import asyncio
import os


# ============================================================
# tools/bash.py
# Bash 工具：coding_agent 在 Docker 容器内执行 shell 命令的核心接口。
#
# 关键设计：持久化 bash 会话（BashSession）
#   - 一次性启动一个长运行的 /bin/bash 进程（而非每条命令单独 fork）
#   - 保持进程间的状态（工作目录、环境变量、已定义的 shell 变量等）
#   - 用哨兵字符串（sentinel）检测命令执行完毕，避免轮询 EOF 或依赖进程退出
# ============================================================


def tool_info():
    """
    返回符合 Claude 工具调用 API 格式的工具描述。

    此描述会被发送给 LLM，LLM 依据此描述理解如何调用 bash 工具。
    'description' 字段中的限制说明（无网络、持久状态等）会直接影响 LLM 的使用策略。
    """
    return {
        "name": "bash",
        "description": """Run commands in a bash shell\n
* When invoking this tool, the contents of the "command" parameter does NOT need to be XML-escaped.\n
* You don't have access to the internet via this tool.\n
* You do have access to a mirror of common linux and python packages via apt and pip.\n
* State is persistent across command calls and discussions with the user.\n
* To inspect a particular line range of a file, e.g. lines 10-25, try 'sed -n 10,25p /path/to/the/file'.\n
* Please avoid commands that may produce a very large amount of output.\n
* Please run long lived commands in the background, e.g. 'sleep 10 &' or start a server in the background.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to run."
                }
            },
            "required": ["command"]
        }
    }


class BashSession:
    """
    维护一个持久化的交互式 bash 进程会话。

    为什么不每次命令都新开子进程：
      cd、export、函数定义等 shell 操作会改变进程状态，若每次都新开进程
      这些状态就会丢失（cd 到新目录后下一条命令仍在原目录）。
      持久化会话让 agent 能够像在真实终端中一样累积状态。

    哨兵机制（sentinel = "<<exit>>"）：
      bash 进程不会主动告知"命令执行完毕"，stdout 也不会自动关闭。
      解决方案：在每条命令后追加 `; echo '<<exit>>'`，
      当我们在 stdout 中看到 "<<exit>>" 时，就知道命令已经执行完毕了。
      这比等待 EOF 或轮询更可靠，也不会影响命令的 exit code。

    直接读取内部缓冲区（_buffer）：
      asyncio 的子进程默认是流式读取（await stdout.read()），
      但那会阻塞等待换行符或 EOF。
      这里绕过标准 API，直接访问 `_buffer` 字节缓冲区，
      配合 `_output_delay` 的轮询，在不阻塞的情况下实时读取输出。
    """

    def __init__(self):
        self._started = False
        self._process = None
        self._timed_out = False
        self._timeout = 120.0        # 单条命令的最长等待时间（秒）
        self._sentinel = "<<exit>>"  # 命令完成的标记字符串
        self._output_delay = 0.2     # 每次轮询缓冲区的间隔（秒），避免 CPU 空转

    async def start(self):
        """
        启动后台 bash 进程（幂等：已启动则跳过）。

        -i：交互式模式，使 bash 加载 .bashrc 等配置文件，
        保证与手动终端行为一致（某些工具依赖 shell 别名或 PATH 设置）。
        preexec_fn=os.setsid：把 bash 放入独立进程组，
        确保后续 terminate() 能清理整个进程树（包括 bash 启动的子进程）。
        """
        if self._started:
            return
        self._process = await asyncio.create_subprocess_shell(
            "/bin/bash -i",
            preexec_fn=os.setsid,  # 独立进程组，便于后续整体终止
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy()  # 继承当前环境变量（PATH、HOME 等）
        )
        self._started = True

    def stop(self):
        """终止 bash 进程并重置状态。若进程已退出则直接清理引用。"""
        if not self._started:
            return
        if self._process.returncode is None:
            self._process.terminate()
        self._process = None
        self._started = False

    async def run(self, command):
        """
        在持久化 bash 进程中执行一条命令，等待执行完毕后返回输出。

        执行流程：
          1. 把命令写入 stdin，末尾追加 `; echo '<<exit>>'`
          2. 轮询 stdout 内部缓冲区，每次等待 _output_delay 秒
          3. 发现哨兵字符串时截取哨兵之前的内容作为输出
          4. 清空缓冲区（防止残留数据污染下一条命令的输出）

        Args:
            command (str): 要执行的 bash 命令字符串。

        Returns:
            tuple[str, str]: (stdout 输出, stderr 输出)，均已 strip 首尾空白。

        Raises:
            ValueError: 会话未启动、进程已退出、或执行超时时抛出。
        """
        if not self._started:
            raise ValueError("Session has not started.")
        if self._process.returncode is not None:
            raise ValueError(f"Bash has exited with returncode {self._process.returncode}")
        if self._timed_out:
            raise ValueError(
                f"Timed out: bash has not returned in {self._timeout} seconds and must be restarted."
            )

        # 向 bash stdin 写入命令 + 哨兵（; 确保即使命令失败也会输出哨兵）
        self._process.stdin.write(
            command.encode() + f"; echo '{self._sentinel}'\n".encode()
        )
        await self._process.stdin.drain()  # 刷新写缓冲区，确保数据发送到 bash

        try:
            output = ''
            start_time = asyncio.get_event_loop().time()

            while True:
                # 超时检测：超过 _timeout 秒则标记为超时并抛出异常
                if asyncio.get_event_loop().time() - start_time > self._timeout:
                    self._timed_out = True
                    raise ValueError(
                        f"Timed out: bash has not returned in {self._timeout} seconds and must be restarted."
                    )

                # 轮询：等待 _output_delay 秒后检查缓冲区
                await asyncio.sleep(self._output_delay)
                # 直接访问 asyncio 流的内部字节缓冲区（非公开 API，但避免了阻塞式读取）
                stdout_data = self._process.stdout._buffer.decode(errors='ignore')
                stderr_data = self._process.stderr._buffer.decode(errors='ignore')

                if self._sentinel in stdout_data:
                    # 找到哨兵：截取哨兵之前的内容（哨兵本身不包含在输出中）
                    output = stdout_data[: stdout_data.index(self._sentinel)]
                    break

            # 清空缓冲区，防止残留数据污染下一次 run() 的输出
            self._process.stdout._buffer.clear()
            self._process.stderr._buffer.clear()

            output = output.strip()
            error = stderr_data.strip()

            return output, error

        except Exception as e:
            self._timed_out = True
            raise ValueError(str(e))


def filter_error(error):
    """
    过滤掉不必要的 stderr 噪音，只保留真正有价值的错误信息。

    "Inappropriate ioctl for device" 是 bash -i（交互式模式）在非 TTY 环境
    （如 subprocess pipe）中运行时产生的警告，与实际命令执行无关，
    会干扰 agent 对错误的判断，需要过滤掉。

    过滤逻辑：
      发现 "Inappropriate ioctl for device" 所在行后，跳过接下来 3 行
      （这几行通常是 ioctl 相关的 bash 提示信息），
      然后继续保留后续的真实错误行。

    Args:
        error (str): 原始 stderr 字符串。

    Returns:
        str: 过滤后的 stderr 字符串。
    """
    filtered_lines = []
    i = 0
    error_lines = error.splitlines()
    while i < len(error_lines):
        line = error_lines[i]

        if "Inappropriate ioctl for device" in line:
            # 跳过 ioctl 错误及紧跟的 3 行 bash 提示
            i += 3
            # 如果接下来是哨兵行，再跳一行
            if '<<exit>>' in error_lines[i]:
                i += 1
            # 保留后续的真实错误行
            while i < len(error_lines) - 1:
                filtered_lines.append(error_lines[i])
                i += 1
            i += 1
            continue

        filtered_lines.append(line)
        i += 1
    return '\n'.join(filtered_lines).strip()


async def tool_function_call(command):
    """
    异步执行 bash 命令的核心实现。

    每次调用都创建新的 BashSession（一个独立 bash 进程）。
    这与"持久化会话"的设计看似矛盾，但此处 BashSession 的生命周期
    仅限于单次工具调用——状态持久化发生在 agent 的"一轮对话"中，
    每次 tool_function 调用都是一个单独的 asyncio.run() 上下文。

    Args:
        command (str): bash 命令字符串。

    Returns:
        str: stdout 输出 + stderr 输出（若有错误）；异常时返回错误描述。
    """
    try:
        bash_session = BashSession()

        if not bash_session._started:
            await bash_session.start()

        output, error = await bash_session.run(command)
        error = filter_error(error)
        result = ""
        if output:
            result += output
        if error:
            result += "\nError:\n" + error
        return result.strip()
    except Exception as e:
        return f"Error: {str(e)}"


def tool_function(command):
    """
    同步包装器：将异步的 tool_function_call 包装为同步调用接口。

    tools/__init__.py 的 load_all_tools() 期望 tool_function 是同步可调用的
    （被 process_tool_call() 直接调用，无 await），
    此函数通过 asyncio.run() 在新事件循环中运行异步逻辑。
    """
    return asyncio.run(tool_function_call(command))


if __name__ == "__main__":
    # 示例用法：直接从命令行运行 bash 工具
    import sys

    if len(sys.argv) < 2:
        print("Usage: python bash.py '<command>'")
    else:
        input_command = ' '.join(sys.argv[1:])
        result = tool_function(input_command)
        print(result)
