from pathlib import Path
import subprocess


# ============================================================
# tools/edit.py
# 文件编辑工具：coding_agent 用于查看文件、创建新文件、覆写现有文件的接口。
#
# 三个命令：
#   - view：查看文件内容（带行号）或目录结构（两层深度）
#   - create：创建新文件（目标路径不能已存在）
#   - edit：覆写已有文件的完整内容（全量替换，无增量编辑）
#
# 设计选择：edit 命令采用"全量覆写"而非"行范围编辑"，
# 是因为 LLM 更容易生成完整文件内容而非精确的行范围 diff，
# 减少了偏移量计算错误的风险。
# ============================================================


def tool_info():
    """
    返回符合 Claude 工具调用 API 格式的工具描述。

    input_schema 约束了 LLM 必须提供的参数：
      - command（必需）：enum 限制只能是 view/create/edit
      - path（必需）：绝对路径字符串
      - file_text（可选）：create/edit 时提供文件内容，view 时不需要
    """
    return {
        "name": "editor",
        "description": """Custom editing tool for viewing, creating, and editing files\n
* State is persistent across command calls and discussions with the user.\n
* If `path` is a file, `view` displays the entire file with line numbers. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep.\n
* The `create` command cannot be used if the specified `path` already exists as a file.\n
* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`.\n
* The `edit` command overwrites the entire file with the provided `file_text`.\n
* No partial/line-range edits or partial viewing are supported.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["view", "create", "edit"],
                    "description": "The command to run: `view`, `create`, or `edit`."
                },
                "path": {
                    "description": "Absolute path to file or directory, e.g. `/repo/file.py` or `/repo`.",
                    "type": "string"
                },
                "file_text": {
                    "description": "Required parameter of `create` or `edit` command, containing the content for the entire file.",
                    "type": "string"
                }
            },
            "required": ["command", "path"]
        }
    }


def maybe_truncate(content: str, max_length: int = 10000) -> str:
    """
    若内容超过最大长度，截断并附加 '<response clipped>' 标记。

    防止 LLM 因超长文件内容超出 context window 而崩溃。
    10000 字符约等于 ~2500 tokens，是个比较保守的阈值。

    Args:
        content (str): 原始文件内容。
        max_length (int): 截断阈值，默认 10000 字符。

    Returns:
        str: 可能被截断的内容字符串。
    """
    if len(content) > max_length:
        return content[:max_length] + "\n<response clipped>"
    return content


def validate_path(path: str, command: str) -> Path:
    """
    验证路径合法性（必须是绝对路径）并检查与命令的约束关系。

    三种命令的路径约束：
      - view：路径必须存在（文件或目录均可）
      - create：路径必须不存在（不允许覆盖已有文件）
      - edit：路径必须存在且必须是文件（不允许"编辑"目录）

    Args:
        path (str): 路径字符串（要求绝对路径，以 '/' 开头）。
        command (str): 操作命令（view/create/edit）。

    Returns:
        pathlib.Path: 验证通过后的 Path 对象。

    Raises:
        ValueError: 路径不符合约束时抛出，错误信息会被 tool_function 包装返回给 agent。
    """
    path_obj = Path(path)

    # 必须是绝对路径（LLM 有时会生成相对路径，需要明确拒绝）
    if not path_obj.is_absolute():
        raise ValueError(
            f"The path {path} is not an absolute path (must start with '/')."
        )

    if command == "view":
        if not path_obj.exists():
            raise ValueError(f"The path {path} does not exist.")
    elif command == "create":
        # create 不允许覆盖已有文件（避免意外数据丢失）
        if path_obj.exists():
            raise ValueError(f"Cannot create new file; {path} already exists.")
    elif command == "edit":
        if not path_obj.exists():
            raise ValueError(f"The file {path} does not exist.")
        if path_obj.is_dir():
            raise ValueError(f"{path} is a directory and cannot be edited as a file.")
    else:
        raise ValueError(f"Unknown or unsupported command: {command}")

    return path_obj


def format_output(content: str, path: str, init_line: int = 1) -> str:
    """
    将文件内容格式化为带行号的输出（模拟 cat -n 效果）。

    带行号的输出让 LLM 在引用特定行时有明确的坐标，
    便于后续的 edit 操作精确描述修改位置（尽管 edit 是全量覆写，
    view 时的行号帮助 LLM 理解文件结构）。

    Args:
        content (str): 文件的原始文本内容。
        path (str): 文件路径（显示在输出标题中）。
        init_line (int): 起始行号，默认 1。

    Returns:
        str: 带行号的格式化输出字符串。
    """
    content = maybe_truncate(content)
    content = content.expandtabs()  # 把 tab 展开为空格，保证对齐一致
    numbered_lines = [
        f"{i + init_line:6}\t{line}"
        for i, line in enumerate(content.split("\n"))
    ]
    return f"Here's the result of running `cat -n` on {path}:\n" + "\n".join(numbered_lines) + "\n"


def read_file(path: Path) -> str:
    """
    读取文件的完整内容。

    Args:
        path (pathlib.Path): 已验证存在的文件路径。

    Returns:
        str: 文件文本内容。

    Raises:
        ValueError: 读取失败时（权限不足、编码错误等）抛出。
    """
    try:
        return path.read_text()
    except Exception as e:
        raise ValueError(f"Failed to read file: {e}")


def write_file(path: Path, content: str):
    """
    将 content 完整写入文件（覆盖原有内容）。

    Args:
        path (pathlib.Path): 目标文件路径。
        content (str): 要写入的完整文件内容。

    Raises:
        ValueError: 写入失败时（权限不足、磁盘已满等）抛出。
    """
    try:
        path.write_text(content)
    except Exception as e:
        raise ValueError(f"Failed to write file: {e}")


def view_path(path_obj: Path) -> str:
    """
    查看文件内容或目录结构。

    - 目录：用 find 命令列出最多 2 层深度的非隐藏文件/目录
    - 文件：读取全文并格式化为带行号输出

    Args:
        path_obj (pathlib.Path): 已验证存在的路径（文件或目录）。

    Returns:
        str: 格式化的文件内容或目录列表。
    """
    if path_obj.is_dir():
        # 目录视图：排除隐藏文件（以 '.' 开头），深度限制为 2 层
        try:
            result = subprocess.run(
                ['find', str(path_obj), '-maxdepth', '2', '-not', '-path', '*/\\.*'],
                capture_output=True,
                text=True
            )
            if result.stderr:
                return f"Error listing directory: {result.stderr}"
            return (
                f"Here's the files and directories up to 2 levels deep in {path_obj}, excluding hidden items:\n"
                + result.stdout
            )
        except Exception as e:
            raise ValueError(f"Failed to list directory: {e}")

    # 文件视图：带行号的完整内容
    content = read_file(path_obj)
    return format_output(content, str(path_obj))


def tool_function(command: str, path: str, file_text: str = None) -> str:
    """
    编辑工具的主入口，分发到具体操作。

    所有异常都在此捕获并转换为错误字符串返回给 agent，
    不向上抛出（保证工具调用的健壮性，让 agent 能基于错误信息自行调整）。

    Args:
        command (str): 操作类型（view/create/edit）。
        path (str): 目标文件/目录的绝对路径。
        file_text (str | None): create/edit 时提供的文件内容，view 时为 None。

    Returns:
        str: 操作结果描述（成功信息、文件内容、或错误信息）。
    """
    try:
        path_obj = validate_path(path, command)

        if command == "view":
            return view_path(path_obj)

        elif command == "create":
            if file_text is None:
                raise ValueError("Missing required `file_text` for 'create' command.")
            write_file(path_obj, file_text)
            return f"File created successfully at: {path}"

        elif command == "edit":
            if file_text is None:
                raise ValueError("Missing required `file_text` for 'edit' command.")
            write_file(path_obj, file_text)
            return f"File at {path} has been overwritten with new content."

        else:
            raise ValueError(f"Unknown command: {command}")

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    # 示例用法
    result = tool_function("view", "./coding_agent.py", view_range=[1, 10])
    print(result)
