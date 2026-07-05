import os


# ============================================================
# prompts/tooluse_prompt.py
# 为不支持原生工具调用的 LLM（如 llama、旧版模型）生成工具使用提示词。
#
# 背景：Claude 和 o3 有原生的 tool_choice API，可以自动解析工具调用；
# 但其他模型（如 llama3.1-405b）没有这个功能，需要在 prompt 中
# 直接嵌入工具的源代码和使用格式，让 LLM 通过文本输出 <tool_use> 标签来"调用"工具。
#
# check_for_tool_use() 在 llm_withtools.py 中会检测 <tool_use> 标签
# 并解析出工具名和参数，实现手动工具调用流程（manual tool calling）。
# ============================================================


def get_tooluse_prompt():
    """
    动态生成工具使用说明提示词，包含当前所有可用工具的源代码。

    为什么嵌入源代码而非仅描述接口：
      不支持原生工具调用的 LLM 需要从源代码中理解工具的功能和参数，
      直接提供源代码比 JSON schema 更直观，也更符合这类模型的训练数据特征。

    工具发现方式：扫描 tools/ 目录下的所有 .py 文件（排除 __init__.py），
    与 load_all_tools() 的发现逻辑保持一致。

    调用格式说明：
      LLM 需要在回复中输出如下格式触发工具调用：
        <tool_use>
        {'tool_name': 'bash', 'tool_input': {'command': 'ls -la'}}
        </tool_use>
      llm_withtools.py 的 check_for_tool_use() 用正则提取并解析此标签。

    Returns:
        str: 包含所有工具源代码和使用格式说明的完整提示词字符串。
    """
    # 定位 tools/ 目录（相对于当前 prompts/ 文件的父目录）
    tool_folder = os.path.join(os.path.dirname(__file__), '../tools')
    tool_files = [
        os.path.join(tool_folder, file)
        for file in os.listdir(tool_folder)
        if file.endswith('.py') and file != '__init__.py'
    ]
    # 读取每个工具的源代码，用 Python 代码块格式包装
    tool_file_contents = [open(file).read().strip() for file in tool_files]
    tools_available = [f"```python\n{tool_content}\n```" for tool_content in tool_file_contents]
    tools_available = '\n\n'.join(tools_available)
    # 构造完整的工具使用提示词（包含工具列表和调用格式示例）
    tooluse_prompt = """Here are the available tools:
{tools_available}

Use the available tools in this format:
```
<tool_use>
{{
    'tool_name': ...,
    'tool_input': ...
}}
</tool_use>
```
""".format(tools_available=tools_available)
    return tooluse_prompt.strip()
