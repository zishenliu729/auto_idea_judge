from pathlib import Path
import importlib


# ============================================================
# tools/__init__.py
# 工具注册表的入口：动态扫描并加载 tools/ 目录下的所有工具模块。
#
# 设计约定：每个工具文件（bash.py、edit.py 等）必须实现两个接口：
#   - tool_info() → dict：返回符合 Claude 工具调用格式的工具描述（name、description、input_schema）
#   - tool_function：可调用对象（函数），接收工具参数并返回执行结果字符串
#
# coding_agent 调用 load_all_tools() 获取所有可用工具，
# llm_withtools.py 将这些工具注册到 LLM 的 tool_choice 中。
# ============================================================


def load_all_tools(logging=print):
    """
    动态扫描 tools/ 目录，加载所有满足接口约定的工具模块。

    为什么用动态导入而非手动 import 列表：
      新增工具时只需创建新的 .py 文件，无需修改任何注册代码，
      符合"开放-封闭原则"——对扩展开放，对修改封闭。

    加载流程：
      1. 扫描 tools/ 目录下所有 .py 文件（排除 __init__.py 自身）
      2. 用 importlib.import_module() 动态导入每个模块
      3. 检查是否实现了 tool_info 和 tool_function 接口
      4. 调用 tool_info() 获取元数据，并收集 tool_function 引用
      5. 任何模块导入失败都会向上抛出异常（fail-fast，避免工具残缺运行）

    Args:
        logging (callable): 日志输出函数，默认 print；
                            Docker 环境中会传入线程安全的 safe_log。

    Returns:
        list[dict]: 每个元素包含：
            - 'info': tool_info() 的返回值（name、description、input_schema）
            - 'function': tool_function 可调用对象
            - 'name': 工具文件名（不含 .py，如 'bash'、'editor'）
    """
    tools_dir = Path(__file__).parent  # tools/ 目录的绝对路径
    tools = []

    # 扫描所有 .py 文件，排除 __init__.py（它不是工具，是注册器本身）
    tool_files = [f for f in tools_dir.glob("*.py") if f.stem != "__init__"]

    for tool_file in tool_files:
        # 构造完整的模块名（如 "tools.bash"）用于 importlib 导入
        module_name = f"tools.{tool_file.stem}"
        try:
            module = importlib.import_module(module_name)

            # 检查模块是否满足约定的接口（缺失则报错，不做容错）
            if hasattr(module, 'tool_info') and hasattr(module, 'tool_function'):
                tools.append({
                    'info': module.tool_info(),       # 工具元数据（name、description 等）
                    'function': module.tool_function,  # 工具的实际执行函数（同步）
                    'name': tool_file.stem             # 工具文件名（如 'bash'、'editor'）
                })
            else:
                raise Exception(f"Tool module {module_name} does not have required functions.")
        except Exception as e:
            logging(f"Failed to import {module_name}: {e}")
            raise e  # 任何工具加载失败都向上抛出，让调用方感知

    return tools
