import json


# ============================================================
# common_utils.py
# 最基础的 IO 工具函数，供整个 dgm 项目各模块调用。
# 职责：统一封装文件读取和 JSON 加载，避免各处重复写 open/json.load。
# ============================================================


def read_file(file_path):
    """
    读取文件的完整内容并以字符串返回。

    注意：调用 strip() 去掉首尾空白（换行符等），
    这样调用方拿到的是干净的内容，不需要自己处理。

    Args:
        file_path (str): 文件的路径（相对或绝对路径均可）。

    Returns:
        str: 文件的完整文本内容，已去除首尾空白。
    """
    with open(file_path, 'r') as f:
        content = f.read().strip()
    return content

def load_json_file(file_path):
    """
    从文件加载 JSON 数据并返回 Python 对象（dict 或 list）。

    与直接 json.loads(read_file(...)) 相比，这里用 json.load(file)
    流式解析，对大文件更友好（不需要先把整个文件读入内存再解析）。

    Args:
        file_path (str): JSON 文件的路径。

    Returns:
        dict | list: JSON 解析后的 Python 对象，具体类型取决于文件内容。

    Raises:
        FileNotFoundError: 文件不存在时抛出。
        json.JSONDecodeError: 文件内容不是合法 JSON 时抛出。
    """
    with open(file_path, 'r') as file:
        return json.load(file)
