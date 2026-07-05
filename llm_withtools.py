import ast
import json
import re
import anthropic
import backoff
import openai
import copy

from llm import create_client, get_response_from_llm
from prompts.tooluse_prompt import get_tooluse_prompt
from tools import load_all_tools


# ============================================================
# llm_withtools.py
# 支持工具调用（Tool Use）的 LLM 封装层——DGM 中所有 agent 对话的核心入口。
#
# 整体架构：
#   三条并行路径，对应三类模型：
#   1. Claude（Bedrock）：原生工具调用 API（client.messages.create + tool_choice）
#   2. o3（OpenAI Responses API）：responses.create + parallel_tool_calls=False
#   3. 其他模型（无原生工具支持）：在 prompt 中嵌入工具说明，解析 <tool_use> 标签
#
# 与 llm.py 的关系：
#   - llm.py：纯文本对话（无工具），用于 diagnose_problem 等高层推理调用
#   - llm_withtools.py：带工具的 agent 对话，用于 coding_agent 实际执行任务
#
# 关键变量：
#   CLAUDE_MODEL：Bedrock 上的 Claude 3.5 Sonnet（工具调用强，适合编码任务）
#   OPENAI_MODEL：o3-mini（推理强，适合多语言任务的 self_improve 模式）
# ============================================================


# 两个全局模型常量：coding_agent 根据 self_improve 参数选择使用哪个
CLAUDE_MODEL = 'bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0'
OPENAI_MODEL = 'o3-mini-2025-01-31'


def process_tool_call(tools_dict, tool_name, tool_input):
    """
    根据工具名查找并执行对应的工具函数。

    工具字典格式：{name: {'info': ..., 'function': callable}}
    tool_input 是从 LLM 响应中提取的参数 dict，直接以 kwargs 方式传入。

    Args:
        tools_dict (dict): {tool_name: tool_dict} 映射（来自 load_all_tools）。
        tool_name (str): LLM 请求调用的工具名。
        tool_input (dict): 传给工具函数的参数。

    Returns:
        str: 工具执行结果（成功）或错误信息字符串（失败）。
    """
    try:
        if tool_name in tools_dict:
            return tools_dict[tool_name]['function'](**tool_input)
        else:
            return f"Error: Tool '{tool_name}' not found"
    except Exception as e:
        return f"Error executing tool '{tool_name}': {str(e)}"


@backoff.on_exception(
    backoff.expo,
    (openai.RateLimitError, openai.APITimeoutError, anthropic.RateLimitError, anthropic.APIStatusError),
    max_time=600,   # 最多重试 10 分钟（比 llm.py 的 120s 更长，因为工具调用链更耗时）
    max_value=60,   # 单次重试最大等待 60 秒
)
def get_response_withtools(
    client, model, messages, tools, tool_choice,
    logging=None, max_retry=3
):
    """
    向支持工具调用的 LLM 发送请求，处理 Claude 与 o3 的 API 差异。

    Claude vs o3 API 的关键差异：
      - Claude：client.messages.create()，tool_choice 为 {"type": "auto"} 形式的 dict
      - o3：client.responses.create()，tool_choice 为 "auto" 字符串，
            必须设置 parallel_tool_calls=False（o3 不支持并行工具调用）

    重试机制：
      - @backoff 处理限流和超时（外层自动重试，无需手动）
      - 内部 max_retry 处理其他临时异常（如网络抖动）
      - 遇到 "Input is too long" 则直接 re-raise，由调用方处理上下文截断

    Args:
        client: 由 create_client() 返回的 API 客户端。
        model (str): 实际传给 API 的模型名。
        messages (list): 完整消息历史。
        tools (list): 当前可用工具列表（已经过 convert_tool_info 格式转换）。
        tool_choice: Claude 用 dict，o3 用字符串 "auto"。
        logging: 日志函数（safe_log 或 print）。
        max_retry (int): 内部重试次数，默认 3。

    Returns:
        API 响应对象（Claude 的 Message 或 o3 的 Response）。
    """
    try:
        if 'claude' in model:
            response = client.messages.create(
                model=model,
                messages=messages,
                max_tokens=4096,
                tool_choice=tool_choice,
                tools=tools,
            )
        elif model.startswith('o3-'):
            response = client.responses.create(
                model=model,
                input=messages,
                tool_choice=tool_choice,
                tools=tools,
                parallel_tool_calls=False   # o3 不支持并行工具调用，强制串行
            )
            response = response
        else:
            raise ValueError(f"Unsupported model: {model}")
        return response
    except Exception as e:
        logging(f"Error in get_response_withtools: {str(e)}")
        if max_retry > 0:
            return get_response_withtools(client, model, messages, tools, tool_choice, logging, max_retry - 1)

        # 上下文长度超限：记录日志但不做任何处理，由调用方捕获
        if 'Input is too long for requested model' in str(e):
            pass

        raise  # 内部重试耗尽后向上抛出，由 @backoff 或调用方处理


def check_for_tool_use(response, model=''):
    """
    从 LLM 响应中提取工具调用信息（如果有的话）。

    三种提取方式对应三类模型：
      1. Claude：stop_reason == "tool_use" → 找 block.type == "tool_use"
      2. o3：遍历 response.output，找 type == "function_call" 的条目
      3. 其他（手动工具调用）：在字符串响应中用正则找 <tool_use>...</tool_use>，
         然后用 ast.literal_eval 解析（比 json.loads 更宽松，允许 Python 字典格式）

    Returns:
        dict | None: 包含 tool_id、tool_name、tool_input 的 dict；无工具调用时返回 None。
    """
    if 'claude' in model:
        # Claude，检查 stop_reason
        if response.stop_reason == "tool_use":
            tool_use_block = next(block for block in response.content if block.type == "tool_use")
            return {
                'tool_id': tool_use_block.id,
                'tool_name': tool_use_block.name,
                'tool_input': tool_use_block.input,
            }

    elif model.startswith('o3-'):
        # o3，遍历 response.output 找 function_call 类型
        for tool_call in response.output:
            if tool_call.type == "function_call":
                break

        if tool_call:
            return {
                'tool_id': tool_call.call_id,
                'tool_name': tool_call.name,
                'tool_input': json.loads(tool_call.arguments),
            }

    else:
        # 无原生工具支持的模型，在文本响应中查找 <tool_use> 标签
        pattern = r'<tool_use>(.*?)</tool_use>'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            tool_use_str = match.group(1).strip()
            try:
                # ast.literal_eval 更宽松：允许 Python 字典格式（单引号、无引号 key 等）
                tool_use_dict = ast.literal_eval(tool_use_str)
                if isinstance(tool_use_dict, dict) and 'tool_name' in tool_use_dict and 'tool_input' in tool_use_dict:
                    return tool_use_dict
            except Exception:
                pass

    # 没有工具调用
    return None


def convert_tool_info(tool_info, model=None):
    """
    将 Claude 格式的 tool_info 转换为目标模型所需的格式。

    Claude 格式（原始格式，各工具的 tool_info() 返回此格式）：
      {'name': ..., 'description': ..., 'input_schema': {...}}

    o3 格式差异：
      1. 外层增加 'type': 'function'
      2. input_schema → parameters（字段名变化）
      3. 增加 'strict': True（o3 要求严格模式）
      4. additionalProperties: False 递归添加到每个含 properties 的层级
      5. 所有参数变为 required（o3 严格模式要求），可选参数通过 ["type", "null"] 联合类型表达

    Args:
        tool_info (dict): 标准 Claude 格式的工具信息。
        model (str): 目标模型 ID。

    Returns:
        dict: 适配目标模型格式的工具信息。
    """
    if 'claude' in model:
        # Claude 格式不变，直接透传
        return {
            'name': tool_info['name'],
            'description': tool_info['description'],
            'input_schema': tool_info['input_schema'],
        }
    elif model.startswith('o3-'):
        def add_additional_properties(d):
            """递归在每个含 properties 的 dict 层级中添加 additionalProperties: False"""
            if isinstance(d, dict):
                if 'properties' in d:
                    d['additionalProperties'] = False
                for k, v in d.items():
                    add_additional_properties(v)
        add_additional_properties(tool_info['input_schema'])

        # o3 严格模式要求所有参数必须在 required 中
        # 对于原本可选的参数，用 [原类型, "null"] 联合类型表达"可空"语义
        for p in tool_info['input_schema']['properties'].keys():
            if not p in tool_info['input_schema']['required']:
                tool_info['input_schema']['required'].append(p)
                t = copy.deepcopy(tool_info['input_schema']['properties'][p]["type"])
                if isinstance(t, str):
                    tool_info['input_schema']['properties'][p]["type"] = [t, "null"]
                elif isinstance(t, list):
                    tool_info['input_schema']['properties'][p]["type"] = t + ["null"]

        return {
            'type': 'function',
            'name': tool_info['name'],
            'description': tool_info['description'],
            'parameters': tool_info['input_schema'],  # o3 用 parameters 而非 input_schema
            "strict": True,
        }
    else:
        return tool_info


def convert_block_claude(block):
    """
    将 Claude 响应中的单个 content block 转换为通用文本格式。

    为什么需要转换：
      Claude 的 content 是结构化的 block 列表（text/tool_use/tool_result），
      而其他模型使用纯文本。将 Claude 格式展平为文本，使消息历史可以
      跨模型复用（cross-model msg_history）。

    转换规则：
      - text block → 直接取 text
      - tool_use block → 转为 <tool_use> 标签文本（与手动工具调用格式一致）
      - tool_result block → 转为 "Tool Result: ..." 文本
      - 其他未知类型 → str(block)

    Args:
        block: Claude content block（dict 或 Anthropic SDK 对象）。

    Returns:
        dict: {"type": "text", "text": ...} 格式的通用文本 block。
    """
    if isinstance(block, dict):
        block_type = block.get('type')
        text = block.get('text')
        tool_name = block.get('name')
        tool_input = block.get('input')
        tool_result = block.get('content')
    else:
        block_type = getattr(block, 'type', None)
        text = getattr(block, 'text', None)
        tool_name = getattr(block, 'name', None)
        tool_input = getattr(block, 'input', None)
        tool_result = getattr(block, 'content', None)

    text = text or ""

    if block_type == "text":
        return {
            "type": "text",
            "text": text
        }
    elif block_type == "tool_use":
        # 转为手动工具调用格式，使消息历史可以被无原生工具支持的模型理解
        return {
            "type": "text",
            "text": f"<tool_use>\n{{'tool_name': {tool_name}, 'tool_input': {tool_input}}}\n</tool_use>"
        }
    elif block_type == "tool_result":
        return {
            "type": "text",
            "text": f"Tool Result: {tool_result}"
        }
    else:
        return {
            "type": "text",
            "text": str(block)
        }


def convert_msg_history_claude(msg_history):
    """
    将 Claude 原生消息历史（含 content blocks）转换为通用文本格式。

    通用格式：每条消息的 content 是 [{"type": "text", "text": ...}] 列表。
    通过 convert_block_claude 逐 block 处理，将结构化内容展平为文本。

    Args:
        msg_history (list): Claude 格式的消息历史（role + content blocks 列表）。

    Returns:
        list: 通用格式的消息历史（可被无原生工具支持的模型直接使用）。
    """
    new_msg_history = []

    for msg in msg_history:
        role = msg.get('role', '')
        content_blocks = msg.get('content', [])
        new_content = []

        for block in content_blocks:
            new_content.append(convert_block_claude(block))

        new_msg_history.append({
            "role": role,
            "content": new_content
        })

    return new_msg_history


def convert_msg_history_openai(msg_history):
    """
    将 o3 原生消息历史（Responses API 格式）转换为通用文本格式。

    o3 的消息格式更复杂：
      - tool 角色的消息 → 转为 user 角色，内容前缀 "Tool Result: "
      - 带 tool_calls 的 assistant 消息 → 转为 <tool_use> 文本格式
      - 普通 assistant/user 消息 → 包装成 content block 格式

    Args:
        msg_history (list): o3 Responses API 格式的消息历史（可能是 dict 或 SDK 对象）。

    Returns:
        list: 通用格式的消息历史。
    """
    new_msg_history = []

    for msg in msg_history:
        if isinstance(msg, dict):
            role = msg.get('role', '')
            content = msg.get('content', '')

            if role == 'tool':
                # o3 的 tool 角色消息 → 转为 user 消息（通用格式没有 tool 角色）
                new_msg = {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Tool Result: {content}",
                        }
                    ],
                }
            else:
                new_msg = {
                    "role": role,
                    "content": content,
                }
        else:
            # SDK 对象，使用 getattr 访问属性
            role = getattr(msg, 'role', None)
            content = getattr(msg, 'content', None)
            tool_calls = getattr(msg, 'tool_calls', None)

            if tool_calls:
                # 带工具调用的 assistant 消息 → 转为 <tool_use> 文本格式
                tool_call = tool_calls[0]
                function_name = getattr(tool_call.function, 'name', '')
                function_args = getattr(tool_call.function, 'arguments', '')
                new_msg = {
                    "role": role,
                    "content": [
                        {
                            "type": "text",
                            "text": f"<tool_use>\n{{'tool_name': {function_name}, 'tool_input': {function_args}}}\n</tool_use>",
                        }
                    ],
                }
            else:
                new_msg = {
                    "role": role,
                    "content": [
                        {
                            "type": "text",
                            "text": content,
                        }
                    ],
                }

        new_msg_history.append(new_msg)

    return new_msg_history


def convert_msg_history(msg_history, model=None):
    """
    根据模型类型，将模型特定格式的消息历史转换为通用格式。

    这个"通用格式"是 DGM 的内部约定：
      每条消息 content 是 [{"type": "text", "text": ...}] 列表，
      工具调用以 <tool_use> 文本标签表示，工具结果以 "Tool Result: " 前缀表示。

    当 convert=True 时，chat_with_agent 会调用此函数，
    使得跨模型共享消息历史成为可能（但目前 o3 路径尚未完全支持跨模型转换）。

    Args:
        msg_history (list): 待转换的消息历史。
        model (str): 当前使用的模型 ID。

    Returns:
        list: 转换后的通用格式消息历史。
    """
    if 'claude' in model:
        return convert_msg_history_claude(msg_history)
    elif model.startswith('o3-'):
        return convert_msg_history_openai(msg_history)
    else:
        return msg_history


def chat_with_agent_manualtools(msg, model, msg_history=None, logging=print):
    """
    为无原生工具支持的模型提供工具调用能力（通过 prompt 嵌入 + 文本解析）。

    工作原理：
      1. 在 system_message 中嵌入所有工具的源码和 <tool_use> 格式说明
         （来自 get_tooluse_prompt()）
      2. 通过普通文本对话（get_response_from_llm）与模型交互
      3. 在每次响应中用正则解析 <tool_use>...</tool_use> 标签
      4. 执行工具，将结果以 "Tool Result: ..." 格式反馈给模型
      5. 循环直到无更多工具调用

    适用场景：deepseek、llama 等无原生工具支持的模型。

    注意：异常被完全吞掉（except: pass），这是为了确保 agent 流程不因单次失败中断，
    代价是错误静默化，调试时需查看日志。

    Args:
        msg (str): 用户指令。
        model (str): 模型 ID。
        msg_history (list): 历史消息。
        logging: 日志函数。

    Returns:
        list: 更新后的消息历史（包含本次对话）。
    """
    if msg_history is None:
        msg_history = []
    # system_message 包含工具说明，让模型知道如何使用工具
    system_message = f'You are a coding agent.\n\n{get_tooluse_prompt()}'
    new_msg_history = msg_history

    try:
        all_tools = load_all_tools(logging=logging)
        tools_dict = {tool['info']['name']: tool for tool in all_tools}

        client, client_model = create_client(model)

        logging(f"Input: {msg}")
        response, new_msg_history = get_response_from_llm(
            msg=msg,
            client=client,
            model=client_model,
            system_message=system_message,
            print_debug=False,
            msg_history=new_msg_history,
        )
        logging(f"Output: {response}")

        # 工具调用循环：解析 → 执行 → 反馈 → 再解析
        tool_use = check_for_tool_use(response, model=client_model)
        while tool_use:
            tool_name = tool_use['tool_name']
            tool_input = tool_use['tool_input']
            tool_result = process_tool_call(tools_dict, tool_name, tool_input)

            tool_msg = f'Tool Used: {tool_name}\nTool Input: {tool_input}\nTool Result: {tool_result}'
            logging(tool_msg)
            response, new_msg_history = get_response_from_llm(
                msg=tool_msg,
                client=client,
                model=client_model,
                system_message=system_message,
                print_debug=False,
                msg_history=new_msg_history,
            )
            logging(f"Output: {response}")

            tool_use = check_for_tool_use(response, model=client_model)

    except Exception:
        pass

    return new_msg_history


def chat_with_agent_claude(
        msg,
        model='bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0',
        msg_history=None,
        logging=print,
    ):
    """
    使用 Claude 原生工具调用 API 的 agent 对话函数。

    Claude 工具调用协议：
      1. 发送请求（tool_choice={"type": "auto"}）
      2. 若 stop_reason == "tool_use"：提取工具调用信息
      3. 将 Claude 的响应（含 tool_use block）作为 assistant 消息追加到历史
      4. 将工具执行结果作为 user 消息（tool_result 格式）追加到历史
      5. 继续发送，循环直到 stop_reason != "tool_use"

    消息格式要点：
      - assistant 消息的 content 直接用 response.content（SDK 对象列表）
      - user 消息（工具结果）格式：
        {"type": "tool_result", "tool_use_id": ..., "content": tool_result}

    Args:
        msg (str): 初始用户指令。
        model (str): Claude 模型 ID（含 'claude'）。
        msg_history (list): 已有的对话历史（用于多轮对话）。
        logging: 日志函数。

    Returns:
        list: 仅包含本次对话新增消息的历史（不含传入的 msg_history）。
    """
    if msg_history is None:
        msg_history = []
    # 将初始指令封装为 Claude 格式的 user 消息（content block 格式）
    new_msg_history = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": msg,
                }
            ],
        }
    ]

    try:
        client, client_model = create_client(model)

        all_tools = load_all_tools(logging=logging)
        tools_dict = {tool['info']['name']: tool for tool in all_tools}
        # 将工具信息转换为 Claude API 所需格式
        tools = [convert_tool_info(tool['info'], model=client_model) for tool in all_tools]

        response = get_response_withtools(
            client=client,
            model=client_model,
            messages=msg_history + new_msg_history,
            tool_choice={"type": "auto"},  # Claude 的 tool_choice 是 dict 格式
            tools=tools,
            logging=logging,
        )

        # Claude 工具调用循环
        tool_use = check_for_tool_use(response, model=client_model)
        while tool_use:
            tool_name = tool_use['tool_name']
            tool_input = tool_use['tool_input']
            tool_result = process_tool_call(tools_dict, tool_name, tool_input)

            # 将 Claude 响应（含 tool_use block）作为 assistant 消息追加
            new_msg_history.append({"role": "assistant", "content": response.content})
            # 将工具结果作为 user 消息（tool_result 格式）追加
            new_msg_history.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use['tool_id'],  # 必须与对应的 tool_use block id 匹配
                        "content": tool_result,
                    }
                ],
            })
            response = get_response_withtools(
                client=client,
                model=client_model,
                messages=msg_history + new_msg_history,
                tool_choice={"type": "auto"},
                tools=tools,
                logging=logging,
            )

            tool_use = check_for_tool_use(response, model=client_model)

        # 提取最终文本响应（跳过非 text 类型的 block）
        final_response = next((block.text for block in response.content if hasattr(block, "text")), None)
        new_msg_history.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": final_response,
                }
            ],
        })

    except Exception:
        pass

    return new_msg_history


def chat_with_agent_openai(
        msg,
        model='o3-mini-2025-01-31',
        msg_history=None,
        logging=print,
    ):
    """
    使用 o3 Responses API 的 agent 对话函数。

    o3 Responses API 与 Claude/chat.completions 的主要差异：
      - API：client.responses.create()（非 client.chat.completions.create）
      - 消息格式：user 消息的 content 用 "input_text" 类型（非 "text"）
      - tool_choice：字符串 "auto"（非 dict）
      - 工具结果格式：{"type": "function_call_output", "call_id": ..., "output": ...}
      - 将整个响应对象（非仅文本）追加到 new_msg_history

    注意：o3 不支持跨模型消息历史转换（当前版本 convert_msg_history 对 o3 路径注释掉了）。

    Args:
        msg (str): 初始用户指令。
        model (str): o3 模型 ID（以 'o3-' 开头）。
        msg_history (list): 已有的对话历史。
        logging: 日志函数。

    Returns:
        list: 仅包含本次对话新增消息的历史。
    """
    if msg_history is None:
        msg_history = []
    # o3 的 user 消息 content 用 "input_text" 类型（不同于 Claude 的 "text"）
    new_msg_history = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": msg,
                }
            ],
        }
    ]
    separator = '=' * 10
    logging(f"\n{separator} User Instruction {separator}\n{msg}")
    try:
        client, client_model = create_client(model)

        all_tools = load_all_tools(logging=logging)
        tools_dict = {tool['info']['name']: tool for tool in all_tools}
        # 转换为 o3 格式（strict 模式，additionalProperties: False 等）
        tools = [convert_tool_info(tool['info'], model=client_model) for tool in all_tools]

        # o3 Responses API：tool_choice 用字符串 "auto"
        response = get_response_withtools(
            client=client,
            model=client_model,
            messages=msg_history + new_msg_history,
            tool_choice="auto",
            tools=tools,
            logging=logging,
        )
        logging(f"\n{separator} Agent Response {separator}\n{response}")

        tool_use = check_for_tool_use(response, model=client_model)
        logging(tool_use)
        while tool_use:
            tool_name = tool_use['tool_name']
            tool_input = tool_use['tool_input']
            tool_result = process_tool_call(tools_dict, tool_name, tool_input)

            logging(f"Tool Used: {tool_name}")
            logging(f"Tool Input: {tool_input}")
            logging(f"Tool Result: {tool_result}")

            # 从响应的 output 列表中提取 function_call 对象追加到历史
            # （注意：追加的是 SDK 对象，不是 dict）
            for tool_call in response.output:
                if tool_call.type == "function_call":
                    break
            new_msg_history.append(tool_call)
            # 工具结果使用 "function_call_output" 格式（o3 Responses API 专属）
            new_msg_history.append({
                "type": "function_call_output",
                "call_id": tool_use['tool_id'],
                "output": tool_result,
            })
            response = get_response_withtools(
                client=client,
                model=client_model,
                messages=msg_history + new_msg_history,
                tool_choice="auto",
                tools=tools,
                logging=logging,
            )

            tool_use = check_for_tool_use(response, model=client_model)

            logging(f"Tool Response: {response}")

        # 将整个响应对象（非仅文本）追加到 new_msg_history（与 Claude 路径不同）
        new_msg_history.append(response)

    except Exception:
        pass

    return new_msg_history


def chat_with_agent(
    msg,
    model=CLAUDE_MODEL,
    msg_history=None,
    logging=print,
    convert=False,  # 是否将消息历史转换为通用格式（使 msg_history 可跨模型复用）
):
    """
    所有 coding agent 对话的统一入口，根据模型类型分发到对应的实现。

    三条分发路径：
      1. Claude（'claude' in model）：chat_with_agent_claude → 原生工具调用
      2. o3（model.startswith('o3-')）：chat_with_agent_openai → Responses API
      3. 其他：chat_with_agent_manualtools → 手动工具调用（prompt 嵌入）

    convert 参数的作用：
      当 convert=True 时，将本次对话生成的消息历史转换为通用格式，
      使得下次调用时可以换用不同的模型（跨模型对话复用历史）。
      目前 o3 路径尚未完全支持此功能（对应代码被注释掉）。

    注意：返回值是完整的消息历史（msg_history + new_msg_history），
    而 chat_with_agent_claude/openai 返回的是"仅本次新增"部分，
    此函数负责拼接。

    Args:
        msg (str): 用户指令。
        model (str): 模型 ID（决定使用哪条路径）。
        msg_history (list): 已有的历史消息（传入时不被修改）。
        logging: 日志函数。
        convert (bool): 是否将消息历史转换为通用格式。

    Returns:
        list: 完整的更新后消息历史（原历史 + 本次对话新增消息）。
    """
    if msg_history is None:
        msg_history = []

    if 'claude' in model:
        new_msg_history = chat_with_agent_claude(msg, model=model, msg_history=msg_history, logging=logging)
        conv_msg_history = convert_msg_history(new_msg_history, model=model)
        logging(conv_msg_history)
        if convert:
            # 使用转换后的通用格式（可跨模型复用）
            new_msg_history = conv_msg_history
        # 拼接原历史和新增消息
        new_msg_history = msg_history + new_msg_history

    elif model.startswith('o3-'):
        new_msg_history = chat_with_agent_openai(msg, model=model, msg_history=msg_history, logging=logging)
        # 当前版本 o3 暂不支持跨模型消息历史转换
        # new_msg_history = convert_msg_history(new_msg_history, model=model)
        new_msg_history = msg_history + new_msg_history

    else:
        # 无原生工具支持的模型路径
        new_msg_history = chat_with_agent_manualtools(msg, model=model, msg_history=msg_history, logging=logging)
        conv_msg_history = convert_msg_history(new_msg_history, model=model)
        if convert:
            new_msg_history = conv_msg_history

    return new_msg_history


if __name__ == "__main__":
    # 简单测试：验证工具调用功能是否正常
    msg = "hello!"
    chat_with_agent(msg)
