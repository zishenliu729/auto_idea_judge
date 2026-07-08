# Code adapted from https://github.com/SakanaAI/AI-Scientist/blob/main/ai_scientist/llm.py.
import json
import os
import re

import anthropic
import backoff
import openai


# ============================================================
# llm.py
# 底层 LLM 客户端封装，提供跨多个 AI 提供商的统一调用接口。
#
# 支持的提供商：
#   - Anthropic 直接 API（claude-* 模型）
#   - Amazon Bedrock（bedrock/ 前缀，Claude 系列）
#   - Vertex AI（vertex_ai/ 前缀，Claude 系列）
#   - OpenAI（gpt-4o-*、o1-*、o3-* 模型）
#   - DeepSeek（deepseek-* 模型，使用 OpenAI 兼容 API）
#   - OpenRouter（llama3.1-405b 等，通过 OpenAI 兼容接口）
#
# 核心函数：
#   - create_client(model)：根据模型名前缀自动选择并创建对应的 API 客户端
#   - get_response_from_llm(...)：单次对话，处理各模型的消息格式差异
#   - get_batch_responses_from_llm(...)：获取 N 个独立回复（用于集成/采样）
#   - extract_json_between_markers(...)：从 LLM 输出中安全提取 JSON 块
# ============================================================


# 所有输出的最大 token 数（输出 token 限制，不影响上下文窗口大小）
MAX_OUTPUT_TOKENS = 4096

# 已知支持的所有模型 ID 列表（供外部代码引用）
AVAILABLE_LLMS = [
    # Anthropic 直接 API
    "claude-3-5-sonnet-20240620",
    "claude-3-5-sonnet-20241022",
    # OpenAI
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "o1-preview-2024-09-12",
    "o1-mini-2024-09-12",
    "o1-2024-12-17",
    "o3-mini-2025-01-31",
    # OpenRouter（通过 OpenAI 兼容接口）
    "llama3.1-405b",
    # Amazon Bedrock 上的 Claude 模型
    "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
    "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
    "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
    "bedrock/anthropic.claude-3-opus-20240229-v1:0",
    "bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    # Vertex AI 上的 Claude 模型
    "vertex_ai/claude-3-opus@20240229",
    "vertex_ai/claude-3-5-sonnet@20240620",
    "vertex_ai/claude-3-5-sonnet-v2@20241022",
    "vertex_ai/claude-3-sonnet@20240229",
    "vertex_ai/claude-3-haiku@20240307",
    # DeepSeek（使用 OpenAI 兼容接口，但指向 deepseek.com）
    "deepseek-chat",
    "deepseek-coder",
    "deepseek-reasoner",
    # XHS MaaS Qwen judge backbone（OpenAI-compatible endpoint with api-key header）
    "maas/Qwen3.5-397B-A17B-FP8",
    "Qwen3.5-397B-A17B-FP8",
]


def create_client(model: str):
    """
    根据模型名称创建并返回对应的 LLM 客户端（client）和实际使用的模型名（client_model）。

    路由规则（按前缀匹配）：
      - "claude-"      → anthropic.Anthropic()（直接 API）
      - "bedrock" + claude → anthropic.AnthropicBedrock()（需要 AWS 凭证环境变量）
      - "vertex_ai" + claude → anthropic.AnthropicVertex()（需要 GCP 认证）
      - "gpt"、"o1-"、"o3-" → openai.OpenAI()
      - "deepseek-"   → openai.OpenAI(base_url="https://api.deepseek.com")
      - "llama3.1-"   → openai.OpenAI(base_url="https://openrouter.ai/api/v1")

    Bedrock/VertexAI 的 client_model：
      这两个平台的模型 ID 格式为 "bedrock/xxx" 或 "vertex_ai/xxx"，
      实际 API 调用时只需要 "xxx" 部分（split("/")[-1]）。

    Args:
        model (str): 模型标识符，来自 AVAILABLE_LLMS 列表。

    Returns:
        tuple[Any, str]: (client 实例, 实际传给 API 的模型名)。

    Raises:
        ValueError: 模型名不匹配任何已知前缀时抛出。
    """
    if model.startswith("claude-"):
        print(f"Using Anthropic API with model {model}.")
        return anthropic.Anthropic(), model
    elif model.startswith("bedrock") and "claude" in model:
        # Bedrock 模型 ID 格式："bedrock/anthropic.claude-xxx"，取最后一段
        client_model = model.split("/")[-1]
        print(f"Using Amazon Bedrock with model {client_model}.")
        client = anthropic.AnthropicBedrock(
            aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_region=os.getenv("AWS_REGION_NAME"),
        )
        return client, client_model
    elif model.startswith("vertex_ai") and "claude" in model:
        # Vertex AI 模型 ID 格式："vertex_ai/claude-xxx@version"，取最后一段
        client_model = model.split("/")[-1]
        print(f"Using Vertex AI with model {client_model}.")
        return anthropic.AnthropicVertex(), client_model
    elif 'gpt' in model or model.startswith("o1-") or model.startswith("o3-"):
        print(f"Using OpenAI API with model {model}.")
        return openai.OpenAI(), model
    elif model.startswith("deepseek-"):
        print(f"Using OpenAI API with {model}.")
        # DeepSeek 使用 OpenAI 兼容接口，但有独立的 API key 和 base_url
        client = openai.OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com"
        )
        return client, model
    elif model.startswith("maas/") or model.startswith("Qwen3.5-"):
        # Qwen judge backbone (2026-07-07): keep the final evaluator on a fixed
        # open-source model via XHS MaaS. The secret stays outside code in MAAS_API_KEY.
        client_model = model.split("/", 1)[1] if model.startswith("maas/") else model
        base_url = os.getenv(
            "MAAS_BASE_URL",
            "https://maas.devops.xiaohongshu.com/dqaservice-cmtagent-397b/v1",
        )
        api_key = os.getenv("MAAS_API_KEY") or os.getenv("QWEN_API_KEY")
        if not api_key:
            raise ValueError("MAAS_API_KEY or QWEN_API_KEY is required for Qwen MaaS evaluation.")
        print(f"Using XHS MaaS Qwen with model {client_model}.")
        client = openai.OpenAI(
            api_key="dummy",
            base_url=base_url,
            default_headers={"api-key": api_key},
        )
        return client, client_model
    elif model == "llama3.1-405b":
        print(f"Using OpenAI API with {model}.")
        client = openai.OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1"
        ), model
    else:
        raise ValueError(f"Model {model} not supported.")


@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError))
def get_batch_responses_from_llm(
        msg,
        client,
        model,
        system_message,
        print_debug=False,
        msg_history=None,
        temperature=0.75,
        n_responses=1,
):
    """
    从 LLM 获取 N 个独立回复（用于结果集成/多次采样）。

    为什么需要批量回复：
      DGM 的 coding_agent 可能对同一个 issue 采样多次，
      选出得分最高的 patch（由 score_tie_breaker 裁决）。
      某些模型（如 gpt-4o）支持原生的 n 参数一次返回多个回复；
      不支持的模型（Claude、o1 等）通过循环调用 get_response_from_llm 模拟。

    注意：此函数使用 @backoff.on_exception 仅处理 OpenAI 的限流/超时，
    Anthropic 的限流处理在 get_response_from_llm 中。

    Args:
        msg (str): 用户消息。
        client: 由 create_client 返回的 API 客户端。
        model (str): 模型 ID。
        system_message (str): 系统提示词。
        print_debug (bool): 是否打印调试信息。
        msg_history (list | None): 历史消息列表，None 表示新对话。
        temperature (float): 采样温度（越高越随机），默认 0.75。
        n_responses (int): 期望获取的独立回复数量，默认 1。

    Returns:
        tuple[list[str], list[list]]:
            - content: N 个回复字符串的列表
            - new_msg_history: 对应的 N 条消息历史（每条是独立的完整对话历史）
    """
    if msg_history is None:
        msg_history = []

    if model in [
        "gpt-4o-2024-05-13",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06",
    ]:
        # gpt-4o 系列：原生支持 n 参数，一次 API 调用返回多个回复
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_OUTPUT_TOKENS,
            n=n_responses,
            stop=None,
            seed=0,
        )
        content = [r.message.content for r in response.choices]
        # 为每个回复构建独立的消息历史
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]
    elif model == "llama-3-1-405b-instruct":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model="meta-llama/llama-3.1-405b-instruct",
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_OUTPUT_TOKENS,
            n=n_responses,
            stop=None,
        )
        content = [r.message.content for r in response.choices]
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]
    else:
        # 其他模型（Claude、o1 等）不支持 n 参数，循环调用 get_response_from_llm
        content, new_msg_history = [], []
        for _ in range(n_responses):
            c, hist = get_response_from_llm(
                msg,
                client,
                model,
                system_message,
                print_debug=False,
                msg_history=None,  # 每次独立调用，不共享历史
                temperature=temperature,
            )
            content.append(c)
            new_msg_history.append(hist)

    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history[0]):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content, new_msg_history


@backoff.on_exception(
    backoff.expo,
    (openai.RateLimitError, openai.APITimeoutError, anthropic.RateLimitError, anthropic.APIStatusError),
    max_time=120,
)
def get_response_from_llm(
        msg,
        client,
        model,
        system_message,
        print_debug=False,
        msg_history=None,
        temperature=0.7,
):
    """
    向 LLM 发送单条消息并获取回复，统一处理各模型的 API 格式差异。

    各模型的消息格式差异：
      - Claude：使用 client.messages.create()，消息格式为 content blocks（list of dict），
        system 在顶层参数中指定（非 messages 列表的一部分）
      - gpt-4o：使用 chat.completions.create()，system 作为 {"role": "system", ...} 消息
      - o1/o3：不支持独立的 system 消息（参数不接受 system_message），
        将 system_message 拼在 user 消息最前面；temperature 固定为 1
      - deepseek-reasoner：额外返回 reasoning_content（内部推理链）
      - llama3.1-*：通过 OpenRouter 的 OpenAI 兼容接口调用

    @backoff 装饰器：遇到限流或超时时，按指数退避自动重试（最多 120 秒）。
    涵盖 OpenAI 和 Anthropic 两类错误，防止瞬时 API 错误导致整个评估失败。

    Args:
        msg (str): 用户消息内容。
        client: 由 create_client 返回的 API 客户端实例。
        model (str): 实际调用的模型 ID。
        system_message (str): 系统提示词（各模型处理方式不同）。
        print_debug (bool): 是否打印对话内容（调试用）。
        msg_history (list | None): 历史消息；None 时从新对话开始。
        temperature (float): 采样温度，默认 0.7；o1/o3 模型固定为 1。

    Returns:
        tuple[str, list]: (回复文本, 包含新消息的完整历史列表)。
    """
    if msg_history is None:
        msg_history = []

    if "claude" in model:
        # Claude 格式：content 为 list[{type, text}]，system 在顶层
        new_msg_history = msg_history + [
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
        response = client.messages.create(
            model=model,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=temperature,
            system=system_message,
            messages=new_msg_history,
        )
        content = response.content[0].text
        new_msg_history = new_msg_history + [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": content,
                    }
                ],
            }
        ]
    elif model.startswith("gpt-4o-"):
        # gpt-4o 格式：system 作为单独的 role=system 消息
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_OUTPUT_TOKENS,
            n=1,
            stop=None,
            seed=0,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif model.startswith("o1-") or model.startswith("o3-"):
        # o1/o3 不支持 system role，将 system_message 拼接到 user 消息最前面
        # temperature 必须为 1（o1 系列的推理模型不接受其他值）
        new_msg_history = msg_history + [{"role": "user", "content": system_message + msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                *new_msg_history,
            ],
            temperature=1,
            n=1,
            seed=0,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif model in ["deepseek-chat", "deepseek-coder"]:
        # DeepSeek 聊天/编程模型：标准 OpenAI 兼容格式
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_OUTPUT_TOKENS,
            n=1,
            stop=None,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif model.startswith("Qwen3.5-"):
        # Qwen judge backbone (2026-07-07): thinking is intentionally enabled by
        # default because SoundnessBench is a reasoning-heavy judge task. Keep a
        # large enough token budget so reasoning_content does not crowd out content.
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        enable_thinking = os.getenv("MAAS_ENABLE_THINKING", "true").lower() not in {"0", "false", "no"}
        max_tokens = int(os.getenv("MAAS_MAX_OUTPUT_TOKENS", "8192"))
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
            stop=None,
            extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
        )
        message = response.choices[0].message
        content = message.content
        if content is None:
            reasoning_tokens = getattr(response.usage, "reasoning_tokens", None)
            raise ValueError(
                "Qwen MaaS returned no final content. Increase MAAS_MAX_OUTPUT_TOKENS "
                f"or set MAAS_ENABLE_THINKING=false. reasoning_tokens={reasoning_tokens}"
            )
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif model in ["deepseek-reasoner"]:
        # DeepSeek-R1 推理模型：不需要温度参数，额外返回推理链（reasoning_content）
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            n=1,
            stop=None,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
        # reasoning_content 是内部推理链，目前代码接收但未使用（可用于调试）
        reasoning_content = response.choices[0].message.reasoning_content
    elif model.startswith("llama3.1-"):
        # Llama3.1 通过 OpenRouter 接口调用，模型 ID 需转换为 OpenRouter 格式
        llama_size = model.split("-")[-1]  # 如 "405b"
        client_model = f"meta-llama/llama-3.1-{llama_size}-instruct"
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=client_model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_OUTPUT_TOKENS,
            n=1,
            stop=None,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
        resoning_content = response.choices[0].message.reasoning_content
    else:
        raise ValueError(f"Model {model} not supported.")

    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        print(f'User: {new_msg_history[-2]["content"]}')
        print(f'Assistant: {new_msg_history[-1]["content"]}')
        print("*" * 21 + " LLM END " + "*" * 21)
        print()
    return content, new_msg_history


def extract_json_between_markers(llm_output):
    """
    从 LLM 输出中提取 ```json ... ``` 代码块内的 JSON 内容并解析。

    两阶段解析策略：
      阶段 1（主路径）：逐行扫描 LLM 输出，找到 ```json 标记后收集内容，
        遇到 ``` 结束标记时停止，将收集到的行拼接后 json.loads。
      阶段 2（兜底路径）：若找不到 ```json 块（某些模型不加代码块标记），
        用正则 r"\{.*?\}" 在全文中搜索最小化 JSON 对象字符串，
        尝试逐个解析，成功则返回。

    控制字符清理：JSON 字符串中若含有 \x00-\x1F 或 \x7F 等控制字符
    （某些模型可能在输出中插入），会导致 json.loads 失败。
    失败后尝试用 re.sub 清除控制字符再解析（第二次机会）。

    Args:
        llm_output (str): LLM 的原始输出文本（可能包含 markdown 代码块）。

    Returns:
        dict | list | None: 成功解析的 Python 对象；解析失败时返回 None。
    """
    inside_json_block = False
    json_lines = []

    for line in llm_output.split('\n'):
        striped_line = line.strip()

        if striped_line.startswith("```json"):
            inside_json_block = True
            continue

        if inside_json_block and striped_line.startswith("```"):
            # 遇到关闭的 ``` 标记，结束收集
            inside_json_block = False
            break

        if inside_json_block:
            json_lines.append(line)

    if not json_lines:
        # 兜底：用正则在全文中找 JSON 对象
        fallback_pattern = r"\{.*?\}"
        matches = re.findall(fallback_pattern, llm_output, re.DOTALL)
        for candidate in matches:
            candidate = candidate.strip()
            if candidate:
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    # 去除控制字符后再试一次
                    candidate_clean = re.sub(r"[\x00-\x1F\x7F]", "", candidate)
                    try:
                        return json.loads(candidate_clean)
                    except json.JSONDecodeError:
                        continue
        return None

    json_string = "\n".join(json_lines).strip()

    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        # 去除控制字符后再试一次（针对模型输出中混入控制字符的情况）
        json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
        try:
            return json.loads(json_string_clean)
        except json.JSONDecodeError:
            return None
