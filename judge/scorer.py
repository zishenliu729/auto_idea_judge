"""
judge/scorer.py
---------------
负责调用 LLM 对单个 SoundnessBench pair 打分，并提供并发批量打分入口。

在整个适配架构中的定位：
  - 是 evaluate.py 和 judge/ 模块之间的"执行层"
  - 接受外部传入的已切分 JSONL 数据，返回预测结果列表
  - 本文件本身也可以被 DGM agent 修改（如改变并发数、解析策略）

与 SoundnessBench rigorbench/evaluation/run.py 的关系：
  - 提取并保留：_strip_markdown_fences / _extract_first_json_object /
    _clamp_confidence / _parse_prediction / _format_experiments_for_eval / _pair_text
  - 不复用 run_evaluation()：原版依赖 eval.yaml 配置、内置 test split、
    snapshot 中间文件 IO，与 DGM 的使用方式（接受外部 JSONL、一次性出结果）不兼容
  - LLM 调用改为 DGM 自己的 llm.py（get_response_from_llm + create_client）：
    继承 DGM 已有的 backoff / Bedrock / VertexAI 配置，不需要新增 API key 依赖
  - normalize_bucket 和 _format_experiments_for_eval 均内联，
    避免对 rigorbench 包产生跨目录 import 依赖（使 judge/ 可独立使用）
"""

from __future__ import annotations

import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from judge.prompts import EvaluationMode, build_scoring_prompt

# llm.py 依赖 anthropic/openai 包（仅在 Docker 容器内安装），
# 采用惰性导入：只在实际调用 LLM 的函数内部才 import，
# 这样辅助函数（_parse_prediction 等）可以在容器外独立测试，不受依赖缺失影响。
def _import_llm():
    """惰性导入 DGM 的 llm.py，在首次 LLM 调用时执行。"""
    # judge/ 在 dgm/ 子目录，把 dgm/ 根目录加入 sys.path 才能 import llm.py
    dgm_root = os.path.join(os.path.dirname(__file__), "..")
    if dgm_root not in sys.path:
        sys.path.insert(0, dgm_root)
    from llm import get_response_from_llm, create_client  # noqa: PLC0415
    return get_response_from_llm, create_client


# ── 内联工具函数（原来在 rigorbench 各子包中，此处内联避免跨包依赖）─────────────

def _normalize_bucket(label: Any) -> str | None:
    """将模型输出的 bucket 标签归一化为 'low'/'high' 或 None（解析失败时）。

    内联自 rigorbench.buckets.normalize_bucket，逻辑完全一致。
    内联原因：使 judge/ 目录可以独立运行，不依赖 SoundnessBench 包的安装路径。
    """
    if label is None:
        return None
    value = str(label).strip().lower()
    return value if value in {"low", "high"} else None


def _format_experiments_for_eval(experiments: list[dict[str, Any]]) -> str:
    """将结构化的 experiment 列表格式化为 prompt 中可读的文本块。

    内联自 rigorbench.extraction.extract._format_experiments_for_eval。
    SoundnessBench 的 experiments 字段是 list[dict]，每个 dict 包含
    Description / Method / Evaluation Metrics 三个可选字段。
    """
    if not experiments:
        return ""
    parts: list[str] = []
    for i, exp in enumerate(experiments, 1):
        lines = [f"Experiment {i}:"]
        if exp.get("Description"):
            lines.append(f"  Description: {exp['Description']}")
        if exp.get("Method"):
            lines.append(f"  Method: {exp['Method']}")
        if exp.get("Evaluation Metrics"):
            lines.append(f"  Evaluation Metrics: {exp['Evaluation Metrics']}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def _strip_markdown_fences(text: str) -> str:
    """去除 LLM 输出中的 markdown 代码块标记（```json ... ```）。"""
    return re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()


def _extract_first_json_object(text: str) -> str | None:
    """用括号深度匹配提取第一个完整 JSON 对象字符串。

    比 regex 更健壮：能正确处理嵌套 JSON、justification 中含花括号的情况。
    """
    text = _strip_markdown_fences(text)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for idx in range(start, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _clamp_confidence(value: Any) -> int | None:
    """将置信度值限制在 [1, 5] 整数范围内，解析失败返回 None。"""
    try:
        return int(round(max(1.0, min(5.0, float(value)))))
    except (TypeError, ValueError):
        return None


def _parse_prediction(response: str) -> dict[str, Any]:
    """将模型输出文本解析为标准预测结构 {rigor_bucket, confidence, justification}。

    两阶段解析策略（与原版一致）：
    1. 主路径：_extract_first_json_object -> json.loads -> 取字段
    2. 兜底：json.loads 失败时用 regex 直接提取 rigor_bucket 和 confidence，
       应对模型在 justification 中插入未转义引号等导致 JSON 不合法的情况。
    """
    parsed: dict[str, Any] = {"rigor_bucket": None, "confidence": None, "justification": None}
    raw = _extract_first_json_object(response)
    if raw is None:
        return parsed
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        # JSON 解析失败：用 regex 兜底提取关键字段
        bucket_match = re.search(r'"(?:rigor_bucket|bucket)"\s*:\s*"([^"]+)"', raw, re.IGNORECASE)
        conf_match = re.search(r'"(?:confidence|reviewer_confidence)"\s*:\s*([-+]?\d+(?:\.\d+)?)', raw)
        if bucket_match:
            parsed["rigor_bucket"] = _normalize_bucket(bucket_match.group(1))
        if conf_match:
            parsed["confidence"] = _clamp_confidence(conf_match.group(1))
        return parsed

    parsed["rigor_bucket"] = _normalize_bucket(obj.get("rigor_bucket", obj.get("bucket")))
    parsed["confidence"] = _clamp_confidence(obj.get("confidence", obj.get("reviewer_confidence")))
    justification = obj.get("justification")
    if isinstance(justification, str) and justification.strip():
        parsed["justification"] = justification.strip()
    return parsed


def _pair_text(pair: dict[str, Any]) -> tuple[str, str]:
    """从 pair dict 中提取 (hypothesis_str, experiment_str) 两个字符串。

    SoundnessBench 数据结构：
    - hypothesis：short_hypothesis 字段（主要），hypothesis 字段（兼容旧格式）
    - experiment：experiments 是结构化 list，需 _format_experiments_for_eval 格式化；
      若无 experiments 则回退到 experiment 字符串字段（兼容旧格式）
    """
    hypothesis = str(pair.get("short_hypothesis") or pair.get("hypothesis") or "").strip()
    if pair.get("experiments"):
        experiment = _format_experiments_for_eval(pair["experiments"])
    else:
        experiment = str(pair.get("experiment") or "").strip()
    return hypothesis, experiment


# ── 核心打分函数 ──────────────────────────────────────────────────────────────

def score_pair(
    pair: dict[str, Any],
    client: Any,
    client_model: str,
    mode: EvaluationMode = "direct_bucket",
) -> dict[str, Any]:
    """对单个 SoundnessBench pair 调用 LLM 打分，返回 {rigor_bucket, confidence, justification}。

    与原版 score_pair_with_llm 的接口差异：
    - 原版接受 rigorbench LLMClient 对象（有 .chat() 方法）
    - 此处接受已由 create_client() 解包的 (client, client_model) 二元组，
      直接传给 get_response_from_llm，复用 DGM 的 backoff 和多模型路由逻辑。
    temperature=0.2：与原版一致，降低评分随机性，使结果更稳定可比。
    """
    hypothesis, experiment = _pair_text(pair)
    system_msg, user_msg = build_scoring_prompt(hypothesis, experiment, mode=mode)
    # 惰性导入 llm.py（仅在此处调用时才加载 anthropic/openai 依赖）
    get_response_from_llm, _ = _import_llm()
    # get_response_from_llm 签名：(msg, client, model, system_message, ...)
    response, _ = get_response_from_llm(
        user_msg, client, client_model, system_msg, temperature=0.2
    )
    return _parse_prediction(response)


def score_pairs_concurrent(
    pairs: list[dict[str, Any]],
    model: str,
    mode: EvaluationMode = "direct_bucket",
    max_workers: int = 8,
) -> list[dict[str, Any]]:
    """并发对多个 pair 打分，返回与输入等长的预测结果列表。

    设计决策：
    - client 在函数入口处创建一次，所有 worker 线程复用同一实例。
      anthropic / openai SDK 的 client 对象是线程安全的，可以复用。
    - max_workers=8：与 SoundnessBench 原版并发数一致，在 API 限流和吞吐量之间折中。
    - 结果列表预先初始化为 None 占位，通过 idx 回填，保证输出顺序与输入一致。
    """
    # 惰性导入 llm.py，一次性创建 client，所有 worker 线程复用同一实例
    _, create_client = _import_llm()
    # 一次性创建 client，后续所有 worker 线程复用，避免重复初始化开销
    client, client_model = create_client(model)

    # 预初始化结果列表（保证输出顺序与输入一致）
    results: list[dict[str, Any]] = [
        {"rigor_bucket": None, "confidence": None, "justification": None}
        for _ in pairs
    ]

    def score_one(idx: int) -> tuple[int, dict[str, Any]]:
        try:
            return idx, score_pair(pairs[idx], client, client_model, mode=mode)
        except Exception as exc:
            pair_id = pairs[idx].get("pair_id") or f"row_{idx}"
            print(f"[scorer] pair_id={pair_id} 打分失败: {exc}", flush=True)
            return idx, {"rigor_bucket": None, "confidence": None, "justification": None}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(score_one, i): i for i in range(len(pairs))}
        for future in as_completed(futures):
            idx, pred = future.result()
            results[idx] = pred

    return results
