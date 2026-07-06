"""
judge/metrics.py
----------------
计算 rigor_bucket 预测的 accuracy 和 Cohen's kappa 系数。

在整个适配架构中的定位：
  - 纯计算模块，无 LLM 调用，输入预测列表和真值列表，输出指标 dict
  - evaluate.py 在收集所有预测后调用此模块计算最终结果
  - DGM 用 summary.rigor_bucket_accuracy 作为 overall_performance 的核心分数

与 SoundnessBench rigorbench/evaluation/metrics.py 的关系：
  - 逻辑完全一致，仅将 normalize_bucket 内联（原版从 rigorbench.buckets import），
    使 judge/ 目录无需依赖 SoundnessBench 包即可独立运行。
"""

from __future__ import annotations
from typing import Any


def _normalize_bucket(label: Any) -> str | None:
    """内联版 normalize_bucket，与 scorer.py 中保持一致。"""
    if label is None:
        return None
    value = str(label).strip().lower()
    return value if value in {"low", "high"} else None


def cohen_kappa(y_pred: list[str], y_true: list[str]) -> float | None:
    """计算两个分类标签列表的 Cohen's kappa 系数。

    kappa = (observed_accuracy - expected_accuracy) / (1 - expected_accuracy)
    衡量预测与随机基准相比的一致性提升，比 accuracy 更能反映真实判断质量。
    返回 None 表示样本数不足（< 2）或输入长度不一致。
    """
    n = len(y_pred)
    if n != len(y_true) or n < 2:
        return None
    labels = sorted(set(y_true) | set(y_pred))
    observed = sum(p == g for p, g in zip(y_pred, y_true)) / n
    expected = sum((y_pred.count(label) / n) * (y_true.count(label) / n) for label in labels)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def compute_bucket_metrics(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    bucket_key: str = "rigor_bucket",
) -> dict[str, Any]:
    """计算 rigor_bucket 预测的 accuracy 和 Cohen's kappa。

    会自动跳过预测或真值为 None 的样本（LLM 解析失败的情况），
    summary.total_n 反映实际参与计算的有效样本数（可能 < 总输入数）。

    返回结构（与原版完全一致，evaluate.py 按此结构取值）：
      {
        "per_field": {"rigor_bucket": {"n", "accuracy", "cohen_kappa"}},
        "summary": {"rigor_bucket_accuracy", "rigor_bucket_kappa", "total_n"}
      }
    """
    pairs: list[tuple[str, str]] = []
    for pred, gt in zip(predictions, ground_truths):
        pred_bucket = _normalize_bucket(pred.get(bucket_key))
        gt_bucket = _normalize_bucket(gt.get(bucket_key))
        # 任意一方为 None（解析失败 or 真值缺失）则跳过，不参与指标计算
        if pred_bucket is not None and gt_bucket is not None:
            pairs.append((pred_bucket, gt_bucket))

    n = len(pairs)
    if n == 0:
        accuracy, kappa = None, None
    else:
        pred_vals = [p for p, _ in pairs]
        gt_vals = [g for _, g in pairs]
        accuracy = sum(p == g for p, g in pairs) / n
        kappa = cohen_kappa(pred_vals, gt_vals)

    return {
        "per_field": {
            bucket_key: {"n": n, "accuracy": accuracy, "cohen_kappa": kappa}
        },
        "summary": {
            "rigor_bucket_accuracy": accuracy,
            "rigor_bucket_kappa": kappa,
            "total_n": n,
        },
    }
