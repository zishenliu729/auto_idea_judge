"""
evaluate.py
-----------
SoundnessBench judge 的评估入口，在 DGM 适配中充当"pytest"的角色。

在整个适配架构中的定位：
  - DGM agent 自改进后调用此脚本验证改进效果（类比原版的 pytest 测试）
  - self_improve_step.py 的 run_harness_judge() 在容器内执行此脚本，
    解析输出的 eval_result.json 作为 overall_performance
  - DGM 演化期间只能使用训练集（--data 指定 train_small 或 train_medium），
    测试集（soundnessbench_test.jsonl）仅在最终报告时使用

典型调用方式：
  # 快速验证（agent 自改进后执行）
  python evaluate.py --data data/soundnessbench_train_small.jsonl --pytest --baseline 0.60

  # 较完整评估（晋升验证阶段）
  python evaluate.py --data data/soundnessbench_train_medium.jsonl

  # 最终报告（演化结束后仅运行一次，不用于 self-improve 循环）
  python evaluate.py --data data/soundnessbench_test.jsonl
"""

import argparse
import json
import os
import sys
from typing import Any

# 确保 judge/ 包和 llm.py 可以被正确 import（无论从哪个目录运行此脚本）
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from judge.scorer import score_pairs_concurrent
from judge.metrics import compute_bucket_metrics

# Judge model switchboard (2026-07-08): default final judge backbone to Qwen
# for controlled SoundnessBench runs, but allow SOUNDNESSBENCH_MODEL/DGM_JUDGE_MODEL
# to switch providers without editing code. Secrets still come from environment.
DEFAULT_MODEL = os.getenv(
    "SOUNDNESSBENCH_MODEL",
    os.getenv("DGM_JUDGE_MODEL", "maas/Qwen3.5-397B-A17B-FP8"),
)

# 默认数据文件：训练集小子集，供 agent 快速验证改进效果
DEFAULT_DATA = os.path.join(_script_dir, "data", "soundnessbench_train_small.jsonl")


def load_pairs(path: str) -> list[dict[str, Any]]:
    """从 JSONL 文件加载 SoundnessBench pair 列表。"""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def run_evaluation(
    data_path: str,
    model: str,
    mode: str,
    max_workers: int,
) -> dict[str, Any]:
    """执行完整评估流程：加载数据 → 并发打分 → 计算指标 → 返回结果 dict。"""
    pairs = load_pairs(data_path)
    print(f"[evaluate] 加载 {len(pairs)} 个样本，来源：{data_path}", flush=True)
    print(f"[evaluate] 模型：{model}，模式：{mode}，并发数：{max_workers}", flush=True)

    # 并发调用 LLM 打分，返回与 pairs 等长的预测列表
    predictions = score_pairs_concurrent(pairs, model=model, mode=mode, max_workers=max_workers)

    # 构建 ground_truth 列表（与 predictions 下标对齐）
    ground_truths = [{"rigor_bucket": p.get("rigor_bucket")} for p in pairs]

    # 计算 accuracy + Cohen's kappa
    metrics = compute_bucket_metrics(predictions, ground_truths)
    summary = metrics["summary"]

    # 整合完整结果（供 self_improve_step.py 的 run_harness_judge 解析）
    result = {
        # overall_performance 是 DGM 读取 fitness 分数的标准字段
        "overall_performance": {
            "accuracy_score": summary["rigor_bucket_accuracy"],
            "kappa": summary["rigor_bucket_kappa"],
            "total_n": summary["total_n"],
        },
        "metrics": metrics,
        "model": model,
        "evaluation_mode": mode,
        "data_path": data_path,
        "n_pairs": len(pairs),
        # 逐条结果，便于 DGM 诊断模型（o1）分析误判样本
        "results": [
            {
                "pair_id": pair.get("pair_id", f"row_{i}"),
                "paper_id": pair.get("paper_id"),
                "ground_truth": pair.get("rigor_bucket"),
                "prediction": pred.get("rigor_bucket"),
                "confidence": pred.get("confidence"),
                "justification": pred.get("justification"),
                "correct": pred.get("rigor_bucket") == pair.get("rigor_bucket"),
            }
            for i, (pair, pred) in enumerate(zip(pairs, predictions))
        ],
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="评估 judge workflow 在 SoundnessBench 上的准确率")
    parser.add_argument(
        "--data", default=DEFAULT_DATA,
        help="JSONL 数据文件路径（应使用训练集，不要用测试集做自改进）"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="调用的 LLM 模型 ID（默认读取 SOUNDNESSBENCH_MODEL/DGM_JUDGE_MODEL，否则 MaaS Qwen）"
    )
    parser.add_argument(
        "--mode", default="direct_bucket",
        choices=["direct_bucket", "direct_bucket_aggressive"],
        help="评估模式：direct_bucket（均衡）或 direct_bucket_aggressive（激进过滤）"
    )
    parser.add_argument(
        "--max-workers", type=int, default=8,
        help="并发打分的线程数（默认 8，与 SoundnessBench 原版一致）"
    )
    parser.add_argument(
        "--output", default=None,
        help="结果 JSON 写入路径（默认：与 --data 同目录下的 eval_result.json）"
    )
    # --pytest 模式：accuracy < baseline 时 exit(1)，供 agent 用来判断改进是否有效
    # 类比 SWE-bench 流程中的 pytest 测试失败机制
    parser.add_argument(
        "--pytest", action="store_true",
        help="CI 模式：accuracy 低于 --baseline 时以 exit(1) 退出（供 agent 验证用）"
    )
    parser.add_argument(
        "--baseline", type=float, default=0.60,
        help="--pytest 模式下的 accuracy 阈值（默认 0.60）"
    )
    args = parser.parse_args()

    # 确定输出路径（默认放在数据文件同目录）
    if args.output:
        output_path = args.output
    else:
        data_dir = os.path.dirname(os.path.abspath(args.data))
        output_path = os.path.join(data_dir, "eval_result.json")

    result = run_evaluation(
        data_path=args.data,
        model=args.model,
        mode=args.mode,
        max_workers=args.max_workers,
    )

    # 写入 JSON 文件（self_improve_step.py 的 run_harness_judge 会读这个文件）
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # 打印核心指标到 stdout（供人工查看 + 容器日志捕获）
    perf = result["overall_performance"]
    accuracy = perf["accuracy_score"]
    kappa = perf["kappa"]
    total_n = perf["total_n"]

    print(f"\n[evaluate] ── 评估结果 ──────────────────────────────")
    print(f"[evaluate] accuracy : {accuracy:.4f}" if accuracy is not None else "[evaluate] accuracy : N/A")
    print(f"[evaluate] kappa    : {kappa:.4f}"    if kappa is not None    else "[evaluate] kappa    : N/A")
    print(f"[evaluate] total_n  : {total_n}")
    print(f"[evaluate] 结果写入 : {output_path}")
    print(f"[evaluate] ────────────────────────────────────────────\n")

    # --pytest 模式：accuracy 低于阈值时 exit(1)，通知 agent 改进未达到要求
    if args.pytest:
        if accuracy is None or accuracy < args.baseline:
            print(f"[evaluate] FAIL: accuracy {accuracy} < baseline {args.baseline}")
            sys.exit(1)
        else:
            print(f"[evaluate] PASS: accuracy {accuracy:.4f} >= baseline {args.baseline}")


if __name__ == "__main__":
    main()
