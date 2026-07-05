import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess


# ============================================================
# swe_bench/report.py
# SWE-bench 评估报告生成器：将 coding_agent 的预测 patch 提交给 SWE-bench 评估框架，
# 生成解决率报告。
#
# 工作流程：
#   1. 加载各个 instance 的预测 JSON 文件（含 model_patch 字段）
#   2. 过滤掉 patch 中对测试目录的修改（避免干扰官方私有测试）
#   3. 将所有预测写入 all_preds.jsonl（SWE-bench 评估器需要的格式）
#   4. 调用 SWE-bench 的 run_evaluation.py 脚本（subprocess），生成测试报告
#   5. 支持多目录并行评估（ThreadPoolExecutor）
#
# 与 evo_utils.get_all_performance() 的关系：
#   此文件负责"生成原始评估结果 JSON"，
#   get_all_performance() 负责"读取这些 JSON 并聚合统计"。
# ============================================================


def load_predictions(paths):
    """
    从指定路径列表加载所有预测 JSON 文件，返回 {instance_id: pred_dict} 映射。

    支持两种路径格式：
      - 单个 JSON 文件：直接加载
      - 目录：加载目录下所有 *.json 文件

    每个预测 JSON 文件格式：
      {
        "instance_id": "django__django-14999",
        "model_patch": "...(diff 内容)...",
        "model_name_or_path": "coding_agent",
        ...
      }

    Args:
        paths (list[str]): 文件或目录路径列表。

    Returns:
        dict[str, dict]: {instance_id: 预测数据 dict} 映射。
    """
    prediction_paths = []
    for path in paths:
        path = Path(path)
        if path.is_file():
            prediction_paths.append(path)
        elif path.is_dir():
            prediction_paths += list(path.glob("*.json"))
        else:
            assert False, path

    predictions = dict()
    for fname in prediction_paths:
        try:
            pred = json.loads(fname.read_text())
        except json.decoder.JSONDecodeError as err:
            raise err

        if "instance_id" not in pred:
            print("Skipping json without instance_id", fname)
            continue

        inst = pred["instance_id"]
        pred["json_fname"] = str(fname)  # 记录来源文件路径，便于调试
        predictions[inst] = pred

    return predictions


def remove_patches_to_tests(model_patch):
    """
    从 model_patch 中移除对测试目录的所有修改。

    为什么要移除测试修改：
      SWE-bench 的评估流程是：先 apply model_patch，再 apply test_patch（官方私有测试），
      然后运行测试。若 model_patch 中包含对 tests/ 目录的修改（agent 自己写的测试），
      可能与官方 test_patch 冲突，导致评估结果不准确。

    过滤逻辑：
      逐行扫描 patch，遇到 `diff --git a/...` 行时判断是否涉及测试目录，
      若是（路径含 /test/、/tests/、/testing/、/test_ 或 /tox.ini）则
      标记 is_tests=True，后续属于该文件的所有 diff 行都被过滤掉。

    Args:
        model_patch (str): unified diff 格式的 patch 字符串。

    Returns:
        str: 过滤掉测试目录修改后的 patch 字符串。
    """
    lines = model_patch.splitlines(keepends=True)
    filtered_lines = []
    is_tests = False

    for line in lines:
        if line.startswith("diff --git a/"):
            pieces = line.split()
            to = pieces[-1]  # "b/path/to/file" 格式
            if to.startswith("b/") and (
                    "/test/" in to
                    or "/tests/" in to
                    or "/testing/" in to
                    or "/test_" in to
                    or "/tox.ini" in to
            ):
                is_tests = True
            else:
                is_tests = False

        if not is_tests:
            filtered_lines.append(line)

    return "".join(filtered_lines)


def preds_to_jsonl(dname, predictions):
    """
    将预测字典写入 SWE-bench 评估器所需的 JSONL 格式文件。

    SWE-bench 的 run_evaluation.py 要求输入为 all_preds.jsonl，
    每行一个 JSON 对象，只需包含三个字段：
      - model_name_or_path：模型标识
      - model_patch：过滤后的 diff（不含测试目录修改）
      - instance_id：任务 ID

    注意：同一批次的所有预测必须使用相同的 model_name_or_path（assert 保证）。

    Args:
        dname (str | Path): 预测文件所在目录（all_preds.jsonl 写在此目录下）。
        predictions (dict[str, dict]): load_predictions() 返回的预测映射。

    Returns:
        str: 生成的 all_preds.jsonl 文件的绝对路径。
    """
    dname = Path(dname)

    predictions_jsonl = str(dname / "all_preds.jsonl")
    model_name_or_path = list(predictions.values())[0]["model_name_or_path"]
    with open(predictions_jsonl, "w") as fh:
        for inst, pred in predictions.items():
            # 确保所有预测使用相同的模型标识（评估器要求一致性）
            assert model_name_or_path == pred["model_name_or_path"]
            minimal_pred = dict(
                model_name_or_path=model_name_or_path,
                model_patch=remove_patches_to_tests(pred["model_patch"]),  # 过滤测试修改
                instance_id=pred["instance_id"],
            )
            fh.write(json.dumps(minimal_pred) + "\n")
    return predictions_jsonl


def run_evals(predictions_jsonl, run_id, dataset_name, root_dir, output_dir, num_eval_procs=5):
    """
    调用 SWE-bench 的 run_evaluation.py 对所有预测进行评估。

    注意 os.chdir 的使用：
      SWE-bench 评估器会把评估结果写入当前工作目录下的子目录，
      因此需要先切换到 output_dir，评估结束后切回 root_dir。
      这个"副作用"在多线程环境中存在竞态风险，
      ThreadPoolExecutor 中的每个线程共享工作目录状态（os.chdir 是进程级全局操作）。

    Args:
        predictions_jsonl (str): all_preds.jsonl 文件路径。
        run_id (str): 此次评估的唯一标识（用于命名输出目录）。
        dataset_name (str): SWE-bench 数据集名称（如 "princeton-nlp/SWE-bench_Verified"）。
        root_dir (str): DGM 根目录（/dgm/）。
        output_dir (str): 评估报告的输出目录。
        num_eval_procs (int): 并行评估进程数，默认 5。
    """
    os.chdir(output_dir)  # 切换到输出目录（评估器在当前目录写报告）
    run_evals_cmd = f"""
python {os.path.join(root_dir, './swe_bench/SWE-bench/swebench/harness/run_evaluation.py')}
    --dataset_name {dataset_name}
    --predictions_path {predictions_jsonl}
    --max_workers {num_eval_procs}
    --run_id {run_id}
"""
    # 将多行命令格式化为单行（去掉换行和多余空格）
    run_evals_cmd = " ".join([line.strip() for line in run_evals_cmd.split() if line.strip()])
    subprocess.run(run_evals_cmd.split(), check=True)
    os.chdir(root_dir)  # 切回 DGM 根目录


def make_report(
        dnames,
        run_ids=None,
        dataset_name="princeton-nlp/SWE-bench_Verified",
        output_dir='./swe_bench',
        dnames_workers=None,
        num_eval_procs=5,
    ):
    """
    并行生成多个目录的 SWE-bench 评估报告。

    使用场景：DGM 在一次评估中分批运行（如 3 个批次各处理 50 个 issue），
    每个批次的预测存放在不同目录。此函数用 ThreadPoolExecutor 并行处理所有批次，
    缩短总评估时间。

    Args:
        dnames (list[str]): 预测文件目录列表（每个目录对应一个批次）。
        run_ids (list[str] | None): 各批次的唯一标识；为 None 时自动生成 "000"、"001"...
        dataset_name (str): SWE-bench 数据集名称。
        output_dir (str): 评估报告输出目录，默认 './swe_bench'。
        dnames_workers (int | None): 并行线程数；为 None 时等于批次数。
        num_eval_procs (int): 每个批次内的并行进程数，默认 5。
    """
    root_dir = os.path.abspath(os.getcwd())  # 记录当前目录（应为 /dgm）
    output_dir = os.path.join(root_dir, output_dir)

    def process_single_dname(dname, run_id):
        """处理单个批次目录的内部函数（在线程中执行）。"""
        dname = Path(os.path.join(root_dir, dname))
        predictions = load_predictions([dname])
        predictions_jsonl = preds_to_jsonl(dname, predictions)
        run_evals(predictions_jsonl, run_id, dataset_name, root_dir, output_dir, num_eval_procs=num_eval_procs)
        print(f"Report generated for {dname}")

    # 自动生成 run_id 列表（如果未提供）
    if run_ids is None or len(run_ids) != len(dnames):
        run_ids = [f"{i:03}" for i in range(len(dnames))]
    if dnames_workers is None:
        dnames_workers = len(dnames)
    # 用线程池并行处理所有批次（注意：os.chdir 在多线程下有竞态风险）
    with ThreadPoolExecutor(max_workers=dnames_workers) as executor:
        executor.map(process_single_dname, dnames, run_ids)

    print("All reports generated.")


def main():
    """命令行入口，供直接调用评估流程时使用。"""
    parser = argparse.ArgumentParser(description="Run evaluations on predictions.")
    parser.add_argument('--dnames', type=str, nargs='+', help="List of directories of predictions to evaluate.")
    parser.add_argument('--run_ids', type=str, nargs='+', default=None, help="Run ID for this evaluation run.")
    parser.add_argument('--dataset_name', type=str, default="princeton-nlp/SWE-bench_Verified", help="Name of the dataset to evaluate on.")
    parser.add_argument('--dnames_workers', type=int, default=None, help="Number of parallel workers to use for processing dnames.")
    parser.add_argument('--num_eval_procs', type=int, default=5, help="Number of parallel processes to use for evaluation.")
    parser.add_argument('--output_dir', type=str, default='./swe_bench', help="Output directory for the reports.")
    args = parser.parse_args()

    make_report(
        args.dnames,
        run_ids=args.run_ids,
        dataset_name=args.dataset_name,
        dnames_workers=args.dnames_workers,
        num_eval_procs=args.num_eval_procs,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
