import argparse
import datetime
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import docker
from datasets import load_dataset

from prompts.testrepo_prompt import get_test_description
from swebench.harness.test_spec import make_test_spec
from swebench.harness.docker_build import build_env_images, build_container, cleanup_container

from swe_bench.utils import (
    copy_to_container,
    copy_from_container,
    log_container_output,
    remove_existing_container,
    safe_log,
    setup_logger,
)
from utils.common_utils import load_json_file


# ============================================================
# swe_bench/harness.py
# SWE-bench 评估框架：在 Docker 容器中运行 coding_agent，收集并保存预测结果。
#
# 整体流程：
#   1. 加载 SWE-bench_Verified 数据集，构建 Docker 镜像（build_env_images）
#   2. 对每个 issue 创建专用 Docker 容器（SWE-bench 官方 test_spec 容器）
#   3. 将 agent 代码（coding_agent.py、tools/、llm.py 等）复制到容器
#   4. 执行 eval.sh 脚本设置目标仓库环境（apply issue patch、安装依赖）
#   5. 可选：按顺序 apply 父代 model_patch（使用演化后的 agent 代码）
#   6. 运行 coding_agent.py（最长 9 小时），让 agent 尝试修复 issue
#   7. 从容器中取回运行日志和 model_patch.diff
#   8. 支持 num_evals > 1（对同一 issue 多次采样，评估 agent 稳定性）
#
# Docker 容器设计：
#   - 使用 SWE-bench 官方的 per-issue Docker 镜像（包含目标仓库和测试环境）
#   - agent 代码在容器外（可更新），每次评估前复制进容器
#   - API 密钥通过环境变量注入（不写入镜像，安全性更好）
# ============================================================


def process_entry(entry, out_dname, model_name_or_path, model_patch_paths):
    """
    处理单个 SWE-bench issue：在 Docker 容器中运行 agent 并收集结果。

    这是 harness 的核心函数，被 ThreadPoolExecutor 并行调用。
    每个 issue 独立占用一个 Docker 容器，互不干扰。

    执行步骤：
      1. 若输出 JSON 已存在，直接跳过（支持断点续跑）
      2. 使用 SWE-bench 的 make_test_spec 创建 issue 专用容器规格
      3. 启动容器，将 agent 文件复制进去
      4. 执行 eval.sh（设置目标仓库环境：reset 到 base_commit、安装测试依赖）
      5. 可选：按顺序 apply 所有父代 model_patch（支持 DGM 演化后的 agent 版本）
      6. 运行 coding_agent.py（9h 超时），等待 agent 完成
      7. 从容器取回日志和 model_patch.diff，写入输出 JSON
      8. 无论成功与否，在 finally 中清理容器

    目录切换逻辑：
      SWE-bench 的 test_spec 可能因 os.chdir 竞态问题而出错，
      代码尝试确保当前目录是 /dgm（agent 代码的根目录）。

    Args:
        entry (dict): SWE-bench 数据集中的 issue 条目（含 instance_id、problem_statement 等）。
        out_dname (Path): 本次评估的输出目录（每个 issue 的结果写入此目录）。
        model_name_or_path (str): 模型标识符（写入结果 JSON，供评估报告使用）。
        model_patch_paths (list[str] | None): 父代 model_patch 文件路径列表（按时间顺序，最老的在前）。

    Returns:
        dict: {"success": bool, "instance_id": str, ["error": str]}
    """
    instance_id = entry['instance_id']
    problem_statement = entry['problem_statement']
    base_commit = entry['base_commit']
    chat_history_file = out_dname / (instance_id + ".md")
    out_fname = out_dname / (instance_id + ".json")
    eval_file = out_dname / f"{instance_id}_eval.sh"

    # 断点续跑：若结果 JSON 已存在则跳过
    if out_fname.exists():
        print(f"Skipping existing entry {instance_id}")
        return {"success": True, "instance_id": instance_id}

    try:
        client = docker.from_env()
        run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        logger = setup_logger(str(out_dname / f"{instance_id}_docker.log"))
        nocache = True
        # make_test_spec：从 SWE-bench 数据集条目生成 Docker 容器规格（镜像名、eval 脚本等）
        test_spec = make_test_spec(entry)
        container_name = test_spec.get_instance_container_name(run_id)
        remove_existing_container(client, container_name)
        container = build_container(test_spec, client, run_id, logger, nocache, force_rebuild=False)
        container.start()

        # 确保当前目录是 /dgm（agent 文件复制的基准目录）
        # 这是防御性代码：多线程环境下 os.chdir 可能被其他线程改变
        tmp_currdir = os.path.abspath(os.getcwd())
        logger.info(f"Current directory: {tmp_currdir}")
        if not tmp_currdir.endswith('/dgm'):
            try:
                os.chdir('dgm')
                tmp_currdir = os.path.abspath(os.getcwd())
                logger.info(f"Changed directory to: {tmp_currdir}")
            except Exception as e:
                pass
        while not tmp_currdir.endswith('/dgm'):
            os.chdir('..')
            tmp_currdir = os.path.abspath(os.getcwd())
            logger.info(f"Changed directory to: {tmp_currdir}")

        # 将 agent 代码复制到容器的 /dgm/ 目录
        # 注意：每次评估都重新复制，确保使用最新版本（支持 DGM 自改进后的新版本）
        copy_to_container(container, 'coding_agent.py', '/dgm/coding_agent.py')
        copy_to_container(container, 'requirements.txt', '/dgm/requirements.txt')
        copy_to_container(container, 'pytest.ini', '/dgm/pytest.ini')
        copy_to_container(container, 'tools/', '/dgm/tools/')
        copy_to_container(container, 'utils/', '/dgm/utils/')
        copy_to_container(container, 'tests/', '/dgm/tests/')
        copy_to_container(container, 'prompts/', '/dgm/prompts/')
        copy_to_container(container, 'llm.py', '/dgm/llm.py')
        copy_to_container(container, 'llm_withtools.py', '/dgm/llm_withtools.py')
        chat_history_file_container = f'/dgm/{chat_history_file.name}'

        # 执行 eval.sh：SWE-bench 官方环境设置脚本
        # 内容包括：git reset 到 base_commit、安装 issue 相关测试依赖等
        logger.info("Setting up environment")
        eval_script = test_spec.eval_script
        eval_file.write_text(eval_script)
        copy_to_container(container, eval_file, '/eval.sh')
        exec_result = container.exec_run("/bin/bash /eval.sh", workdir='/')
        log_container_output(exec_result)
        exec_result = container.exec_run("rm /eval.sh", workdir='/')
        log_container_output(exec_result)

        # 根据 eval.sh 内容推断测试命令，生成测试描述给 agent 参考
        test_description = get_test_description(eval_script=eval_script, swerepo=True)

        # 按顺序 apply 父代 model_patch（重现当前 agent 版本的代码状态）
        # 如果 model_patch_paths 为 None（baseline），不 apply 任何 patch
        if model_patch_paths:
            safe_log("Applying model patches")
            for model_patch_path in model_patch_paths:
                copy_to_container(container, model_patch_path, '/dgm/parent_patch.txt')
                exec_result = container.exec_run("/bin/sh -c 'patch -p1 < /dgm/parent_patch.txt'", workdir='/dgm')
                log_container_output(exec_result)
                exec_result = container.exec_run("rm /dgm/parent_patch.txt", workdir='/dgm')
                log_container_output(exec_result)

        # 安装 agent 自身的依赖（可能因 self-improve 后 requirements.txt 有更新）
        safe_log("Installing more requirements")
        exec_result = container.exec_run("python -m pip install -r /dgm/requirements.txt", workdir='/')
        log_container_output(exec_result)

        # 将主机的 API 密钥注入容器环境变量
        env_vars = {
            "ANTHROPIC_API_KEY": os.getenv('ANTHROPIC_API_KEY'),
            "AWS_REGION": os.getenv('AWS_REGION'),
            "AWS_REGION_NAME": os.getenv('AWS_REGION_NAME'),
            "AWS_ACCESS_KEY_ID": os.getenv('AWS_ACCESS_KEY_ID'),
            "AWS_SECRET_ACCESS_KEY": os.getenv('AWS_SECRET_ACCESS_KEY'),
            "OPENAI_API_KEY": os.getenv('OPENAI_API_KEY'),
        }
        safe_log("Running the agent")
        # timeout 32400 = 9 小时；单个 issue 的 agent 运行不应超过这个时间
        # /testbed/ 是 SWE-bench 容器中目标仓库（有 bug 的代码）的标准路径
        cmd = [
            "timeout", "32400",
            "python", "/dgm/coding_agent.py",
            "--problem_statement", problem_statement,
            "--git_dir", "/testbed/",           # 目标仓库路径（SWE-bench 容器约定）
            "--chat_history_file", chat_history_file_container,
            "--base_commit", base_commit,
            "--outdir", "/dgm/",
            "--test_description", test_description,
            "--instance_id", instance_id,
        ]
        exec_result = container.exec_run(cmd, environment=env_vars, workdir='/')
        log_container_output(exec_result)

        # 取回 agent 运行日志（主日志 + 可能的额外日志文件）
        logger.info("Copying output files back to host")
        copy_from_container(container, chat_history_file_container, chat_history_file)
        # 查找额外的日志文件（如 agent 自己创建的多次尝试日志）
        exec_result = container.exec_run(f"find /dgm/ -name '{instance_id}_*.md'", workdir='/')
        chat_history_files_container = exec_result.output.decode().split()
        for chat_history_file_container in chat_history_files_container:
            chat_history_file = out_dname / Path(chat_history_file_container).name
            copy_from_container(container, chat_history_file_container, chat_history_file)

        # 取回 model_patch.diff（agent 对目标仓库的修改）
        logger.info("Getting model_patch")
        exec_result = container.exec_run("cat /dgm/model_patch.diff")
        log_container_output(exec_result)
        model_patch = exec_result.output.decode()

        # 查找额外的候选 patch 文件（如 agent 进行了多次尝试）
        proposed_model_patches = []
        exec_result = container.exec_run("find /dgm/ -name 'model_patch_*.diff'", workdir='/')
        model_patch_files_container = exec_result.output.decode().split()
        for model_patch_file_container in model_patch_files_container:
            exec_result = container.exec_run(f"cat {model_patch_file_container}")
            log_container_output(exec_result)
            proposed_model_patch = exec_result.output.decode()
            proposed_model_patches.append(proposed_model_patch)

        # 将结果写入 JSON 文件（供 swe_bench/report.py 读取并评估）
        result = {
            "instance_id": instance_id,
            "model_name_or_path": model_name_or_path,
            "model_patch": model_patch,
            'proposed_model_patches': proposed_model_patches,
        }
        out_fname.write_text(json.dumps(result, indent=4))

        return {"success": True, "instance_id": instance_id}

    except Exception as e:
        print(f"Error processing entry {instance_id}: {str(e)}")
        return {"success": False, "instance_id": instance_id, "error": str(e)}

    finally:
        # 无论成功与否都清理容器（防止容器泄漏占用资源）
        try:
            cleanup_container(client, container, logger)
        except Exception as e:
            print(f"Error cleaning up Docker container for {instance_id}: {e}")


def harness(
        test_task_list=None,
        num_samples=-1,
        max_workers=4,
        model_name_or_path=None,
        model_patch_paths=None,
        num_evals=1,
        num_evals_parallel=1,
        pred_dname='./swe_bench/predictions',
    ):
    """
    SWE-bench 并行评估框架：统一管理多 issue 并行处理和多次重复评估。

    并行设计（两层 ThreadPoolExecutor）：
      外层：num_evals_parallel 个线程，并行处理多次重复评估（num_evals 次）
      内层：max_workers 个线程，在单次评估中并行处理多个 issue

    重复评估的意义：
      agent 是随机的（LLM 采样），同一 issue 多次运行结果不同。
      num_evals > 1 可以评估 agent 的稳定性，取平均成绩更可靠。
      每次评估的结果保存到独立的子目录（model_name_or_path_{eval_idx}）。

    环境镜像预构建：
      build_env_images 在评估开始前预先构建所有需要的 Docker 环境镜像，
      避免每个 issue 在运行时重复构建（提高并行效率）。

    Args:
        test_task_list (list[str] | None): 要处理的 instance_id 列表；None 表示处理全部。
        num_samples (int): 最多处理的 issue 数；-1 表示全部。
        max_workers (int): 单次评估内的并行 issue 数。
        model_name_or_path (str | None): 模型标识符；None 时自动生成（时间戳）。
        model_patch_paths (list[str] | None): 父代 patch 路径列表。
        num_evals (int): 对每个 issue 重复评估的次数（用于稳定性分析）。
        num_evals_parallel (int): 并行运行的评估次数。
        pred_dname (str): 预测结果输出目录。

    Returns:
        list[Path]: 所有评估轮次的输出目录列表（供 make_report 读取）。
    """
    dataset = load_dataset("princeton-nlp/SWE-bench_Verified")
    dataset = dataset['test']

    if model_name_or_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_or_path = f"{timestamp}--claude-3-5-sonnet-20241022"
    pred_dname = Path(pred_dname)
    pred_dname.mkdir(exist_ok=True)
    out_dnames = []

    # 过滤数据集：只保留指定的 issue
    entries = list(dataset)
    if test_task_list:
        entries = [entry for entry in entries if entry['instance_id'] in test_task_list]
    if num_samples > 0:
        entries = entries[:num_samples]

    # 预先构建所有需要的 Docker 环境镜像（不强制重建，复用已有镜像）
    client = docker.from_env()
    build_env_images(client, dataset=entries, force_rebuild=False, max_workers=max_workers)

    def process_evaluation(eval_idx):
        """处理单次评估：为所有 issue 创建输出目录，并行运行 agent。"""
        # 每次评估有独立的输出目录（model_name_or_path_{eval_idx}）
        model_name_or_path_inst = f"{model_name_or_path}_{eval_idx}"
        out_dname = pred_dname / model_name_or_path_inst
        out_dname.mkdir(exist_ok=True)

        print(f"Starting evaluation {eval_idx} for model {model_name_or_path}")

        # 内层线程池：并行处理所有 issue
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_entry = {
                executor.submit(process_entry, entry, out_dname, model_name_or_path_inst, model_patch_paths): entry
                for entry in entries
            }

            # 使用 as_completed 按完成顺序处理（而非按提交顺序），即时打印进度
            for future in as_completed(future_to_entry):
                result = future.result()
                if result["success"]:
                    print(f"Successfully processed entry {result['instance_id']} for eval {eval_idx}")
                else:
                    print(f"Failed to process entry {result['instance_id']} for eval {eval_idx}: {result.get('error', 'Unknown error')}")
        return out_dname

    # 外层线程池：并行运行多次重复评估
    with ThreadPoolExecutor(max_workers=num_evals_parallel) as eval_executor:
        out_dnames = list(eval_executor.map(process_evaluation, range(num_evals)))

    print(f"All evaluations completed for model {model_name_or_path}")
    return out_dnames


def main():
    """
    命令行入口：直接运行 SWE-bench 评估（不通过 DGM 演化框架）。

    常用场景：
      1. 评估基准 agent（不传 model_patch_paths）
      2. 评估特定版本的 agent（传入具体的 patch 路径）
      3. 只评估子集（'small'：12个 issue；'medium'：50个 issue；None：全部 500个）
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples to process")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of concurrent threads")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="Model name or path")
    parser.add_argument("--model_patch_paths", type=str, default=None, help="Paths to the model patches")
    parser.add_argument("--num_evals", type=int, default=1, help="Repeated number of swe evaluations")
    parser.add_argument("--num_evals_parallel", type=int, default=1, help="Number of parallel repeated evaluations")
    parser.add_argument("--pred_dname", type=str, default="./swe_bench/predictions", help="Output directory for predictions")
    parser.add_argument("--test_task_list", type=str, default=None, help="Subset of swe issues to process")
    args = parser.parse_args()

    # 'small' 和 'medium' 是预定义的 issue 子集（存储在 swe_bench/subsets/ 目录中）
    if args.test_task_list == 'small':
        test_task_list = load_json_file("./swe_bench/subsets/small.json")
    elif args.test_task_list == 'medium':
        test_task_list = load_json_file("./swe_bench/subsets/medium.json")
    else:
        test_task_list = None

    harness(
        test_task_list=test_task_list,
        num_samples=args.num_samples,
        max_workers=args.max_workers,
        model_name_or_path=args.model_name_or_path,
        model_patch_paths=args.model_patch_paths,
        num_evals=args.num_evals,
        num_evals_parallel=args.num_evals_parallel,
        pred_dname=args.pred_dname,
    )


if __name__ == "__main__":
    main()
