import argparse
import datetime
import json
import os
import docker

from llm import create_client, get_response_from_llm, extract_json_between_markers
from prompts.self_improvement_prompt import get_diagnose_prompt_polyglot, get_diagnose_prompt_swe, get_problem_description_prompt
from prompts.diagnose_improvement_prompt import get_diagnose_improvement_prompt
from prompts.testrepo_prompt import get_test_description
from swe_bench.harness import harness
from polyglot.harness import harness as polyglot_harness
from swe_bench.report import make_report
from utils.common_utils import load_json_file
from utils.evo_utils import get_model_patch_paths, get_all_performance, is_compiled_self_improve
from utils.docker_utils import (
    build_dgm_container,
    cleanup_container,
    copy_from_container,
    copy_to_container,
    log_container_output,
    remove_existing_container,
    setup_logger,
    safe_log,
)


# ============================================================
# self_improve_step.py
# DGM 自改进的核心执行模块：一次完整的"代码演化"流程。
#
# 整体流程（self_improve 函数）：
#   1. 创建 DGM Docker 容器（agent 代码的运行环境）
#   2. 按 patch chain 将父代所有 model_patch 按顺序 apply 到容器中
#      （重现父代 agent 的代码状态）
#   3. git commit 当前状态，获得干净的 base commit hash
#   4. 调用 diagnose_problem（o1 模型）：分析父代运行日志，生成改进方案
#   5. 在容器中运行 coding_agent.py --self_improve（30分钟超时）：
#      agent 根据改进方案修改自己的代码
#   6. 从容器取回 model_patch.diff（本次自改进的代码变更）
#   7. 验证 patch 不为空
#   8. 调用 run_harness_swe/polyglot：评估新版 agent 在 SWE-bench/Polyglot 上的表现
#   9. 可选：调用 diagnose_improvement（o1 模型）：分析本次改进是否有效
#  10. 保存 metadata.json（含运行信息、评估结果、改进诊断）
#
# 与 DGM_outer.py 的关系：
#   DGM_outer.py 是外层演化循环（80代），管理 archive 和多次并行 self_improve；
#   self_improve_step.py 是单次演化步骤，关注具体的改进和评估。
#
# DGM Docker 容器 vs SWE-bench Docker 容器：
#   - DGM 容器：运行 agent 自改进的环境（agent 修改自己的代码）
#   - SWE-bench 容器：运行 agent 修复 issue 的环境（agent 修改目标仓库代码）
# ============================================================


# 全局数据集变量（在 self_improve 调用时初始化，避免重复加载）
dataset = None
# 用于诊断的高推理能力模型（o1 系列，专门用于分析和推理，不用于代码执行）
diagnose_model = 'o1-2024-12-17'


def diagnose_problem(entry, commit, root_dir, out_dir, patch_files=[], max_attempts=3, polyglot=False):
    """
    调用 o1 模型分析父代 agent 的运行日志，生成改进问题陈述（problem_statement）。

    这个 problem_statement 就是自改进步骤的"任务书"：
    它告诉 coding_agent.py "你需要改进哪里、改进成什么样"。

    递归重试机制：
      o1 可能偶尔返回格式不正确的 JSON（没有按指定格式输出）。
      使用 max_attempts 控制最大重试次数，避免无限递归。
      失败时返回 None，让调用方决定如何处理（通常会跳过本次改进）。

    Args:
        entry (str): SWE-bench 任务 ID 或特殊值（'solve_empty_patches' 等）。
        commit (str): 父代 agent 的版本 commit hash。
        root_dir (str): DGM 根目录（/dgm/）。
        out_dir (str): 父代评估结果目录（用于读取运行日志）。
        patch_files (list[str]): 父代所有 patch 文件路径列表（用于在代码读取时叠加）。
        max_attempts (int): 最大重试次数，默认 3。
        polyglot (bool): 是否为 Polyglot 模式。

    Returns:
        str | None: 格式化的改进问题陈述（包含 agent 架构说明 + 具体改进任务），
                    或 None（重试耗尽后返回）。
    """
    client = create_client(diagnose_model)
    if polyglot:
        diagnose_sys_message, diagnose_prompt = get_diagnose_prompt_polyglot(
            entry, commit, root_dir, out_dir, dataset,
            patch_files=patch_files,
        )
    else:
        diagnose_sys_message, diagnose_prompt = get_diagnose_prompt_swe(
            entry, commit, root_dir, out_dir, dataset,
            patch_files=patch_files,
        )
    try:
        response, msg_history = get_response_from_llm(
            msg=diagnose_prompt,
            client=client[0],
            model=client[1],
            system_message=diagnose_sys_message,
            print_debug=False,
            msg_history=None,
        )
        safe_log(f"Message history: {msg_history}")
        response_json = extract_json_between_markers(response)
        assert response_json, "empty response json"
        problem_statement = get_problem_description_prompt(response_json, polyglot)
    except Exception as e:
        safe_log(f"Error while diagnosing the problem: {e}")
        if max_attempts > 0:
            return diagnose_problem(
                entry, commit, root_dir, out_dir,
                patch_files=patch_files,
                max_attempts=max_attempts-1,
                polyglot=polyglot,
            )
        else:
            return None
    return problem_statement


def diagnose_improvement(
        entry, parent_commit, root_dir, model_patch_file, out_dir, run_id,
        patch_files=[], max_attempts=3,
    ):
    """
    调用 o1 模型评估本次自改进的效果（改进后分析）。

    在 self_improve 完成后调用，让 o1 对比改进前后的运行日志，
    评估 model_patch 是否真正提升了 agent 的能力。

    评估结果包含：
      - impact：改进对 agent 性能的详细分析
      - improvements：具体提升点
      - regressions：引入的回归问题
      - score：-2 到 2 的数值评分

    score 保存在 metadata 中，DGM_outer.py 可以用它来决定是否将本次改进加入 archive。

    Args:
        entry (str): SWE-bench 任务 ID。
        parent_commit (str): 父代 commit hash（改进前）。
        root_dir (str): DGM 根目录。
        model_patch_file (str): 本次自改进的 model_patch.diff 路径。
        out_dir (str): 评估结果输出目录。
        run_id (str): 本次自改进的 run_id（改进后版本的标识）。
        patch_files (list[str]): 父代所有 patch 文件路径列表。
        max_attempts (int): 最大重试次数，默认 3。

    Returns:
        dict | None: 包含 impact/improvements/regressions/score 的 dict，
                     或 None（重试耗尽后返回）。
    """
    client = create_client(diagnose_model)
    diagnose_sys_message, diagnose_prompt = get_diagnose_improvement_prompt(
        entry, parent_commit, root_dir, model_patch_file, out_dir, run_id, dataset,
        patch_files=patch_files,
    )
    safe_log(f"Diagnosing the improvement: {diagnose_prompt}")
    try:
        response, msg_history = get_response_from_llm(
            msg=diagnose_prompt,
            client=client[0],
            model=client[1],
            system_message=diagnose_sys_message,
            print_debug=False,
            msg_history=None,
        )
        safe_log(f"Message history: {msg_history}")
        response_json = extract_json_between_markers(response)
        assert response_json, "empty response json"
        improvement_diagnosis = response_json
    except Exception as e:
        safe_log(f"Error while diagnosing the improvement: {e}")
        if max_attempts > 0:
            return diagnose_improvement(
                entry, parent_commit, root_dir, model_patch_file, out_dir, run_id,
                patch_files=patch_files, max_attempts=max_attempts-1,
            )
        else:
            return None
    return improvement_diagnosis


def save_metadata(metadata, output_dir):
    """
    将本次自改进的元数据保存到 metadata.json。

    metadata 包含：
      - run_id：本次 run 的唯一标识（时间戳）
      - parent_commit：父代版本的 run_id
      - entry：诊断时使用的 SWE-bench 任务 ID
      - problem_statement：传给 coding_agent 的改进任务描述
      - model_patch_exists/notempty：patch 文件是否存在且非空
      - swe_dnames：SWE-bench 评估输出目录列表
      - overall_performance：聚合评估结果（解决率等）
      - is_compiled：是否通过编译检查
      - improvement_diagnosis：改进效果的诊断结果（如果启用）

    Args:
        metadata (dict): 要保存的元数据。
        output_dir (str): 输出目录路径（metadata.json 写入此目录）。
    """
    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)


def run_harness_swe(entry, model_name_or_path, patch_files, num_evals, output_dir, metadata, run_id, test_more_threshold, test_task_list, test_task_list_more):
    """
    对 SWE-bench 任务运行评估（支持两阶段评估：small → medium）。

    两阶段评估逻辑：
      第一阶段：在 small 子集（12个 issue）上评估
      如果通过率 >= test_more_threshold（默认 40%），则继续第二阶段
      第二阶段：在 medium 子集（50个 issue）上评估（更全面的性能评估）

    这个设计节省了计算资源：
      大多数改进不会显著提升性能（第一阶段就会低于阈值），
      只有有潜力的版本才值得花更多时间做更全面的评估。

    Args:
        entry (str): 用于诊断时的任务 ID（影响日志路径，不影响评估任务列表）。
        model_name_or_path (str): 模型标识符（当前 run_id）。
        patch_files (list[str]): 当前版本的完整 patch 链。
        num_evals (int): 对每个 issue 的重复评估次数。
        output_dir (str): 本次自改进的输出目录。
        metadata (dict): 元数据 dict（本函数会更新其中的 swe_dnames 和 overall_performance）。
        run_id (str): 本次 run 的唯一标识。
        test_more_threshold (float | None): 触发第二阶段评估的阈值（None 表示不做第二阶段）。
        test_task_list (list[str] | None): 第一阶段的评估任务列表（None 时只用 entry）。
        test_task_list_more (list[str] | None): 第二阶段的评估任务列表。
    """
    safe_log('Start harness')
    # 如果没有传入 test_task_list，就只评估 entry 这一个任务
    test_task_list = [entry] if test_task_list is None else test_task_list
    # 并行度取任务数和 5 中的较小值（避免资源耗尽）
    dnames = harness(
        test_task_list=test_task_list,
        num_samples=-1,
        max_workers=min(5, len(test_task_list)),
        model_name_or_path=model_name_or_path,
        model_patch_paths=patch_files,
        num_evals=num_evals,
        num_evals_parallel=5,
        pred_dname=os.path.join(output_dir, "predictions"),
    )
    metadata['swe_dnames'] = [str(dn) for dn in dnames]
    safe_log('Start make_report')
    # 生成 SWE-bench 官方评估报告
    make_report(
        dnames,
        run_ids=[f"{run_id}_{i}" for i in range(len(dnames))],
        dataset_name="princeton-nlp/SWE-bench_Verified",
        output_dir=output_dir,
        dnames_workers=5,
    )
    safe_log('Start get_performance')
    performances, overall_performance = get_all_performance(model_name_or_path, results_dir=output_dir)
    metadata['overall_performance'] = overall_performance
    safe_log("End of evaluation")

    # 第二阶段：如果性能超过阈值，进行更全面的评估
    if (overall_performance and \
        test_more_threshold is not None and test_task_list_more is not None and \
            overall_performance.get('total_resolved_instances', 0) >= len(test_task_list) * test_more_threshold):
        safe_log("Start additional evaluation cycle")
        dnames = harness(
            test_task_list=test_task_list_more,
            num_samples=-1,
            max_workers=min(5, len(test_task_list_more)),
            model_name_or_path=model_name_or_path,
            model_patch_paths=patch_files,
            num_evals=num_evals,
            num_evals_parallel=5,
            pred_dname=os.path.join(output_dir, "predictions"),
        )
        safe_log('Start make_report more')
        make_report(
            dnames,
            run_ids=[f"{run_id}_{i}" for i in range(len(dnames))],
            dataset_name="princeton-nlp/SWE-bench_Verified",
            output_dir=output_dir,
            dnames_workers=5,
        )
        safe_log('Start get_performance')
        performances, overall_performance = get_all_performance(model_name_or_path, results_dir=output_dir)
        metadata['overall_performance'] = overall_performance
        safe_log("End of evaluation more")


def run_harness_polyglot(entry, model_name_or_path, patch_files, num_evals, output_dir, metadata, run_id, test_more_threshold, test_task_list, test_task_list_more):
    """
    对 Polyglot 任务运行评估（与 SWE-bench 版本类似，但使用 Polyglot harness）。

    差异点：
      - 使用 polyglot_harness 而非 swe_bench.harness
      - max_workers 默认取 min(10, ...)（Polyglot 任务较轻量，可以更多并行）
      - 第二阶段 max_workers 直接用 50（Polyglot 任务更简单，扩展性更强）
      - 不调用 make_report（Polyglot 有自己的评估报告逻辑）
      - 第二阶段结果保存到 overall_performance_deep（区分于第一阶段）

    Args: （参数同 run_harness_swe，不再重复说明）
    """
    safe_log('Start harness')
    test_task_list = [entry] if test_task_list is None else test_task_list
    safe_log(f'workers {min(10, len(test_task_list))}')
    dnames = polyglot_harness(
        test_task_list=test_task_list,
        num_samples=-1,
        max_workers=min(10, len(test_task_list)),
        model_name_or_path=model_name_or_path,
        model_patch_paths=patch_files,
        num_evals=num_evals,
        num_evals_parallel=min(5, num_evals),
        pred_dname=os.path.join(output_dir, "predictions"),
        output_dir=output_dir
    )
    metadata['swe_dnames'] = [str(dn) for dn in dnames]
    safe_log('Start get_performance')
    performances, overall_performance = get_all_performance(model_name_or_path, results_dir=output_dir)
    metadata['overall_performance'] = overall_performance
    safe_log("End of evaluation")

    # 第二阶段：性能超过阈值时进行更大规模评估
    if (overall_performance and \
        test_more_threshold is not None and test_task_list_more is not None and \
            overall_performance.get('total_resolved_instances', 0) >= len(test_task_list) * test_more_threshold):
        safe_log("Start additional evaluation cycle")
        dnames = polyglot_harness(
            test_task_list=test_task_list_more,
            num_samples=-1,
            max_workers=50,  # Polyglot 任务轻量，50并行不会造成资源问题
            model_name_or_path=model_name_or_path,
            model_patch_paths=patch_files,
            num_evals=num_evals,
            num_evals_parallel=min(5, num_evals),
            pred_dname=os.path.join(output_dir, "predictions"),
            output_dir=output_dir
        )
        safe_log('Start get_performance')
        performances, overall_performance = get_all_performance(model_name_or_path, results_dir=output_dir)
        # 注意：第二阶段保存到 overall_performance_deep（而非覆盖 overall_performance）
        metadata['overall_performance_deep'] = overall_performance
        safe_log("End of evaluation more")


def self_improve(
    parent_commit='initial',  # 'initial' 表示从原始版本开始，否则为父代的 run_id
    output_dir='output_selfimprove/',
    force_rebuild=False,
    num_evals=1,
    post_improve_diagnose=True,
    entry=None,
    test_task_list=None,      # None 表示只评估 entry 这一个任务
    # 两阶段评估参数
    test_more_threshold=None,
    test_task_list_more=None,
    full_eval_threshold=None,
    # 基线对比参数
    run_baseline=None,
    polyglot=False
):
    """
    执行一次完整的 DGM 自改进步骤：诊断问题 → 修改代码 → 评估效果。

    这是 DGM 演化的核心函数，被 DGM_outer.py 的主循环并行调用。

    重要实现细节：

    1. parent_commit 与 run_id 的区别：
       - parent_commit：父代版本的标识（要改进的"起点"）
       - run_id：本次改进的唯一标识（本次结果的"终点"）

    2. patch chain 的重建：
       get_model_patch_paths 从 parent_commit 开始逆向追溯所有祖先 patch，
       按从老到新的顺序 apply，重现父代 agent 的代码状态。

    3. git commit 的意义：
       在 DGM 容器中 apply 完所有父代 patch 后，做一次 git commit，
       获得一个干净的 base_commit（作为 coding_agent.py 计算 diff 的基准）。
       这样 model_patch.diff 只包含本次改进的变化，不包含历史 patch 内容。

    4. 'initial' 父代的特殊处理：
       如果 parent_commit == 'initial'，patch_files 为空，
       不 apply 任何 patch（直接使用容器内的初始 agent 代码）。

    5. polyglot 模式的代码准备：
       - 删除 swe 版 coding_agent.py，将 polyglot 版重命名为 coding_agent.py
       - 删除 SWE-bench 专用工具（eval_utils.py, swe_log_parsers.py）

    6. run_baseline='no_selfimprove' 的含义：
       跳过 apply parent patches 的步骤，始终使用原始 agent 代码；
       但仍然运行评估，用于建立基准线（对比自改进版本的提升幅度）。

    Args:
        parent_commit (str): 父代版本标识（'initial' 或 run_id）。
        output_dir (str): 本次运行的输出目录（将在其中创建 {run_id}/ 子目录）。
        force_rebuild (bool): 是否强制重建 DGM Docker 镜像（通常 False，复用缓存）。
        num_evals (int): 每个 issue 的重复评估次数（用于稳定性分析）。
        post_improve_diagnose (bool): 是否在评估后调用 o1 诊断改进效果。
        entry (str | None): 用于诊断的 SWE-bench 任务 ID（None 时直接退出）。
        test_task_list (list | None): 第一阶段评估的任务列表（None 时只用 entry）。
        test_more_threshold (float | None): 触发第二阶段的阈值（None 时不做第二阶段）。
        test_task_list_more (list | None): 第二阶段的任务列表。
        full_eval_threshold (float | None): 触发全量评估的阈值（预留参数，当前未实现）。
        run_baseline (str | None): 基线模式（'no_selfimprove' 或 None）。
        polyglot (bool): 是否为 Polyglot 模式。

    Returns:
        dict: 本次自改进的元数据（包含评估结果、改进诊断等）。
    """
    global dataset
    # 加载数据集（Polyglot 从本地 JSON，SWE-bench 从 HuggingFace）
    if polyglot:
        with open("polyglot/polyglot_benchmark_metadata.json") as f:
            dataset = json.loads(f.read())
    else:
        from datasets import load_dataset
        dataset = load_dataset("princeton-nlp/SWE-bench_Verified")
        dataset = dataset['test']

    metadata = {}
    root_dir = os.path.abspath('./')  # 应为 /dgm/
    run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out_dir_base = output_dir
    output_dir = os.path.join(root_dir, f"{output_dir}/{run_id}/")
    os.makedirs(output_dir, exist_ok=True)
    metadata['run_id'] = run_id
    metadata['parent_commit'] = parent_commit
    # 预加载大任务列表（用于 full eval，当前传给 run_harness 但未实现完整功能）
    test_task_list_big = load_json_file("./swe_bench/subsets/big.json")

    logger = setup_logger(os.path.join(output_dir, "self_improve.log"))

    # 创建 DGM Docker 容器（agent 代码的修改在这里进行）
    image_name = "dgm"
    container_name = f"dgm-container-{run_id}"
    client = docker.from_env()
    remove_existing_container(client, container_name)
    container = build_dgm_container(
        client, root_dir, image_name, container_name,
        force_rebuild=force_rebuild,
    )
    container.start()

    # Polyglot 模式：替换 coding_agent.py 并清理 SWE-bench 专用文件
    if polyglot:
        exec_result = container.exec_run("rm /dgm/coding_agent.py", workdir='/')
        log_container_output(exec_result)
        exec_result = container.exec_run("mv /dgm/coding_agent_polyglot.py /dgm/coding_agent.py", workdir='/')
        log_container_output(exec_result)
        exec_result = container.exec_run("rm /dgm/utils/eval_utils.py", workdir='/')
        log_container_output(exec_result)
        exec_result = container.exec_run("rm /dgm/utils/swe_log_parsers.py", workdir='/')
        log_container_output(exec_result)
    else:
        # SWE 模式：清理 polyglot 版本
        exec_result = container.exec_run("rm /dgm/coding_agent_polyglot.py", workdir='/')

    # 重建 patch chain：找到从 parent_commit 到 'initial' 的所有祖先 patch
    # get_model_patch_paths 返回有序列表（最老的 patch 在前，最新的在后）
    patch_files = get_model_patch_paths(root_dir, os.path.join(output_dir, '../'), parent_commit)

    # 按顺序 apply 所有父代 patch（重现父代 agent 的代码状态）
    # run_baseline='no_selfimprove' 时跳过，始终使用初始代码
    if run_baseline not in ['no_selfimprove']:
        for patch_file in patch_files:
            copy_to_container(container, patch_file, '/dgm/parent_patch.txt')
            exec_result = container.exec_run("/bin/sh -c 'patch -p1 < /dgm/parent_patch.txt'", workdir='/dgm')
            log_container_output(exec_result)
            exec_result = container.exec_run("rm /dgm/parent_patch.txt", workdir='/dgm')
            log_container_output(exec_result)

    # git commit 当前状态：获得干净的 base commit hash
    # 这确保 model_patch.diff 只包含本次自改进的变化
    exec_result = container.exec_run("git add --all", workdir='/dgm/')
    log_container_output(exec_result)
    exec_result = container.exec_run("git -c user.name='user' -c user.email='you@example.com' commit -m 'a nonsense commit message'", workdir='/dgm/')
    log_container_output(exec_result)
    commit_output = exec_result.output.decode('utf-8')
    # git commit 输出格式：`[master (root-commit) <hash>] a nonsense commit message`
    # 取第二个词（去掉方括号），得到 7 位 commit hash
    commit_hash = commit_output.split()[1].strip("[]")

    # 重新安装 requirements（父代可能修改了 requirements.txt）
    exec_result = container.exec_run("python -m pip install -r /dgm/requirements.txt", workdir='/')
    log_container_output(exec_result)

    # 调用 diagnose_problem：让 o1 分析日志，生成改进任务描述
    if entry:
        safe_log(f"Task to improve: {entry}")
        problem_statement = diagnose_problem(entry, parent_commit, root_dir, out_dir_base, patch_files=patch_files, polyglot=polyglot)
        safe_log(f"problem_statement: {problem_statement}")
    else:
        safe_log("No entry provided. Exiting.")
        cleanup_container(container)
        save_metadata(metadata, output_dir)
        return metadata

    metadata['entry'] = entry
    metadata['problem_statement'] = problem_statement
    if not problem_statement:
        safe_log("Failed to diagnose the problem statement. Exiting.")
        cleanup_container(container)
        save_metadata(metadata, output_dir)
        return metadata

    # 在容器中运行 coding_agent.py --self_improve
    # agent 会读取 problem_statement，然后修改 /dgm/ 中的代码文件
    safe_log("Running self-improvement")
    chat_history_file_container = "/dgm/self_evo.md"
    test_description = get_test_description(swerepo=False)  # 自改进时不是修复 SWE issue
    env_vars = {
        "ANTHROPIC_API_KEY": os.getenv('ANTHROPIC_API_KEY'),
        "AWS_REGION": os.getenv('AWS_REGION'),
        "AWS_REGION_NAME": os.getenv('AWS_REGION_NAME'),
        "AWS_ACCESS_KEY_ID": os.getenv('AWS_ACCESS_KEY_ID'),
        "AWS_SECRET_ACCESS_KEY": os.getenv('AWS_SECRET_ACCESS_KEY'),
        "OPENAI_API_KEY": os.getenv('OPENAI_API_KEY'),
    }
    # timeout 1800 = 30 分钟；自改进不需要像 issue 修复那样花 9 小时
    # --git_dir /dgm/：agent 修改自己所在目录的代码
    # --self_improve：告诉 agent 这是自改进模式（instance_id = 'dgm'）
    cmd = [
        "timeout", "1800",
        "python", "/dgm/coding_agent.py",
        "--problem_statement", problem_statement,
        "--git_dir", "/dgm/",
        "--chat_history_file", chat_history_file_container,
        "--base_commit", commit_hash,
        "--outdir", "/dgm/",
        "--test_description", test_description,
        "--self_improve",
    ]
    exec_result = container.exec_run(cmd, environment=env_vars, workdir='/')
    log_container_output(exec_result)

    # 从容器取回运行日志和 model_patch.diff
    chat_history_file = os.path.join(output_dir, "self_evo.md")
    copy_from_container(container, chat_history_file_container, chat_history_file)
    model_patch_file = os.path.join(output_dir, "model_patch.diff")
    copy_from_container(container, "/dgm/model_patch.diff", model_patch_file)

    # 验证 model_patch.diff 是否有效（存在且非空）
    try:
        if not os.path.exists(model_patch_file):
            raise Exception("Model patch file is empty or does not exist")
        with open(model_patch_file, 'r') as f:
            patch_content = f.read()
            if not patch_content.strip():
                raise Exception("Model patch file is empty")
    except Exception as e:
        safe_log(f"Failed to read model patch file: {str(e)}")
        save_metadata(metadata, output_dir)
        return metadata

    # 将本次 patch 添加到 patch chain 末尾（供后续评估使用）
    patch_files.append(model_patch_file)

    # 评估结束后清理容器（避免资源泄漏）
    cleanup_container(container)

    # 评估本次自改进的效果
    model_patch_exists = os.path.exists(model_patch_file)
    metadata['model_patch_exists'] = model_patch_exists
    model_patch_notempty = os.path.getsize(model_patch_file) > 0
    metadata['model_patch_notempty'] = model_patch_notempty
    model_name_or_path = run_id
    if model_patch_exists and model_patch_notempty:
        try:
            if not polyglot:
                run_harness_swe(entry, model_name_or_path, patch_files, num_evals, output_dir, metadata, run_id, test_more_threshold, test_task_list, test_task_list_more)
            else:
                run_harness_polyglot(entry, model_name_or_path, patch_files, num_evals, output_dir, metadata, run_id, test_more_threshold, test_task_list, test_task_list_more)
        except Exception as e:
            safe_log(f"Error while evaluating the self-improvement: {e}")

    # 可选：评估后诊断（让 o1 分析改进是否真的有效）
    if post_improve_diagnose:
        safe_log("Diagnosing the self-improvement")
        metadata['is_compiled'] = is_compiled_self_improve(metadata)
        if metadata['is_compiled']:
            safe_log("The self-improvement succeed to be complied")
            improvement_diagnosis = diagnose_improvement(
                entry, parent_commit, root_dir,
                model_patch_file, out_dir_base, run_id,
                patch_files=patch_files,
            )
            metadata['improvement_diagnosis'] = improvement_diagnosis
            safe_log(f"Improvement diagnosis: {improvement_diagnosis}")
        else:
            safe_log("The self-improvement fail to be complied")
            metadata['improvement_diagnosis'] = "Fail to complied. Ignore this."

    save_metadata(metadata, output_dir)
    return metadata


def main():
    """
    命令行入口：直接运行单次自改进步骤（调试或手动触发用）。

    通常不直接调用这个入口，而是通过 DGM_outer.py 的演化循环调用 self_improve 函数。
    但在测试或调试单次改进时很有用。

    注意：main 中有 `cp -r initial/ {args.output_dir}` 步骤，
    确保初始版本的评估结果被复制到工作目录。
    """
    parser = argparse.ArgumentParser(description="Self-improvement step for the repository.")
    parser.add_argument('--parent_commit', default="initial", type=str)
    parser.add_argument('--output_dir', default="./output_selfimprove", type=str)
    parser.add_argument('--force_rebuild', default=False, action='store_true')
    parser.add_argument('--num_evals', default=1, type=int)
    parser.add_argument('--no_post_improve_diagnose', default=False, action='store_true')
    parser.add_argument('--entry', default="django__django-10999", type=str)
    parser.add_argument('--test_task_list', default=None, type=str)
    args = parser.parse_args()

    # 将初始版本的评估结果复制到工作目录（self_improve 依赖这些结果）
    os.system(f"cp -r initial/ {args.output_dir}")

    metadata = self_improve(
        parent_commit=args.parent_commit,
        output_dir=args.output_dir,
        force_rebuild=args.force_rebuild,
        num_evals=args.num_evals,
        post_improve_diagnose=not args.no_post_improve_diagnose,
        entry=args.entry,
        test_task_list=args.test_task_list,
    )


if __name__ == "__main__":
    main()
