import argparse
import datetime
import json
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, TimeoutError

from prompts.self_improvement_prompt import find_selfimprove_eval_logs
from self_improve_step import self_improve
from utils.common_utils import load_json_file
from utils.docker_utils import setup_logger
from utils.evo_utils import load_dgm_metadata, is_compiled_self_improve


# ============================================================
# DGM_outer.py
# Darwin Gödel Machine 外层演化循环——负责管理整个自改进演化过程。
#
# 核心概念：
#   Archive（档案库）：保存所有"合格"的 agent 版本（commit hash 列表）
#     - 每个版本用其 run_id（时间戳字符串）标识
#     - 特殊初始版本用 'initial' 标识
#     - 演化通过"选父代 → 改进 → 评估 → 更新档案"的循环进行
#
# 演化循环（每代 generation）：
#   1. choose_selfimproves：从 archive 中选父代，决定本代要尝试哪些改进
#   2. 并行运行 num self_improve 步骤（每个步骤 = 一次完整的改进+评估）
#   3. filter_compiled：过滤掉未通过编译/评估验证的改进结果
#   4. update_archive：将合格的新版本加入 archive
#   5. 保存 dgm_metadata.jsonl（记录每代的演化历史）
#
# 父代选择策略（choose_selfimproves_method）：
#   - random：随机选父代（基线方法）
#   - score_prop：按性能评分的 sigmoid 值加权随机选取（高分更可能被选中）
#   - score_child_prop（默认）：性能评分 × 1/(1+children_count)
#     （平衡高分和探索：高分但子代少的版本更有价值）
#   - best：直接选性能最高的几个版本
#
# 任务选择策略（SWE-bench）：
#   对每个选中的父代，有 25% 概率选择特殊任务类型：
#   - solve_empty_patches：解决 agent 不生成 patch 的问题
#   - solve_stochasticity：提升 agent 的稳定性
#   - solve_contextlength：解决上下文超限问题
#   否则随机从"未解决"的 issue 中选一个
# ============================================================


def initialize_run(output_dir, prevrun_dir=None, polyglot=False, judge=False):
    """
    初始化或恢复 DGM 运行（设置 archive 和起始代数）。

    两种模式：
      1. 全新运行（prevrun_dir=None）：
         - archive 从 ['initial'] 开始
         - 将初始版本的评估结果复制到 output_dir/initial/
         - initial/ 目录由外部预先生成（需要在初始 agent 上运行评估）

      2. 恢复运行（prevrun_dir 为上次 DGM run 的目录）：
         - 从 dgm_metadata.jsonl 读取最后一行的 archive 状态
         - start_gen_num = 上次 generation + 1（避免重复代数）
         - 适用于意外中断后的断点续跑

    initial_folder_name 区分：
      'initial'（SWE-bench）vs 'initial_polyglot'（Polyglot）vs 'initial_judge'（judge 模式），
      三个模式各自的初始评估结果格式不同，用不同目录名避免混淆。

    Args:
        output_dir (str): 本次 DGM run 的输出目录。
        prevrun_dir (str | None): 上次运行的目录（恢复时传入）。
        polyglot (bool): 是否为 Polyglot 模式。
        judge (bool): 是否为 judge 模式。

    Returns:
        tuple[list, int]: (archive, start_gen_num)
    """
    start_gen_num = 0
    if not prevrun_dir:
        archive = ['initial']
    else:
        # 读取上次运行的最后一代 metadata，恢复 archive 状态
        metadata_path = os.path.join(prevrun_dir, "dgm_metadata.jsonl")
        metadata = load_dgm_metadata(metadata_path, last_only=True)
        archive = metadata['archive']
        start_gen_num = metadata['generation'] + 1

    # judge 模式使用独立的初始目录，避免与 SWE/Polyglot 的初始评估混淆
    # initial_judge/ 需要在运行 DGM 前预先生成（运行一次 evaluate.py 得到初始 accuracy）
    if judge:
        initial_folder_name = 'initial_judge'
    elif polyglot:
        initial_folder_name = 'initial_polyglot'
    else:
        initial_folder_name = 'initial'

    # 若未恢复已有运行且初始目录不存在，则从预置目录复制
    # 始终检查 output_dir/initial（archive key 固定为 'initial'），
    # 而非 output_dir/{initial_folder_name}——两者不同时会导致每次都重复覆盖
    if not prevrun_dir and not os.path.exists(os.path.join(output_dir, "initial")):
        if os.path.exists(initial_folder_name):
            os.system(f"cp -r {initial_folder_name}/ {output_dir}/initial")
        else:
            raise RuntimeError("Error: Need to properly configure evaluation results for the initial version.")

    return archive, start_gen_num


def any_exceeding_context_length(output_dir, commit_id, instance_ids):
    """
    检测指定版本的评估日志中是否有 issue 遇到了上下文超限错误。

    上下文超限是严重问题：当错误出现两次（重复报错）时，
    说明 agent 没有对该错误做任何处理，而是陷入了无限重试。
    这是 choose_selfimproves 中选择 'solve_contextlength' 任务的依据。

    判断方式：在日志中查找连续两行相同的错误信息，
    这比"出现一次"更严格，避免误报（偶发的 API 错误不算）。

    Args:
        output_dir (str): 评估结果目录。
        commit_id (str): agent 代码版本标识。
        instance_ids (list[str]): 要检查的 issue ID 列表（通常是 empty + unresolved）。

    Returns:
        bool: True 表示至少一个 issue 遇到了重复的上下文超限错误。
    """
    for instance_id in instance_ids:
        md_logs, _, _, _ = find_selfimprove_eval_logs(instance_id, output_dir, commit_id, filter=False)
        md_log = md_logs[0]
        error_str = "Error in get_response_withtools: Error code: 400 - {'message': 'Input is too long for requested model.'}"
        # 连续出现两次相同错误 = agent 没有处理，陷入循环
        if f'{error_str}\n{error_str}' in md_log:
            return True
    return False


def choose_selfimproves(output_dir, archive, selfimprove_size, method='random', run_baseline=None, polyglot=False, judge=False):
    """
    为当前代选择自改进尝试：(父代 commit, 改进任务) 的列表。

    父代选择（candidates 构建）：
      读取每个 archive 成员的 metadata.json，提取：
      - accuracy_score：当前版本的解决率（主要性能指标，SWE/judge 通用）
      - total_unresolved_ids：未解决的 issue 列表（仅 SWE/Polyglot 模式）
      - total_emptypatch_ids：生成了空 patch 的 issue 列表（仅 SWE/Polyglot 模式）
      - children_count：该版本已有多少子代版本（影响 score_child_prop）

    judge 模式的简化：
      - candidates 只读取 accuracy_score 和 children_count，无 SWE 专属字段
      - entry 固定为 'improve_judge'（不需要选具体 issue，改进目标是整体准确率）
      - 父代选择策略（score_prop、score_child_prop 等）与 SWE 模式相同

    SWE-bench 任务选择的特殊逻辑（Polyglot/judge 不适用）：
      按优先级依次尝试（使用 continue 跳过后续判断）：
      1. 如果空 patch 比例 >= 10% AND 随机数 < 0.25 → 选 solve_empty_patches
      2. 如果随机数 < 0.25 → 选 solve_stochasticity
      3. 如果有上下文超限 AND 随机数 < 0.25 → 选 solve_contextlength
      4. 否则：随机选一个未解决的 issue

    Args:
        output_dir (str): 评估结果目录。
        archive (list[str]): 当前的档案库（commit hash 列表）。
        selfimprove_size (int): 本代要并行尝试的改进数量。
        method (str): 父代选择方法。
        run_baseline (str | None): 基线模式（'no_darwin' 时强制选最后一个版本）。
        polyglot (bool): 是否为 Polyglot 模式。
        judge (bool): 是否为 judge 模式。

    Returns:
        list[tuple[str, str]]: [(parent_commit, entry_id), ...] 的列表，
                               每个元素代表一次改进尝试。
    """
    selfimprove_entries = []

    # 构建候选版本字典（只包含成功评估的版本）
    candidates = {}
    for commit in archive:
        try:
            metadata_path = os.path.join(output_dir, commit, "metadata.json")
            metadata = load_json_file(metadata_path)
            if judge:
                # judge 模式：overall_performance 只有 {accuracy_score, kappa, total_n}
                # 不读取 SWE 专属字段（total_unresolved_ids 等），避免 KeyError
                candidates[commit] = {
                    'accuracy_score': metadata['overall_performance']['accuracy_score'],
                    'children_count': 0,
                }
            else:
                candidates[commit] = {
                    'accuracy_score': metadata['overall_performance']['accuracy_score'],
                    'total_unresolved_ids': metadata['overall_performance']['total_unresolved_ids'],
                    'total_emptypatch_ids': metadata['overall_performance']['total_emptypatch_ids'],
                    'total_resolved_ids': metadata['overall_performance']['total_resolved_ids'],
                    'children_count': 0,  # 初始为 0，后续累积
                }
            # 累积父代的 children_count（自身排除 'initial'，避免键不存在）
            if commit != 'initial':
                parent_commit = metadata['parent_commit']
                candidates[parent_commit]['children_count'] += 1
        except Exception as e:
            # 评估失败（代码编译错误、SWE 评估异常等），排除出候选列表
            print(f"{commit} not eligible for being a parent: {e}")
            continue

    # 基线：no_darwin 不使用进化选择，始终选最后一个版本
    if run_baseline == 'no_darwin':
        commits = list(candidates.keys())
        parent_commits = commits[-1:]
    elif method == 'score_prop':
        # sigmoid 加权：性能越高越可能被选中
        # sigmoid(10*(score - 0.5)) 在 score=0.5 时为 0.5，score=1 时接近 1
        commits = list(candidates.keys())
        scores = [candidates[commit]['accuracy_score'] for commit in commits]
        scores = [1 / (1 + math.exp(-10*(score-0.5))) for score in scores]
        probabilities = [score / sum(scores) for score in scores]
        print(commits)
        parent_commits = random.choices(commits, probabilities, k=selfimprove_size)
    elif method == 'score_child_prop':
        # sigmoid 性能得分 × 1/(1+children_count)
        # 鼓励探索高性能但子代少的版本（避免重复改进已被充分探索的版本）
        commits = list(candidates.keys())
        scores = [candidates[commit]['accuracy_score'] for commit in commits]
        scores = [1 / (1 + math.exp(-10*(score-0.5))) for score in scores]
        children_counts = [candidates[commit]['children_count'] for commit in commits]
        children_counts = [1 / (1 + count) for count in children_counts]
        probabilities = [score * count for score, count in zip(scores, children_counts)]
        probabilities = [prob / sum(probabilities) for prob in probabilities]
        parent_commits = random.choices(commits, probabilities, k=selfimprove_size)
    elif method == 'best':
        # 直接选性能最高的版本
        sorted_commits = sorted(candidates, key=lambda x: candidates[x]['accuracy_score'])
        parent_commits = sorted_commits[:min(selfimprove_size, len(sorted_commits))]
        if len(parent_commits) < selfimprove_size:
            # 候选不足时允许重复选取
            parent_commits.extend(random.choices(parent_commits, k=selfimprove_size - len(parent_commits)))
    else:
        # 默认：纯随机选择（用于对比实验）
        parent_commits = random.choices(list(candidates.keys()), k=selfimprove_size)

    # 为每个选中的父代确定改进任务
    for parent_commit in parent_commits:
        if judge:
            # judge 模式：任务固定为 'improve_judge'（无需选具体 issue，目标是整体准确率）
            # 每次改进都是针对 judge/ workflow 的整体优化，不区分单个失败样本
            selfimprove_entries.append((parent_commit, 'improve_judge'))
            continue

        empty_ids = candidates[parent_commit]['total_emptypatch_ids']
        resolved_ids = candidates[parent_commit]['total_resolved_ids']
        unresolved_ids = candidates[parent_commit]['total_unresolved_ids']

        if polyglot:
            # Polyglot：优先从未解决/空 patch 的任务中随机选一个
            entry_ids = empty_ids + unresolved_ids
            if not entry_ids:
                # 全部解决了？就从所有任务中随机选
                entry_ids = resolved_ids + empty_ids + unresolved_ids
        else:
            # SWE-bench：按优先级选择特殊任务类型或普通未解决 issue
            num_total_ids = len(empty_ids) + len(resolved_ids) + len(unresolved_ids)

            # 空 patch 比例超过 10% 时，有 25% 概率触发专项改进
            if len(empty_ids) >= 0.1 * num_total_ids and random.random() < 0.25:
                entry = 'solve_empty_patches'
                selfimprove_entries.append((parent_commit, entry))
                continue  # 已选定任务，跳过后续判断

            # 25% 概率触发随机性改进（无论当前是否有问题，鼓励提升稳定性）
            if random.random() < 0.25:
                entry = 'solve_stochasticity'
                selfimprove_entries.append((parent_commit, entry))
                continue

            # 检测到重复上下文超限错误且 25% 概率触发
            if any_exceeding_context_length(output_dir, parent_commit, empty_ids + unresolved_ids) and \
                random.random() < 0.25:
                entry = 'solve_contextlength'
                selfimprove_entries.append((parent_commit, entry))
                continue

            # 正常情况：从未解决的 issue 中随机选一个
            if unresolved_ids == 0:
                continue  # 所有 issue 都解决了，跳过（理论上不会发生）
            entry_ids = unresolved_ids

        entry = random.choice(entry_ids)
        selfimprove_entries.append((parent_commit, entry))

    return selfimprove_entries


def filter_compiled(run_ids, output_dir, num_swe_issues=[], logger=None, judge=False):
    """
    过滤掉未通过"编译"验证的自改进结果。

    SWE/Polyglot 模式的"编译"含义（is_compiled_self_improve 的判断标准）：
      1. metadata 中存在必要的键（patch 文件、评估结果等）
      2. resolved + unresolved > 0（成功运行了评估）
      3. total_submitted >= num_swe_issues 阈值（评估任务数量足够）

    judge 模式的简化：
      is_compiled 字段由 self_improve_step.py 在 post_improve_diagnose 块中写入
      （Step 4 Bug 1 修复），语义为"evaluate.py 是否产出了有效的 accuracy_score"。
      直接读取该字段，不调用 is_compiled_self_improve（含 SWE 专属字段检查）。

    只有通过验证的版本才会被加入 archive，
    这防止了崩溃、编译错误、空 patch 等情况污染档案库。

    Args:
        run_ids (list[str]): 本代所有自改进结果的 run_id 列表。
        output_dir (str): DGM run 的输出目录。
        num_swe_issues (list[int]): 期望的评估任务数量（用于验证完整性，仅 SWE 模式使用）。
        logger: 日志器实例。
        judge (bool): 是否为 judge 模式。

    Returns:
        list[str]: 通过验证的 run_id 列表（可以加入 archive）。
    """
    run_ids_compiled = []

    logger.info(f"num_swe_issues: {num_swe_issues}")
    for run_id in run_ids:
        metadata_path = os.path.join(output_dir, run_id, "metadata.json")
        metadata = load_json_file(metadata_path)
        logger.info(f"{run_id} metadata: {metadata}")
        if judge:
            # judge 模式：直接读取 is_compiled 字段（由 Step 4 的 post_improve_diagnose 块写入）
            # 不调用 is_compiled_self_improve，避免该函数检查 SWE 专属字段时 KeyError
            if metadata.get('is_compiled', False):
                run_ids_compiled.append(run_id)
        else:
            if is_compiled_self_improve(metadata, num_swe_issues=num_swe_issues, logger=logger):
                run_ids_compiled.append(run_id)
    return run_ids_compiled


def get_original_score(output_dir):
    """
    读取初始版本（original DGM）的性能评分。

    这个评分用于 update_archive 的 'keep_better' 策略——
    只有性能超过初始版本（减去 noise_leeway 容差）的改进才加入 archive。

    Args:
        output_dir (str): DGM run 的输出目录（包含 initial/ 子目录）。

    Returns:
        float: 初始版本的 accuracy_score（0 到 1 之间）。
    """
    metadata = load_json_file(os.path.join(output_dir, "initial", "metadata.json"))
    return metadata["overall_performance"]["accuracy_score"]


def update_archive(output_dir, archive, new_ids, method='keep_all', noise_leeway=0.1):
    """
    将本代通过验证的自改进结果更新到 archive 中。

    两种策略：
      1. 'keep_all'（默认）：接受所有通过 filter_compiled 的版本
         - 好处：保留多样性，即使某个版本性能稍低，它可能在不同 issue 上有优势
         - 缺点：archive 会不断增大，计算开销增加

      2. 'keep_better'：只接受性能高于初始版本（减去噪声容差）的版本
         - noise_leeway：容差值（默认 0.1），允许略低于初始性能（考虑评估噪声）
         - 好处：保持 archive 质量，避免性能回归
         - 缺点：过于保守，可能错过有潜力的中间版本

    Args:
        output_dir (str): DGM run 的输出目录。
        archive (list[str]): 当前 archive（会被直接修改）。
        new_ids (list[str]): 本代通过验证的 run_id 列表。
        method (str): 更新策略（'keep_better' 或 'keep_all'）。
        noise_leeway (float): 性能比较时的容差（用于 'keep_better'）。

    Returns:
        list[str]: 更新后的 archive。
    """
    if method == 'keep_better':
        # 获取初始版本的性能基线（减去容差，允许轻微下降）
        original_score = get_original_score(output_dir) - noise_leeway
        for run_id in new_ids:
            metadata = load_json_file(os.path.join(output_dir, run_id, "metadata.json"))
            score = metadata["overall_performance"]["accuracy_score"]
            if score >= original_score:
                archive.append(run_id)
    else:
        # keep_all：无条件接受所有通过验证的版本
        archive += new_ids

    return archive


def get_full_eval_threshold(output_dir, archive):
    """
    计算触发"全量评估"的性能阈值。

    全量评估（big subset）代价高昂，只对"有潜力超过当前最佳"的版本执行。
    阈值 = archive 中所有版本得分的第二高（次高值）。

    设计理由：
      如果一个版本的 small/medium 评估得分超过了 archive 中的次高分，
      它就有可能成为新的最优版本，值得花资源做全量评估。
      使用次高分而非最高分，给了"挑战者"一定的发展空间。

    安全保障：阈值至少为 0.4（避免阈值过低导致过多不必要的全量评估）。

    注意：只有做过"足够全面评估"（>= 90% 的 num_full_eval 任务数）的版本
    才计入 archive_scores，确保比较的公平性。

    Args:
        output_dir (str): DGM run 的输出目录。
        archive (list[str]): 当前 archive（不含 'initial'，'initial' 单独读取）。

    Returns:
        float: 触发全量评估的性能阈值（[0.4, 1.0] 范围内）。
    """
    archive_scores = []
    # 计算"全量评估"所包含的总任务数
    num_full_eval = sum(len(load_json_file(f"./swe_bench/subsets/{size}.json"))
                       for size in ['small', 'medium', 'big'])

    # 将初始版本的得分作为基准
    original_score = get_original_score(output_dir)
    archive_scores.append(original_score)

    for run_id in archive:
        metadata = load_json_file(os.path.join(output_dir, run_id, "metadata.json"))
        total_submitted_instances = metadata["overall_performance"]["total_submitted_instances"]
        # 只考虑做过足够全面评估的版本（90% 阈值，允许少量评估失败）
        if total_submitted_instances < num_full_eval * 0.9:
            continue
        score = metadata["overall_performance"]["accuracy_score"]
        archive_scores.append(score)

    # 取第二高分作为阈值（如果只有一个分数，用它自身）
    threshold = sorted(archive_scores, reverse=True)[1] if len(archive_scores) > 1 else archive_scores[0]
    # 确保阈值至少为 0.4（防止低门槛触发大量全量评估）
    threshold = max(threshold, 0.4)

    return threshold


def main():
    """
    DGM 外层演化循环的命令行入口。

    关键参数说明：
      --max_generation：最大演化代数（默认 80）
      --selfimprove_size：每代并行尝试的改进数（默认 2）
      --selfimprove_workers：并行工作线程数（默认 2，与 selfimprove_size 匹配）
      --choose_selfimproves_method：父代选择策略（默认 score_child_prop）
      --continue_from：从上次中断的运行恢复（传入上次的 output_dir）
      --update_archive：archive 更新策略（keep_all/keep_better）
      --shallow_eval：只做 small 子集评估（不触发 medium 扩展评估）
      --polyglot：使用 Polyglot 任务集（而非 SWE-bench）
      --eval_noise：keep_better 策略的容差值（默认 0.1）
      --post_improve_diagnose：是否在评估后调用 o1 诊断改进效果

    超时保护：
      每个 self_improve 步骤有 1.5 小时的超时保护（future.result(timeout=1.5*60*60)）。
      如果某个线程超时，会记录错误并继续等待其他线程完成。
      注意：future.cancel() 对已在运行的线程无效（ThreadPoolExecutor 的限制），
      但可以防止尚未开始的任务启动。

    演化状态持久化：
      每代结束后将 dgm_metadata.jsonl append 一行，记录：
      - generation：代数
      - selfimprove_entries：本代选择的 (parent, entry) 对
      - children：本代所有改进结果的 run_id
      - children_compiled：通过验证的 run_id
      - archive：更新后的 archive 状态
      使用 JSONL（每行一个 JSON）而非单个 JSON，支持高效的追加写入和增量读取。
    """
    parser = argparse.ArgumentParser(description="Darwin Godel Machine!")
    parser.add_argument("--max_generation", type=int, default=80)
    parser.add_argument("--selfimprove_size", type=int, default=2)
    parser.add_argument("--selfimprove_workers", type=int, default=2)
    parser.add_argument(
        "--choose_selfimproves_method", type=str, default='score_child_prop',
        choices=['random', 'score_prop', 'score_child_prop' 'best'],
    )
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--update_archive", type=str, default='keep_all', choices=['keep_better', 'keep_all'])
    parser.add_argument("--num_swe_evals", type=int, default=1)
    parser.add_argument('--post_improve_diagnose', default=False, action='store_true')
    parser.add_argument("--shallow_eval", default=False, action='store_true')
    parser.add_argument("--polyglot", default=False, action='store_true')
    parser.add_argument("--eval_noise", type=float, default=0.1)
    parser.add_argument("--no_full_eval", default=False, action='store_true')
    parser.add_argument("--run_baseline", type=str, default=None, choices=['no_selfimprove', 'no_darwin'])
    # ── judge 模式参数（Step 6 新增）────────────────────────────────────────
    # --judge：启用 judge 模式，跳过 SWE 评估，改为跑 evaluate.py 测量 rigor_bucket 准确率
    # --judge_data_small/medium：传给 self_improve() 的 judge 数据路径
    #   默认 None 时 self_improve_step.py 会使用 DEFAULT_DATA（train_small.jsonl）
    parser.add_argument("--judge", default=False, action='store_true',
                        help='启用 judge 模式：评估 rigor_bucket 准确率而非 SWE-bench 解决率')
    parser.add_argument("--judge_data_small", default=None, type=str,
                        help='judge 阶段一数据路径（默认：data/soundnessbench_train_small.jsonl）')
    parser.add_argument("--judge_data_medium", default=None, type=str,
                        help='judge 阶段二数据路径（None=不做阶段二）')
    args = parser.parse_args()

    # 确定本次运行的 ID（新运行 = 时间戳，恢复运行 = 上次的 ID）
    if not args.continue_from:
        run_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S_%f")
    else:
        run_id = os.path.basename(args.continue_from)

    output_dir = os.path.join("./output_dgm", run_id)
    os.makedirs(output_dir, exist_ok=True)

    # 初始化 archive 和起始代数
    archive, start_gen_num = initialize_run(
        output_dir, prevrun_dir=args.continue_from, polyglot=args.polyglot, judge=args.judge,
    )

    # 加载评估子集（small=12, medium=50 个 issue）
    # judge 模式不依赖 SWE/Polyglot 子集，数据路径通过 --judge_data_small/medium 传入
    if args.judge:
        swe_issues_sm = []
        swe_issues_med = []
    elif not args.polyglot:
        swe_issues_sm = load_json_file("./swe_bench/subsets/small.json")
        swe_issues_med = load_json_file("./swe_bench/subsets/medium.json")
    else:
        swe_issues_sm = load_json_file("./polyglot/subsets/small.json")
        swe_issues_med = load_json_file("./polyglot/subsets/medium.json")

    logger = setup_logger(os.path.join(output_dir, "dgm_outer.log"))
    logger.info(f"Starting DGM run {run_id} with arguments: {vars(args)}")
    logger.info(f"Archive: {archive}")

    # 触发第二阶段评估的阈值：small 子集上解决率 >= 40%
    test_more_threshold = 0.4

    # 主演化循环（最多 max_generation 代）
    for gen_num in range(start_gen_num, args.max_generation):
        # 选择本代的改进任务
        selfimprove_entries = choose_selfimproves(
            output_dir, archive, args.selfimprove_size,
            method=args.choose_selfimproves_method,
            run_baseline=args.run_baseline,
            polyglot=args.polyglot,
            judge=args.judge,
        )
        logger.info(f"Self-improve entries for generation {gen_num}: {selfimprove_entries}")

        # 并行执行本代所有改进步骤
        selfimprove_ids = []
        with ThreadPoolExecutor(max_workers=args.selfimprove_workers) as executor:
            futures = [
                executor.submit(
                    self_improve,
                    parent_commit=parent_commit,
                    output_dir=output_dir,
                    force_rebuild=False,
                    num_evals=args.num_swe_evals,
                    post_improve_diagnose=args.post_improve_diagnose,
                    entry=entry,
                    test_task_list=None if args.judge else swe_issues_sm,
                    test_more_threshold=None if (args.shallow_eval or args.judge) else test_more_threshold,
                    test_task_list_more=None if (args.shallow_eval or args.judge) else swe_issues_med,
                    polyglot=args.polyglot,
                    # judge 模式不做全量评估（get_full_eval_threshold 依赖 SWE 专属字段）
                    full_eval_threshold=None if (args.no_full_eval or args.judge) else get_full_eval_threshold(output_dir, archive),
                    run_baseline=args.run_baseline,
                    # judge 模式新增参数：传入 judge 标志和数据路径
                    judge=args.judge,
                    judge_data_small=args.judge_data_small if args.judge else None,
                    judge_data_medium=args.judge_data_medium if args.judge else None,
                )
                for parent_commit, entry in selfimprove_entries
            ]

            # 按完成顺序收集结果（as_completed 避免阻塞等待最慢的任务）
            for future in as_completed(futures):
                try:
                    # 1.5 小时超时保护（整个 self_improve 流程，包括评估）
                    metadata = future.result(timeout=1.5*60*60)
                    selfimprove_ids.append(metadata['run_id'])
                except TimeoutError:
                    logger.error("Self-improvement attempt timed out.")
                    future.cancel()
                except Exception as e:
                    import traceback
                    logger.error(f"Self-improvement step failed: {e}")
                    logger.error(f"Traceback:\n{traceback.format_exc()}")

        # 过滤掉未通过验证的改进结果
        logger.info(f"Updating archive for generation {gen_num}")
        selfimprove_ids_compiled = filter_compiled(
            selfimprove_ids,
            output_dir,
            # judge 模式不用 num_swe_issues 阈值（直接看 is_compiled 字段）
            num_swe_issues=[] if args.judge else (
                [len(swe_issues_sm)] if args.shallow_eval else [len(swe_issues_sm), len(swe_issues_med)]
            ),
            logger=logger,
            judge=args.judge,
        )

        # 将通过验证的版本加入 archive
        archive = update_archive(output_dir, archive, selfimprove_ids_compiled, method=args.update_archive, noise_leeway=args.eval_noise)

        # 追加写入 dgm_metadata.jsonl（每代一行，支持断点续跑和演化历史分析）
        with open(os.path.join(output_dir, "dgm_metadata.jsonl"), "a") as f:
            f.write(json.dumps({
                "generation": gen_num,
                "selfimprove_entries": selfimprove_entries,
                "children": selfimprove_ids,               # 所有尝试（含失败）
                "children_compiled": selfimprove_ids_compiled,  # 通过验证的
                "archive": archive,                         # 更新后的档案库
            }, indent=2) + "\n")


if __name__ == "__main__":
    main()
