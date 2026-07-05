import json
import os

from utils.common_utils import load_json_file, read_file


# ============================================================
# evo_utils.py
# 进化算法相关的工具函数集合，供 DGM_outer.py（主循环）和 self_improve_step.py 使用。
#
# 核心职责：
#   1. load_dgm_metadata：从 JSONL 格式的 metadata 文件中解析进化历史记录
#   2. get_model_patch_paths：沿 parent_commit 链逆向遍历，收集完整 patch 链
#   3. get_all_performance：聚合多个评估结果 JSON 文件，计算整体性能指标
#   4. is_compiled_self_improve：验证某次运行是否完整、有效地完成了自改进
# ============================================================


def load_dgm_metadata(dgm_metadata_path, last_only=False):
    """
    从指定路径加载 DGM 进化元数据文件（JSONL 格式），返回所有历史记录。

    为什么用 JSONL（每行一个 JSON）而不是单个大 JSON 数组：
      - 进化过程中每轮结束后都需要 append 新记录，JSONL 直接追加即可，
        无需读取再修改整个文件（对大文件更安全，不会因中途崩溃损坏全部数据）
      - 但实现上用 '\n{' 分割而非真正的 JSONL（每行独立 JSON），
        说明实际写入时每个 JSON 对象可能跨多行（pretty-print 格式）

    解析策略：以 '\n{' 为分隔符将文件内容拆分成多个 JSON 片段，
    每个片段补回缺失的 '{' 后独立解析。这种方式比逐字节扫描括号配对简单，
    但依赖 JSON 对象以 '{' 开头这一格式约定。

    Args:
        dgm_metadata_path (str): metadata 文件的路径（通常为 dgm_metadata.jsonl）。
        last_only (bool): 若为 True，只返回最后一条记录（最新一代），
                         用于 DGM_outer.py 判断上一代表现时避免加载全部历史。

    Returns:
        list[dict] | dict: last_only=False 时返回所有元数据的列表；
                           last_only=True 时返回最后一条元数据的 dict。

    Raises:
        FileNotFoundError: metadata 文件不存在时抛出。
    """
    # 文件不存在时直接报错，不做静默回退（避免掩盖配置错误）
    if not os.path.exists(dgm_metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {dgm_metadata_path}")
    # 用 read_file（会 strip 首尾空白）读取全文
    content = read_file(dgm_metadata_path)
    # 以 '\n{' 为分隔符拆分：每个 JSON 对象的第一个 '{' 恰好出现在新行开头
    json_entries = content.split('\n{')
    # 解析所有 JSON 片段
    dgm_metadata = []
    for json_entry in json_entries:
        # 除第一个片段外，split 会把开头的 '{' 也截掉，需要补回来
        if not json_entry.startswith('{'):
            json_entry = '{' + json_entry
        # 解析单个元数据记录（dict 格式，包含 run_id、parent_commit、overall_performance 等）
        metadata = json.loads(json_entry)
        dgm_metadata.append(metadata)

    if last_only:
        return dgm_metadata[-1]
    return dgm_metadata

def get_model_patch_paths(root_dir, dgm_dir, parent_commit):
    """
    从指定的 parent_commit 开始，沿 parent_commit 链逆向遍历整个进化历史，
    收集从初始版本到该版本（不含）所经历的所有 patch 文件路径列表。

    为什么需要遍历 patch 链而非直接用 git 历史：
      DGM 的 git 历史只记录主仓库（dgm/）的演进，但每次自改进
      把修改保存为 model_patch.diff 文件，而不是直接 commit 到 coding_agent 所在的 repo。
      要复现某个进化版本，需要将从初始版本开始的所有 patch 依次 apply。

    遍历终止条件：parent_commit == 'initial'，表示已到达进化树的根节点。

    返回的列表已经过反转（从早到晚），直接按顺序 apply 即可得到目标版本的状态。

    Args:
        root_dir (str): DGM 工作区根目录。
        dgm_dir (str): 进化历史记录目录名（如 'swe_bench_verified_mini_oracle'）。
        parent_commit (str): 目标版本的 parent_commit hash（最新一代的父节点）。

    Returns:
        list[str]: 按时间顺序排列的 model_patch.diff 路径列表（最早到最晚）。
    """
    prev_commit = parent_commit
    patch_files = []  # 从最新到最老收集，最后再反转
    while prev_commit != 'initial':
        # 该版本的存档目录：root_dir/dgm_dir/<commit_hash>/
        parent_dir = os.path.join(root_dir, dgm_dir, prev_commit)
        parent_patch_file = os.path.join(parent_dir, "model_patch.diff")
        if os.path.exists(parent_patch_file):
            patch_files.append(parent_patch_file)
        else:
            # patch 文件丢失时不中断，只打印警告（容错：可能是历史记录不完整）
            print(f"Parent patch file not found: {parent_patch_file}")
        # 从该版本的 metadata.json 中读取其父节点的 commit hash，继续往上追溯
        parent_metadata = load_json_file(os.path.join(parent_dir, "metadata.json"))
        prev_commit = parent_metadata.get('parent_commit', 'initial')
    # patch_files 当前是从新到旧的顺序，反转后变为从旧到新，apply 时按此顺序执行
    return patch_files[::-1]  # reverse the list to get the correct order

def get_all_performance(run_keyword, results_dir='./swe_bench'):
    """
    按关键词匹配指定目录下的评估结果 JSON 文件，汇总整体性能指标。

    使用场景：一次评估可能分批运行（如分 3 批各评估 50 个 issue），
    结果分散在多个 JSON 文件中。此函数按 run_keyword 聚合所有批次，
    计算整体的解决率（accuracy_score）和各类 issue 的 ID 列表。

    Args:
        run_keyword (str): 用于匹配文件名的关键词（如 run_id 前缀）。
        results_dir (str): 评估结果 JSON 文件所在的目录，默认 './swe_bench'。

    Returns:
        tuple[list[dict], dict] | tuple[None, None]:
            - performance_results: 每个匹配文件的性能详情列表
            - overall_performance: 跨所有文件的聚合性能指标
            - 找不到匹配文件时返回 (None, None)
    """
    # 筛选目录下文件名包含 run_keyword 且以 .json 结尾的所有文件
    matching_files = [
        f for f in os.listdir(results_dir)
        if f.endswith('.json') and run_keyword in f
    ]

    # 没有匹配文件时提前返回 None（调用方需要检查返回值）
    if not matching_files:
        print(f"No evaluation files found matching the keyword '{run_keyword}'.")
        return None, None

    # 逐文件累加各项统计指标
    performance_results = []
    total_resolved_instances = 0       # 成功解决的 issue 总数
    total_submitted_instances = 0      # 提交了非空 patch 的 issue 总数
    total_unresolved_ids = []          # 提交了 patch 但未解决的 issue ID 列表
    total_resolved_ids = []            # 成功解决的 issue ID 列表
    total_emptypatch_ids = []          # 生成了空 patch（放弃修复）的 issue ID 列表
    for file_name in matching_files:
        eval_agent_path = os.path.join(results_dir, file_name)
        eval_results = load_json_file(eval_agent_path)
        resolved_instances = eval_results.get('resolved_instances', 0)
        submitted_instances = eval_results.get('submitted_instances', 0)
        total_resolved_instances += resolved_instances
        total_submitted_instances += submitted_instances
        # 单文件准确率：解决数 / 提交数（避免除以零）
        accuracy_score = resolved_instances / submitted_instances if submitted_instances > 0 else 0
        # 把文件名和准确率追加到原始 eval_results dict 中，方便调用方按文件查看
        performance_results.append({'file': file_name, 'accuracy_score': accuracy_score, **eval_results})
        total_unresolved_ids.extend(eval_results.get('unresolved_ids', []))
        total_emptypatch_ids.extend(eval_results.get('empty_patch_ids', []))
        total_resolved_ids.extend(eval_results.get('resolved_ids', []))

    # 构建跨所有批次的汇总指标
    overall_performance = {}
    overall_performance['accuracy_score'] = total_resolved_instances / total_submitted_instances if total_submitted_instances > 0 else 0
    overall_performance['total_resolved_instances'] = total_resolved_instances
    overall_performance['total_submitted_instances'] = total_submitted_instances
    overall_performance['files'] = matching_files
    overall_performance['total_unresolved_ids'] = total_unresolved_ids
    overall_performance['total_emptypatch_ids'] = total_emptypatch_ids
    overall_performance['total_resolved_ids'] = total_resolved_ids

    return performance_results, overall_performance

def is_compiled_self_improve(metadata, num_swe_issues=[], logger=None):
    """
    判断某次自改进运行是否有效完成——即成功生成了性能评估结果且评估覆盖了足够多的 issue。

    三个必要条件（全部满足才返回 True）：
      1. overall_performance 字段存在且包含所有必需的统计 key
         （缺失 key 说明评估流程未走完，数据不完整）
      2. resolved + unresolved > 0
         （全为空 patch 说明 coding_agent 在每个 issue 上都放弃了，等同于无效运行）
      3. total_submitted_instances >= num_swe_issues[0]
         （提交数低于下限说明有 issue 的 Docker 环境没跑起来，编译失败了）

    Args:
        metadata (dict): 单次运行的元数据 dict（含 overall_performance 字段）。
        num_swe_issues (list): [最低提交数阈值, ...] —— 目前只用 [0] 位置的值。
        logger: Python logging.Logger 实例，用于记录失败原因。

    Returns:
        bool: True 表示该次运行有效，可以进入进化选择；False 表示无效，丢弃。
    """
    overall_perf = metadata.get('overall_performance', {})
    # 进化框架关心的四个核心字段
    required_keys = ['accuracy_score', 'total_unresolved_ids', 'total_resolved_ids', 'total_emptypatch_ids']

    # 条件 1：必须包含全部必需字段
    if not overall_perf or not all(k in overall_perf for k in required_keys):
        logger.info(f"no required keys")
        return False

    # 条件 2：至少有一个 issue 生成了非空 patch（不管有没有解决）
    num_resolved = len(overall_perf['total_resolved_ids'])
    num_unresolved = len(overall_perf['total_unresolved_ids'])
    if (num_resolved + num_unresolved) == 0:
        logger.info(f"no non-empty patch")
        return False

    # 条件 3：实际评估的 issue 数不少于期望最低值（避免把"只跑了一半"当作有效结果）
    total_evaluated = overall_perf['total_submitted_instances']
    if total_evaluated < num_swe_issues[0]:
        logger.info(f"not match num_issues")
        return False

    return True
