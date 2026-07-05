import os
import git
import subprocess


# ============================================================
# git_utils.py
# Git 操作工具函数集合，供 DGM 进化框架和 coding_agent 使用。
#
# 核心用途：
#   - 在 Docker 容器内的仓库上 apply patch（agent 修改代码后保存结果）
#   - 生成当前修改相对某个 commit 的 diff（用于提取 model_patch）
#   - 重置仓库到某个历史 commit（用于复现进化历史中的某个版本）
#   - 从 patch 字符串中过滤或移除特定文件的 diff 块
# ============================================================


def get_git_commit_hash(repo_path='.'):
    """
    获取指定仓库当前 HEAD commit 的完整 SHA hash。

    用途：在自改进流程中，coding_agent 修改完代码后，
    调用此函数获取当前 commit hash 作为这一代进化的唯一标识符
    （即 run_id / child_commit），写入 metadata.json 记录进化历史。

    Args:
        repo_path (str): 仓库根目录路径，默认为当前目录 '.'。

    Returns:
        str | None: 40 位十六进制 commit hash 字符串；
                    若发生任何异常（非 git 仓库、空仓库等）则返回 None。
    """
    try:
        # Load the repository
        repo = git.Repo(repo_path)
        # Get the current commit hash
        commit_hash = repo.head.commit.hexsha
        return commit_hash
    except Exception as e:
        print("Error while getting git commit hash:", e)
        return None

def apply_patch(git_dname, patch_str):
    """
    将 unified diff 格式的 patch 字符串应用到指定 git 仓库。

    使用 `git apply --reject` 而非 `patch` 命令的原因：
      - git apply 能正确处理 git 格式的 diff（包含 a/ b/ 前缀）
      - --reject 参数：即使部分 hunk 无法应用，也继续处理其余 hunk，
        而不是直接失败退出（比 --abort 宽容，适合 agent 产生的不完美 patch）
      - 通过 stdin 传入 patch 内容，避免临时文件的创建和清理

    Args:
        git_dname (str): 目标 git 仓库的根目录路径（容器内路径，如 /testbed/）。
        patch_str (str): unified diff 格式的 patch 字符串。

    Returns:
        None

    副作用：
        - 修改 git_dname 下的文件（应用 patch 的内容）
        - 失败的 hunk 会生成 .rej 文件（--reject 行为）
        - 打印应用结果（成功/失败）到标准输出
    """
    cmd = ["git", "-C", git_dname, "apply", "--reject", "-"]
    result = subprocess.run(
        cmd,
        input=patch_str,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False  # 不抛出异常，手动检查 returncode
    )
    # Check if the patch was applied successfully
    if result.returncode != 0:
        print(f"apply_patch error: Patch did not fully apply. Return code: {result.returncode}, stdout: {result.stdout}, stderr: {result.stderr}")
    else:
        print("apply_patch successful")

def diff_versus_commit(git_dname, commit):
    """
    生成当前工作区相对于指定 commit 的完整 diff，包含未跟踪文件。

    为什么要包含未跟踪文件（untracked files）：
      agent 可能会创建全新的文件（如新增工具模块）而不只是修改现有文件，
      这些新文件不在 git 的跟踪范围内，`git diff` 不会包含它们。
      需要额外用 `git ls-files --others` 找出这些文件，
      再用 `git diff --no-index /dev/null <file>` 生成它们的"新增"风格 diff。

    注意：此函数不修改仓库状态（不 stage、不 commit），只读取。

    Args:
        git_dname (str): git 仓库的根目录路径。
        commit (str): 基准 commit 的 hash 或引用（如 'HEAD'、具体 SHA）。

    Returns:
        str: 完整的 unified diff 字符串，可直接写入 model_patch.diff 文件。
    """
    # 第一步：获取已跟踪文件相对于 commit 的 diff
    diff_cmd = ["git", "-C", git_dname, "diff", commit]
    result = subprocess.run(diff_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    diff_output = result.stdout.decode()

    # 第二步：找出所有未跟踪文件（agent 新创建的文件）
    untracked_files_cmd = ["git", "-C", git_dname, "ls-files", "--others", "--exclude-standard"]
    result = subprocess.run(untracked_files_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    untracked_files = result.stdout.decode().splitlines()

    # 第三步：为每个未跟踪文件生成 "从空文件到当前内容" 的 diff
    for file in untracked_files:
        # Diff untracked file against /dev/null (empty file)
        file_path = os.path.join(git_dname, file)
        devnull = '/dev/null'
        if os.name == 'nt':  # Handle Windows
            devnull = 'NUL'
        # --no-index：让 git diff 比较任意两个文件（不需要它们在 git 仓库中）
        diff_file_cmd = ["git", "-C", git_dname, "diff", "--no-index", devnull, file]
        result = subprocess.run(
            diff_file_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # 把 stderr 合并到 stdout，避免漏掉错误信息
            cwd=git_dname,
            check=False
        )
        diff_file_output = result.stdout.decode('utf-8', errors='replace')
        diff_output += diff_file_output

    return diff_output

def reset_to_commit(git_dname, commit):
    """
    将指定 git 仓库完全重置到某个历史 commit 状态。

    两步操作确保彻底重置：
      1. `git reset --hard <commit>`：将已跟踪文件恢复到 commit 时的状态
      2. `git clean -fd`：删除所有未跟踪文件和目录（agent 新创建的文件）
      只做第一步会遗留未跟踪文件，导致复现的环境不干净。

    用途：DGM 进化框架在评估某个历史版本时，需要把容器内的 /dgm/
    恢复到对应的代码状态，再 apply 该版本的 patch 链。

    Args:
        git_dname (str): git 仓库的根目录路径。
        commit (str): 目标 commit 的 hash 或引用。

    副作用：
        - 不可逆地丢弃工作区的所有修改和未跟踪文件
        - 打印操作结果到标准输出
    """
    # Step 1: Hard-reset tracked files（恢复已跟踪文件到 commit 状态）
    reset_cmd = ["git", "-C", git_dname, "reset", "--hard", commit]
    result_reset = subprocess.run(
        reset_cmd,
        capture_output=True,
        text=True,
        check=False
    )
    if result_reset.returncode != 0:
        print(f"reset_to_commit error: Failed to reset {git_dname} to commit '{commit}'. STDOUT: {result_reset.stdout} STDERR: {result_reset.stderr}")
    else:
        print(f"reset_to_commit successful: {commit}")

    # Step 2: Clean untracked files（删除未跟踪文件和目录，-f 强制，-d 包含目录）
    clean_cmd = ["git", "-C", git_dname, "clean", "-fd"]
    result_clean = subprocess.run(
        clean_cmd,
        capture_output=True,
        text=True,
        check=False
    )
    if result_clean.returncode != 0:
        print(f"reset_to_commit clean error: Failed to clean {git_dname}. STDOUT: {result_clean.stdout} STDERR: {result_clean.stderr}")
    else:
        print(f"reset_to_commit clean successful: {commit}")


def filter_patch_by_files(patch_str, target_files):
    """
    从完整的 patch 字符串中，只保留指定文件的 diff 块，其余全部丢弃。

    使用场景：DGM 自改进时，agent 可能同时修改了 coding_agent.py、
    tools/bash.py、utils/eval_utils.py 等多个文件。如果只想提取
    某个特定文件的修改（例如只看工具文件的改动），用此函数过滤。

    实现原理：逐行扫描 patch，遇到 `diff --git a/... b/...` 行时，
    判断文件名是否在 target_files 列表中；若是则开启"包含模式"，
    后续行（包括 @@ 块和 +/- 行）都保留到结果中，直到下一个文件的 diff 开始。

    Args:
        patch_str (str): 完整的 unified diff 格式 patch 字符串。
        target_files (list[str]): 要保留的文件名列表（不含 a/ b/ 前缀），
                                  如 ['coding_agent.py', 'bash.py']。

    Returns:
        str: 只包含目标文件 diff 块的 patch 字符串。
    """
    lines = patch_str.splitlines()
    filtered_lines = []
    include_block = False  # 标记当前是否处于目标文件的 diff 块中

    for line in lines:
        # When we encounter a new diff block header, check if the block is for any of the target files.
        if line.startswith("diff --git"):
            # 检查这个 diff 块是否对应 target_files 中的某个文件
            # unified diff 格式：`diff --git a/path/to/file b/path/to/file`
            include_block = any(f"a/{target}" in line and f"b/{target}" in line for target in target_files)
        if include_block:
            filtered_lines.append(line)
    return "\n".join(filtered_lines)


def remove_patch_by_files(patch_str, keyword='polyglot'):
    """
    从完整的 patch 字符串中，移除文件名包含指定关键词的所有 diff 块。

    使用场景：DGM 同时维护 SWE-bench 和 Polyglot 两套 coding_agent，
    当自改进 patch 同时涉及两个版本的文件时，可以用此函数
    把 polyglot 相关文件的 diff 从 patch 中剔除，只保留 SWE-bench 部分
    （反之亦然）。

    与 filter_patch_by_files 的区别：
      - filter_patch_by_files：白名单模式，只保留指定文件
      - remove_patch_by_files：黑名单模式，移除匹配关键词的文件

    Args:
        patch_str (str): 完整的 unified diff 格式 patch 字符串。
        keyword (str): 要移除的文件名关键词，大小写不敏感，默认 'polyglot'。

    Returns:
        str: 移除匹配文件 diff 块后的 patch 字符串。
    """
    lines = patch_str.splitlines()
    filtered_lines = []
    include_block = True  # 默认保留，遇到匹配关键词的 diff 块时设为 False

    for line in lines:
        # When we encounter a new diff block header, check if the block contains the keyword
        if line.startswith("diff --git"):
            # 大小写不敏感匹配：只要文件路径包含 keyword 就排除整个 diff 块
            include_block = keyword.lower() not in line.lower()
        if include_block:
            filtered_lines.append(line)

    return "\n".join(filtered_lines)
