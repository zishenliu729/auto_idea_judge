import tarfile
import io
import logging
import threading
from typing import Union, Optional
from pathlib import Path
import docker


# ============================================================
# swe_bench/utils.py
# SWE-bench 评估流程中的 Docker 容器工具函数。
#
# 与 utils/docker_utils.py 的关系：
#   两个文件结构几乎相同，但属于不同的模块：
#   - utils/docker_utils.py：用于 DGM 自改进步骤（自改进容器的文件传输）
#   - swe_bench/utils.py：用于 SWE-bench 评估（SWE 容器的文件传输）
#
#   主要差异：
#   1. logger 名称：此文件用 'docker_logger_{thread_id}'，
#      docker_utils.py 用 'selfimprove_logger_{thread_id}'
#   2. log_container_output 多了 raise_error 参数（默认 True），
#      允许调用方选择是否在退出码非零时抛出异常
#
# 线程安全设计与 docker_utils.py 相同，见该文件的详细注释。
# ============================================================


# 线程本地存储：每个线程独立的 logger 实例，避免并行评估时日志混淆
_thread_local = threading.local()


def get_thread_logger():
    """获取当前线程对应的 logger 实例。"""
    return getattr(_thread_local, 'logger', None)


def setup_logger(log_file):
    """
    为当前线程创建线程私有的文件日志器。

    logger 名称用 'docker_logger_{thread_id}' 区分（与 selfimprove_logger_ 不同，
    防止两个模块的 logger 冲突）。

    Args:
        log_file (str): 日志文件路径。

    Returns:
        logging.Logger: 已配置的线程私有 logger 实例。
    """
    thread_id = threading.get_ident()
    logger_name = f'docker_logger_{thread_id}'  # 注意：与 docker_utils.py 中的名称不同
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # 清除旧 handler（防止重复调用时日志重复写入）
    for handler in logger.handlers:
        logger.removeHandler(handler)

    # 文件 handler + 显式锁（高并发写入安全）
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    handler.stream.lock = threading.Lock()

    formatter = logging.Formatter('%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    _thread_local.logger = logger

    return logger


def safe_log(message: str, level: int = logging.INFO):
    """线程安全日志输出，优先使用线程私有 logger，无 logger 时降级到 print。"""
    logger = get_thread_logger()
    if logger:
        logger.log(level, message)
    else:
        print(f"Warning: No logger found for thread {threading.get_ident()}")


def remove_existing_container(client, container_name):
    """
    检查并移除同名的已有 Docker 容器（防止名称冲突）。

    Args:
        client: docker.DockerClient 实例。
        container_name (str): 目标容器名称。

    Raises:
        docker.errors.APIError: Docker API 出错时向上抛出。
    """
    try:
        existing_container = client.containers.get(container_name)
        safe_log(f"Removing existing container with name {container_name}")
        existing_container.stop()
        existing_container.remove()
    except docker.errors.NotFound:
        safe_log(f"No existing container with name {container_name} found.")
    except docker.errors.APIError as e:
        safe_log(f"Error removing existing container {container_name}: {e}", logging.ERROR)
        raise


def create_archive(path: Union[str, Path], data: Optional[bytes] = None) -> bytes:
    """
    创建内存 tar 归档，用于 Docker 容器文件传输。

    Docker SDK 的 put_archive/get_archive 只接受 tar 流格式，
    此函数将文件或目录打包为内存 tar bytes，避免创建临时文件。

    Args:
        path (Union[str, Path]): 文件在 tar 内的路径名（单文件模式），
                                 或待打包的源目录路径（目录模式）。
        data (Optional[bytes]): 单文件时的文件内容字节；目录模式时为 None。

    Returns:
        bytes: 完整的 tar 归档字节串。
    """
    tar_stream = io.BytesIO()
    with tarfile.open(fileobj=tar_stream, mode='w') as tar:
        if data is not None:
            # 单文件：手动构造 TarInfo，指定文件名和大小
            tarinfo = tarfile.TarInfo(name=str(path))
            tarinfo.size = len(data)
            tar.addfile(tarinfo, io.BytesIO(data))
        else:
            # 目录：递归添加，arcname 用目录本身的名字作为 tar 内根节点
            path = Path(path)
            arcname = path.name
            tar.add(path, arcname=arcname)

    tar_stream.seek(0)
    return tar_stream.read()


def copy_to_container(container, source_path: Union[str, Path], dest_path: Union[str, Path]) -> None:
    """
    将本地文件或目录复制到 Docker 容器内指定路径。

    实现：create_archive() 打包 → container.put_archive() 传入容器并解压。

    Args:
        container: docker.Container 实例。
        source_path (Union[str, Path]): 本地源路径（文件或目录）。
        dest_path (Union[str, Path]): 容器内目标路径。

    Raises:
        FileNotFoundError: 源路径不存在时抛出。
        Exception: 传输失败时抛出。
    """
    source_path = Path(source_path)
    dest_path = Path(dest_path)

    try:
        if not source_path.exists():
            raise FileNotFoundError(f"Source path not found: {source_path}")

        if source_path.is_file():
            # 文件模式：目标目录 = dest_path 的父目录
            container_dest_dir = str(dest_path.parent)
            archive_path = dest_path.name
            with open(source_path, 'rb') as source_file:
                data = source_file.read()
            archive = create_archive(archive_path, data)
        else:
            # 目录模式：递归打包整个目录
            container_dest_dir = str(dest_path.parent)
            archive = create_archive(source_path)

        # 确保容器内目标目录存在
        container.exec_run(f"mkdir -p {container_dest_dir}")

        safe_log(f"Copying {source_path} to container at {dest_path}")
        success = container.put_archive(container_dest_dir, archive)

        if not success:
            raise Exception(f"Failed to copy {source_path} to container")

        safe_log(f"Successfully copied {source_path} to container")

    except Exception as e:
        safe_log(f"Error copying to container: {e}", logging.ERROR)
        raise


def copy_from_container(container, source_path: Union[str, Path], dest_path: Union[str, Path]) -> None:
    """
    将 Docker 容器内的文件或目录复制到本地指定路径。

    实现：container.get_archive() 取出 tar 流 → 内存解压 → 写入本地路径。

    注意：`stat -f '%HT'` 是 macOS/BSD 的 stat 语法，Linux 容器中可能需要
    改用 `stat -c '%F'`（'-c' 参数和格式字符串不同）。

    Args:
        container: docker.Container 实例。
        source_path (Union[str, Path]): 容器内源路径。
        dest_path (Union[str, Path]): 本地目标路径。

    Raises:
        FileNotFoundError: 容器内源路径不存在时抛出。
        Exception: 传输失败时抛出。
    """
    source_path = Path(source_path)
    dest_path = Path(dest_path)

    try:
        # 检查容器内源路径是否存在
        result = container.exec_run(f"test -e {source_path}")
        if result.exit_code and result.exit_code != 0:
            raise FileNotFoundError(f"Source path not found in container: {source_path}")

        # 判断是文件还是目录（注意：这里用 BSD stat 格式，Linux 容器可能不兼容）
        result = container.exec_run(f"stat -f '%HT' {source_path}")
        is_file = result.output.decode().strip() == 'Regular File'

        dest_path.parent.mkdir(parents=True, exist_ok=True)

        safe_log(f"Copying from container {source_path} to local path {dest_path}")

        # 获取 tar 流（bits 是分块迭代器，stat 是文件元数据）
        bits, stat = container.get_archive(str(source_path))
        archive_data = b''.join(bits)  # 合并所有分块

        stream = io.BytesIO(archive_data)
        with tarfile.open(fileobj=stream, mode='r') as tar:
            if is_file:
                # 单文件：直接写入目标路径
                member = tar.getmembers()[0]
                with tar.extractfile(member) as source_file:
                    data = source_file.read()
                    with open(dest_path, 'wb') as dest_file:
                        dest_file.write(data)
            else:
                # 目录：解压到父目录，按需重命名
                tar.extractall(path=str(dest_path.parent))
                extracted_path = dest_path.parent / Path(stat['name']).name
                if extracted_path != dest_path and extracted_path.exists():
                    extracted_path.rename(dest_path)

        safe_log(f"Successfully copied from container to {dest_path}")

    except Exception as e:
        safe_log(f"Error copying from container: {e}", logging.ERROR)
        raise


def log_container_output(exec_result, raise_error=True):
    """
    记录容器命令执行的输出，可选择是否在退出码非零时抛出异常。

    与 docker_utils.log_container_output 的区别：
      此版本多了 raise_error 参数（默认 True）。
      调用方可以传 raise_error=False 来允许容器命令失败（仅记录不抛出），
      这在某些"预期可能失败"的场景下很有用（如探测性命令）。

    Args:
        exec_result: docker.Container.exec_run() 的返回值。
        raise_error (bool): 退出码非零时是否抛出 Exception，默认 True。

    Raises:
        Exception: 退出码非零且 raise_error=True 时抛出。
    """
    if isinstance(exec_result.output, bytes):
        # 非流式：整体解码一次记录
        safe_log(f"Container output: {exec_result.output.decode()}")
    else:
        # 流式：逐块解码记录
        for chunk in exec_result.output:
            if chunk:
                safe_log(f"Container output: {chunk.decode().strip()}")

    # 根据 raise_error 参数决定是否在失败时抛出异常
    if raise_error:
        if exec_result.exit_code and exec_result.exit_code != 0:
            error_msg = f"Script failed with exit code {exec_result.exit_code}"
            safe_log(error_msg, logging.ERROR)
            raise Exception(error_msg)
