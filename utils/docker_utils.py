import io
import logging
import os
import threading
from pathlib import Path
import tarfile
from typing import Optional, Union
import docker


# ============================================================
# docker_utils.py
# Docker 容器操作工具函数，供 self_improve_step.py 和 swe_bench/harness.py 调用。
#
# 设计核心：
#   - 线程安全日志（threading.local 存储每个线程的 logger，避免并行评估时日志混淆）
#   - 文件传输基于 tar 归档（Docker SDK 的 put_archive/get_archive API 只支持 tar 流）
#   - 容器生命周期管理（构建镜像、启动容器、清理容器）
# ============================================================


# 线程本地存储对象：每个线程各自持有一个 logger 实例，互不干扰
# 用于并行评估多个 SWE-bench issue 时，每个线程的日志写入独立文件
_thread_local = threading.local()


def get_thread_logger():
    """
    获取当前线程对应的 logger 实例。

    返回 None 而不是抛异常，让 safe_log 能优雅降级到 print。

    Returns:
        logging.Logger | None: 当前线程的 logger，未初始化时返回 None。
    """
    return getattr(_thread_local, 'logger', None)


def setup_logger(log_file):
    """
    为当前线程创建一个线程私有的文件日志器。

    为什么用线程 ID 作为 logger 名称：
      Python 的 logging.getLogger(name) 是全局注册表，同名 logger 会复用同一实例。
      用线程 ID 命名确保每个线程拿到不同的 logger，绑定到不同的日志文件，
      避免多线程并行评估时日志内容互相交织。

    FileHandler 的 lock 替换：
      默认的 StreamHandler 使用内部锁，但 FileHandler 在高并发下可能不够安全，
      此处显式为 handler.stream 设置 threading.Lock() 加强写入互斥。

    Args:
        log_file (str): 日志文件路径（通常为 run_dir/run.log）。

    Returns:
        logging.Logger: 已配置的线程私有 logger 实例。
    """
    # 用线程 ID 生成唯一 logger 名，避免与其他线程的 logger 冲突
    thread_id = threading.get_ident()
    logger_name = f'selfimprove_logger_{thread_id}'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # 清除旧的 handler（防止重复调用 setup_logger 时日志重复写入）
    for handler in logger.handlers:
        logger.removeHandler(handler)

    # 创建文件 handler 并加锁，确保多线程写同一文件时不会字节交错
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    handler.stream.lock = threading.Lock()

    # 格式：时间 - 线程名 - 级别 - 消息（包含线程名便于日志追踪）
    formatter = logging.Formatter('%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    # 将 logger 存入线程本地存储，后续 safe_log 通过 get_thread_logger() 取用
    _thread_local.logger = logger

    return logger


def safe_log(message: str, level: int = logging.INFO):
    """
    线程安全的日志输出函数。

    优先使用当前线程的 logger；若 logger 未初始化（如在主线程直接调用），
    则降级到 print 输出警告，不会因缺少 logger 而抛异常。

    Args:
        message (str): 日志消息内容。
        level (int): 日志级别，默认 INFO。
    """
    logger = get_thread_logger()
    if logger:
        logger.log(level, message)
    else:
        print(f"Warning: No logger found for thread {threading.get_ident()}")


def remove_existing_container(client, container_name):
    """
    检查并移除同名的已有容器（防止容器名冲突导致启动失败）。

    为什么需要此步骤：Docker 不允许两个容器同名。若上次运行因异常崩溃而未清理，
    再次启动时会因名称冲突报错。此函数先检查是否存在，存在则 stop + remove。

    Args:
        client: docker.DockerClient 实例。
        container_name (str): 目标容器名称。

    Raises:
        docker.errors.APIError: Docker API 调用出错时向上抛出（不静默吞掉，避免掩盖系统问题）。
    """
    try:
        existing_container = client.containers.get(container_name)
        safe_log(f"Removing existing container with name {container_name}")
        existing_container.stop()
        existing_container.remove()
    except docker.errors.NotFound:
        # 容器不存在是正常情况，不需要任何操作
        safe_log(f"No existing container with name {container_name} found.")
    except docker.errors.APIError as e:
        safe_log(f"Error removing existing container {container_name}: {e}", logging.ERROR)
        raise


def create_archive(path: Union[str, Path], data: Optional[bytes] = None) -> bytes:
    """
    创建内存 tar 归档，用于向 Docker 容器传输文件或目录。

    为什么用 tar：Docker SDK 的 put_archive() / get_archive() 接口只接受 tar 流，
    不能直接传文件字节。此函数把文件/目录打包成内存中的 tar bytes，
    避免创建临时文件。

    两种使用场景：
      1. data 不为 None：单文件模式，把 bytes 内容放进 tar，文件名由 path 指定
      2. data 为 None：目录模式，递归将 path 目录加入 tar，保留目录结构

    Args:
        path (Union[str, Path]): 文件/目录在 tar 归档中的路径名
                                 （单文件模式：目标文件名；目录模式：源目录路径）。
        data (Optional[bytes]): 单文件时的文件内容字节；目录模式时为 None。

    Returns:
        bytes: 完整的 tar 归档字节串，可直接传给 container.put_archive()。
    """
    # 用内存 BytesIO 作为 tar 的写入目标，避免磁盘 I/O
    tar_stream = io.BytesIO()
    with tarfile.open(fileobj=tar_stream, mode='w') as tar:
        if data is not None:
            # 单文件：手动构造 TarInfo（指定文件名和大小），再写入内容
            tarinfo = tarfile.TarInfo(name=str(path))
            tarinfo.size = len(data)
            tar.addfile(tarinfo, io.BytesIO(data))
        else:
            # 目录：tar.add() 递归添加目录下所有文件，arcname 指定 tar 内的根目录名
            path = Path(path)
            arcname = path.name  # 用目录本身的名字作为 tar 内的根节点
            tar.add(path, arcname=arcname)

    tar_stream.seek(0)  # 重置读取位置，让调用方从头读取
    return tar_stream.read()


def build_dgm_container(
        client,
        repo_path='./',
        image_name='app',
        container_name='app-container',
        force_rebuild=False,
    ):
    """
    构建 DGM 应用的 Docker 镜像（若需要）并启动容器。

    两个阶段：
      1. 镜像构建：若 force_rebuild=True 或镜像不存在，则从 repo_path 的 Dockerfile 构建；
         否则复用已有镜像（加快速度）。
      2. 容器启动：以 detach=True（后台运行）启动容器，返回容器对象供后续操作。

    Args:
        client: docker.DockerClient 实例。
        repo_path (str): 包含 Dockerfile 的目录路径，默认当前目录。
        image_name (str): Docker 镜像标签名，默认 'app'。
        container_name (str): 启动的容器名称，默认 'app-container'。
        force_rebuild (bool): 是否强制重新构建镜像（即使已存在同名镜像）。

    Returns:
        docker.Container | None: 成功启动的容器对象；构建或启动失败时返回 None。
    """
    try:
        # 判断是否需要构建：强制重建，或镜像列表中找不到同名 tag。
        # WebIDE Docker bootstrap (2026-07-07): Docker normalizes tag names to
        # e.g. "dgm:latest", so test both the bare image name and explicit tags;
        # otherwise every DGM run rebuilds even when the image is already reusable.
        existing_images = client.images.list()
        expected_tags = {image_name, f"{image_name}:latest"}
        image_exists = any(expected_tags.intersection(set(image.tags)) for image in existing_images)
        if force_rebuild or not image_exists:
            safe_log("Building the Docker image...")
            # rm=True：构建完成后删除中间层容器，节省磁盘空间
            # WebIDE Docker bootstrap (2026-07-07): this nested daemon runs with
            # bridge networking disabled; host network keeps pip/API access working.
            image, logs = client.images.build(path=repo_path, tag=image_name, rm=True, network_mode="host")
            for log_entry in logs:
                if 'stream' in log_entry:
                    safe_log(log_entry['stream'].strip())
            safe_log("Image built successfully.")
        else:
            safe_log(f"Docker image '{image_name}' already exists. Skipping build.")
            # 从已有镜像列表中找到目标镜像（用 next 取第一个匹配）
            image = next((img for img in client.images.list() if expected_tags.intersection(set(img.tags))), None)
    except Exception as e:
        safe_log(f"Error while building the Docker image: {e}")
        return None

    try:
        # detach=True：后台运行容器，不阻塞当前线程
        # WebIDE Docker bootstrap (2026-07-07): use host network because the
        # temporary nested daemon has no bridge network/DNS, and agents need LLM API access.
        volumes = {}
        host_site_packages = "/usr/local/lib/python3.12/dist-packages"
        if os.path.isdir(host_site_packages):
            # WebIDE Docker bootstrap (2026-07-08): the imported local Python
            # base image is intentionally slim; mount host-installed packages
            # read-only so DGM smoke runs do not rebuild or pip-install every time.
            volumes[host_site_packages] = {"bind": host_site_packages, "mode": "ro"}
        container = client.containers.run(
            image=image_name,
            command=["tail", "-f", "/dev/null"],
            name=container_name,
            detach=True,
            network_mode="host",
            volumes=volumes or None,
        )
        safe_log(f"Container '{container_name}' started successfully.")
        return container
    except Exception as e:
        safe_log(f"Error while starting the container: {e}")
        return None


def cleanup_container(container):
    """
    停止并删除指定的 Docker 容器（释放资源）。

    顺序：必须先 stop（发送 SIGTERM，等待进程退出）再 remove（删除容器记录），
    直接 remove 一个运行中的容器会报错。

    Args:
        container: docker.Container 实例（由 build_dgm_container 返回）。
    """
    safe_log(f"Stopping container '{container.name}'...")
    container.stop()
    container.remove()
    safe_log(f"Container '{container.name}' removed.")


def copy_to_container(container, source_path: Union[str, Path], dest_path: Union[str, Path]) -> None:
    """
    将本地文件或目录复制到 Docker 容器内指定路径。

    实现原理：用 create_archive() 将源文件/目录打包成 tar 流，
    再通过 container.put_archive() 传入容器并解压到目标目录。
    此方式无需在容器内安装任何额外工具，也不依赖网络挂载。

    注意：目标路径语义与 `docker cp` 一致——
      - 文件模式：dest_path 的父目录作为解压根，文件名由 archive 内的路径指定
      - 目录模式：dest_path 的父目录作为解压根，目录名由 source_path.name 决定

    Args:
        container: docker.Container 实例。
        source_path (Union[str, Path]): 本地源文件/目录路径。
        dest_path (Union[str, Path]): 容器内目标路径。

    Raises:
        FileNotFoundError: 源路径在本地不存在时抛出。
        Exception: tar 创建失败、put_archive 失败等情况时抛出。
    """
    source_path = Path(source_path)
    dest_path = Path(dest_path)

    try:
        if not source_path.exists():
            raise FileNotFoundError(f"Source path not found: {source_path}")

        if source_path.is_file():
            # 文件模式：将文件内容打包，目标目录为 dest_path 的父目录
            container_dest_dir = str(dest_path.parent)
            archive_path = dest_path.name  # tar 内文件名 = 目标文件名
            with open(source_path, 'rb') as source_file:
                data = source_file.read()
            archive = create_archive(archive_path, data)
        else:
            # 目录模式：递归打包整个目录，解压到 dest_path 的父目录
            container_dest_dir = str(dest_path.parent)
            archive = create_archive(source_path)

        # 确保容器内目标目录存在（mkdir -p 幂等，不存在才创建）
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

    实现原理：通过 container.get_archive() 取出 tar 字节流，
    在内存中用 tarfile 解压到本地路径，无需中间临时文件。

    文件类型判断：通过 `stat -f '%HT'` 检测路径类型（仅 macOS/BSD stat 格式，
    Linux 容器需注意兼容性，若容器是 Linux 可能需改用 `stat -c '%F'`）。

    Args:
        container: docker.Container 实例。
        source_path (Union[str, Path]): 容器内源文件/目录路径。
        dest_path (Union[str, Path]): 本地目标路径。

    Raises:
        FileNotFoundError: 容器内源路径不存在时抛出。
        Exception: get_archive 失败、tar 解压失败等情况时抛出。
    """
    source_path = Path(source_path)
    dest_path = Path(dest_path)

    try:
        # 检查容器内源路径是否存在（test -e 存在返回 0，不存在返回非 0）
        result = container.exec_run(f"test -e {source_path}")
        if result.exit_code and result.exit_code != 0:
            raise FileNotFoundError(f"Source path not found in container: {source_path}")

        # 判断是文件还是目录（stat -f '%HT' 返回 'Regular File' 或 'Directory'）
        result = container.exec_run(f"stat -f '%HT' {source_path}")
        is_file = result.output.decode().strip() == 'Regular File'

        # 确保本地目标目录存在
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        safe_log(f"Copying from container {source_path} to local path {dest_path}")

        # get_archive 返回 (bits_generator, stat_dict)，bits 是 tar 内容的分块迭代器
        bits, stat = container.get_archive(str(source_path))

        # 将所有分块合并为完整 bytes（内存中操作，适合中等大小的文件）
        archive_data = b''.join(bits)

        # 用 BytesIO 包装，给 tarfile 提供类文件接口
        stream = io.BytesIO(archive_data)

        with tarfile.open(fileobj=stream, mode='r') as tar:
            if is_file:
                # 单文件：取 tar 内第一个成员，直接写入目标路径
                member = tar.getmembers()[0]
                with tar.extractfile(member) as source_file:
                    data = source_file.read()
                    with open(dest_path, 'wb') as dest_file:
                        dest_file.write(data)
            else:
                # 目录：解压到目标目录的父目录，然后按需重命名
                tar.extractall(path=str(dest_path.parent))
                # stat['name'] 是 tar 归档时的根目录名，若与期望路径不同则重命名
                extracted_path = dest_path.parent / Path(stat['name']).name
                if extracted_path != dest_path and extracted_path.exists():
                    extracted_path.rename(dest_path)

        safe_log(f"Successfully copied from container to {dest_path}")

    except Exception as e:
        safe_log(f"Error copying from container: {e}", logging.ERROR)
        raise


def log_container_output(exec_result):
    """
    记录容器命令执行的输出，并在退出码非零时抛出异常。

    为什么需要区分流式和非流式：
      Docker SDK 的 exec_run() 默认返回非流式输出（output 为 bytes）；
      若以 stream=True 调用，则 output 是 generator，需逐块读取。
      两种模式下输出的类型不同，必须分别处理。

    Args:
        exec_result: docker.Container.exec_run() 的返回值，
                     包含 output（bytes 或 generator）和 exit_code（int）。

    Raises:
        Exception: 容器命令退出码非零时抛出，让调用方知道执行失败。
    """
    if isinstance(exec_result.output, bytes):
        # 非流式：整体解码后一次性记录
        safe_log(f"Container output: {exec_result.output.decode()}")
    else:
        # 流式：逐块解码记录（每块 strip 去除多余换行）
        for chunk in exec_result.output:
            if chunk:
                safe_log(f"Container output: {chunk.decode().strip()}")

    # 退出码非零表示容器内命令执行失败，向上抛异常让调用方处理
    if exec_result.exit_code and exec_result.exit_code != 0:
        error_msg = f"Script failed with exit code {exec_result.exit_code}"
        safe_log(error_msg, logging.ERROR)
        raise Exception(error_msg)
