import re
from enum import Enum


# ============================================================
# swe_log_parsers.py
# SWE-bench 测试日志解析器集合。
#
# 核心问题：SWE-bench 涵盖 18+ 个开源 Python 项目，各项目使用不同测试框架
#（pytest、Django unittest、sympy 自定义框架、seaborn 等），
# 输出格式各异，必须为每种框架实现专用的解析函数。
#
# 统一输出格式：所有解析器返回 dict[str, str]，即 {测试用例名: 测试状态}，
# 测试状态为 TestStatus enum 的字符串值（PASSED/FAILED/ERROR/SKIPPED/XFAIL）。
#
# MAP_REPO_TO_PARSER：将 "owner/repo" 格式的仓库名映射到对应的解析函数，
# eval_utils.py 的 parse_eval_output() 用此映射自动选择解析器。
# ============================================================


class TestStatus(Enum):
    """
    测试用例可能的状态值枚举。

    - FAILED：测试断言失败（代码行为与预期不符）
    - PASSED：测试通过
    - SKIPPED：测试被跳过（通常因为缺少依赖或条件不满足）
    - ERROR：测试本身发生异常（非断言失败，而是执行时崩溃）
    - XFAIL：预期失败（标记为 @pytest.mark.xfail 且实际失败，视为"正常"）
    """
    FAILED = "FAILED"
    PASSED = "PASSED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"
    XFAIL = "XFAIL"


def parse_log_pytest(log: str) -> dict[str, str]:
    """
    解析标准 pytest 输出日志（短摘要格式）。

    pytest 在 -v 模式下，每个测试用例结束后会在单独一行输出：
      PASSED tests/test_foo.py::test_bar
      FAILED tests/test_foo.py::test_baz - AssertionError: ...
      ERROR  tests/test_foo.py::test_qux
      SKIPPED tests/test_foo.py::test_skip

    解析策略：逐行扫描，检测行首是否以某个 TestStatus 值开头，
    提取第二个空格分隔字段作为测试用例名。

    FAILED 行的特殊处理：pytest 在 FAILED 后可能附加 " - <错误摘要>"，
    如 "FAILED tests/foo.py::bar - AssertionError"。
    用 replace(" - ", " ") 把 " - " 替换成空格，
    使 split() 后第二个 token 仍为纯测试路径，不含错误摘要。

    Args:
        log (str): pytest 运行的完整日志字符串。

    Returns:
        dict[str, str]: {测试名: 状态字符串} 映射。
    """
    test_status_map = {}
    for line in log.split("\n"):
        if any([line.startswith(x.value) for x in TestStatus]):
            # 移除 FAILED 行中的 " - <摘要>" 部分，避免路径解析错误
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")
            test_case = line.split()
            if len(test_case) <= 1:
                continue
            # test_case[0] = 状态（如 "PASSED"），test_case[1] = 测试名
            test_status_map[test_case[1]] = test_case[0]
    return test_status_map


def parse_log_pytest_options(log: str) -> dict[str, str]:
    """
    解析带参数化选项的 pytest 日志（适用于 pydicom、requests、pylint 等）。

    参数化测试（@pytest.mark.parametrize）的测试名格式：
      tests/test_foo.py::test_bar[option_value]
    其中 option_value 可能是文件路径（如 /tmp/test_data.dcm）。

    特殊处理：若 option 是绝对路径（以 '/' 开头但不以 '//' 开头，且不含 '*'），
    则只保留路径的最后一个分段（basename），避免因路径中含有 '/' 被 split 截断。

    例：tests/foo.py::test_bar[/very/long/path/to/file.dcm]
      → tests/foo.py::test_bar[/file.dcm]（只保留文件名部分，加上 '/' 前缀）

    Args:
        log (str): pytest 运行的完整日志字符串。

    Returns:
        dict[str, str]: {测试名（含选项）: 状态字符串} 映射。
    """
    # 匹配 "base_test_name[option_content]" 格式
    option_pattern = re.compile(r"(.*?)\[(.*)\]")
    test_status_map = {}
    for line in log.split("\n"):
        if any([line.startswith(x.value) for x in TestStatus]):
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")
            test_case = line.split()
            if len(test_case) <= 1:
                continue
            has_option = option_pattern.search(test_case[1])
            if has_option:
                main, option = has_option.groups()
                # 路径选项归一化：'/a/b/c.dcm' → '/c.dcm'（只保留文件名）
                if option.startswith("/") and not option.startswith("//") and "*" not in option:
                    option = "/" + option.split("/")[-1]
                test_name = f"{main}[{option}]"
            else:
                test_name = test_case[1]
            test_status_map[test_name] = test_case[0]
    return test_status_map


def parse_log_django(log: str) -> dict[str, str]:
    """
    解析 Django 的 unittest 风格测试日志（verbose 模式）。

    Django 测试输出格式与 pytest 截然不同：
      test_method_name (tests.module.TestClass) ... ok
      test_method_name (tests.module.TestClass) ... FAIL
      test_method_name (tests.module.TestClass) ... ERROR
      test_method_name (tests.module.TestClass) ... skipped 'reason'

    其中 "..." 分隔测试名和结果。此函数还处理了若干 Django 特有的边缘情况：
      1. 单行特殊测试（"--version is equivalent to version"）
      2. 多行输出：某些测试在 "..." 和 "ok" 之间插入了额外内容（如 "System check..."）
      3. prev_test 机制：记录上一个遇到 " ... " 的测试，用于处理 "ok" 出现在下一行的情况

    Args:
        log (str): Django 测试运行的完整日志字符串。

    Returns:
        dict[str, str]: {测试名: 状态字符串} 映射。
    """
    test_status_map = {}
    lines = log.split("\n")

    prev_test = None  # 记录上一个出现 " ... " 的测试名，用于处理跨行 "ok"
    for line in lines:
        line = line.strip()

        # 特殊单行测试（--version 相关，固定格式，直接硬编码处理）
        if "--version is equivalent to version" in line:
            test_status_map["--version is equivalent to version"] = TestStatus.PASSED.value

        # 记录遇到 " ... " 的测试（结果可能在下一行出现）
        if " ... " in line:
            prev_test = line.split(" ... ")[0]

        # 检测各种通过标志（Django 有多种变体）
        pass_suffixes = (" ... ok", " ... OK", " ...  OK")
        for suffix in pass_suffixes:
            if line.endswith(suffix):
                # 特殊 case：某些行把两段内容拼在一行（django__django-7188 已知 bug）
                # 格式："Applying sites.0002...test_no_migrations ... ok"，需要截取后半段
                if line.strip().startswith("Applying sites.0002_alter_domain_unique...test_no_migrations"):
                    line = line.split("...", 1)[-1].strip()
                test = line.rsplit(suffix, 1)[0]
                test_status_map[test] = TestStatus.PASSED.value
                break
        if " ... skipped" in line:
            test = line.split(" ... skipped")[0]
            test_status_map[test] = TestStatus.SKIPPED.value
        if line.endswith(" ... FAIL"):
            test = line.split(" ... FAIL")[0]
            test_status_map[test] = TestStatus.FAILED.value
        if line.startswith("FAIL:"):
            # 格式："FAIL: test_method (module.Class)" —— 行首是 FAIL
            test = line.split()[1].strip()
            test_status_map[test] = TestStatus.FAILED.value
        if line.endswith(" ... ERROR"):
            test = line.split(" ... ERROR")[0]
            test_status_map[test] = TestStatus.ERROR.value
        if line.startswith("ERROR:"):
            test = line.split()[1].strip()
            test_status_map[test] = TestStatus.ERROR.value

        if line.lstrip().startswith("ok") and prev_test is not None:
            # "ok" 出现在独立一行，说明上一个 prev_test 通过了
            test = prev_test
            test_status_map[test] = TestStatus.PASSED.value

    # 处理 Django logger 的已知 bug：某些测试的 "ok" 前被插入了长段多行输出
    # 用正则从完整 log 字符串中匹配这三种跨行模式
    patterns = [
        r"^(.*?)\s\.\.\.\sTesting\ against\ Django\ installed\ in\ ((?s:.*?))\ silenced\)\.\nok$",
        r"^(.*?)\s\.\.\.\sInternal\ Server\ Error:\ \/(.*)\/\nok$",
        r"^(.*?)\s\.\.\.\sSystem check identified no issues \(0 silenced\)\nok$"
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, log, re.MULTILINE):
            test_name = match.group(1)
            test_status_map[test_name] = TestStatus.PASSED.value
    return test_status_map


def parse_log_pytest_v2(log: str) -> dict[str, str]:
    """
    解析较新版本 pytest 的日志（带 ANSI 颜色转义码）。

    与 parse_log_pytest 的区别：
      新版 pytest 在终端输出中嵌入了 ANSI 转义序列（如 \x1b[32m 表示绿色），
      直接按行首匹配会因为转义码干扰而失败。
      此函数先用正则去除 "[\d+m" 格式的颜色码，再清除其余控制字符，
      然后再做与 parse_log_pytest 相同的解析逻辑。

    兼容旧格式：额外支持状态出现在行尾（"test_name STATUS" 格式），
    兼容某些旧版 pytest 插件的输出。

    Args:
        log (str): 带 ANSI 转义码的 pytest 日志字符串。

    Returns:
        dict[str, str]: {测试名: 状态字符串} 映射。
    """
    test_status_map = {}
    # 控制字符集合（chr(1)~chr(31)），用于清除所有非打印字符
    escapes = "".join([chr(char) for char in range(1, 32)])
    for line in log.split("\n"):
        # 去除 ANSI 颜色码（如 \x1b[32m、\x1b[0m）
        line = re.sub(r"\[(\d+)m", "", line)
        # 去除剩余控制字符（\t、\r 等）
        translator = str.maketrans("", "", escapes)
        line = line.translate(translator)
        if any([line.startswith(x.value) for x in TestStatus]):
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")
            test_case = line.split()
            test_status_map[test_case[1]] = test_case[0]
        # 兼容旧格式：状态在行尾（如 "test_name PASSED"）
        elif any([line.endswith(x.value) for x in TestStatus]):
            test_case = line.split()
            test_status_map[test_case[0]] = test_case[1]
    return test_status_map


def parse_log_seaborn(log: str) -> dict[str, str]:
    """
    解析 seaborn 项目的测试日志（混合格式）。

    seaborn 的测试输出混合了多种格式：
      - 失败：行首 "FAILED test_name"
      - 通过（格式一）：" test_name PASSED " —— 状态夹在中间
      - 通过（格式二）：行首 "PASSED test_name"

    Args:
        log (str): seaborn 测试日志字符串。

    Returns:
        dict[str, str]: {测试名: 状态字符串} 映射。
    """
    test_status_map = {}
    for line in log.split("\n"):
        if line.startswith(TestStatus.FAILED.value):
            # 格式：FAILED tests/test_foo.py::test_bar
            test_case = line.split()[1]
            test_status_map[test_case] = TestStatus.FAILED.value
        elif f" {TestStatus.PASSED.value} " in line:
            # 格式：... test_name PASSED ...（状态在中间位置）
            parts = line.split()
            if parts[1] == TestStatus.PASSED.value:
                test_case = parts[0]
                test_status_map[test_case] = TestStatus.PASSED.value
        elif line.startswith(TestStatus.PASSED.value):
            # 格式：PASSED tests/test_foo.py::test_bar
            parts = line.split()
            test_case = parts[1]
            test_status_map[test_case] = TestStatus.PASSED.value
    return test_status_map


def parse_log_sympy(log: str) -> dict[str, str]:
    """
    解析 sympy 项目的自定义测试框架日志。

    sympy 不用 pytest，而是自己的 runtests 框架，输出格式独特：
      - 失败摘要：以 "___" 包围的行，格式 "_ path/to/file.py:test_name _"
      - 逐测试状态行：以 "test_" 开头，结尾可能是 " ok"、" F"、" E"，
        或带 "[FAIL]" / "[OK]" 后缀

    两轮扫描策略：
      1. 先用正则从 "___" 分隔行中批量提取失败的测试（这些是总结段）
      2. 再逐行扫描以 "test_" 开头的详细状态行，覆盖或补充结果

    Args:
        log (str): sympy runtests 的完整日志字符串。

    Returns:
        dict[str, str]: {测试名: 状态字符串} 映射。
    """
    test_status_map = {}
    # 匹配失败摘要行：形如 "__ path/to/file.py:test_name __"
    pattern = r"(_*) (.*)\.py:(.*) (_*)"
    matches = re.findall(pattern, log)
    for match in matches:
        test_case = f"{match[1]}.py:{match[2]}"
        test_status_map[test_case] = TestStatus.FAILED.value

    # 逐行扫描详细状态（覆盖上面批量提取的结果，以最新状态为准）
    for line in log.split("\n"):
        line = line.strip()
        if line.startswith("test_"):
            # 去掉行尾的 "[FAIL]" 或 "[OK]" 标记（只留测试名部分）
            if line.endswith("[FAIL]") or line.endswith("[OK]"):
                line = line[: line.rfind("[")]
                line = line.strip()
            if line.endswith(" E"):
                test = line.split()[0]
                test_status_map[test] = TestStatus.ERROR.value
            if line.endswith(" F"):
                test = line.split()[0]
                test_status_map[test] = TestStatus.FAILED.value
            if line.endswith(" ok"):
                test = line.split()[0]
                test_status_map[test] = TestStatus.PASSED.value
    return test_status_map


def parse_log_matplotlib(log: str) -> dict[str, str]:
    """
    解析 matplotlib 项目的 pytest 日志（预处理鼠标按钮枚举值）。

    matplotlib 的测试名中可能含有 "MouseButton.LEFT" 或 "MouseButton.RIGHT"
    这样的 Python 枚举值（作为参数化测试的选项），这些值在行分割时会引起歧义。
    将其替换为对应的整数值（LEFT=1，RIGHT=3）后，再用标准 pytest 解析逻辑处理。

    Args:
        log (str): matplotlib pytest 日志字符串。

    Returns:
        dict[str, str]: {测试名: 状态字符串} 映射。
    """
    test_status_map = {}
    for line in log.split("\n"):
        # 替换枚举值为整数，避免字符串中的点号干扰后续解析
        line = line.replace("MouseButton.LEFT", "1")
        line = line.replace("MouseButton.RIGHT", "3")
        if any([line.startswith(x.value) for x in TestStatus]):
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")
            test_case = line.split()
            if len(test_case) <= 1:
                continue
            test_status_map[test_case[1]] = test_case[0]
    return test_status_map


# ============================================================
# 以下是解析器别名定义：使用标准 pytest 格式的仓库直接复用对应解析器，
# 避免重复实现相同的解析逻辑。
# ============================================================

# 使用标准 pytest 输出格式的项目（直接复用 parse_log_pytest）
parse_log_astroid = parse_log_pytest
parse_log_flask = parse_log_pytest
parse_log_marshmallow = parse_log_pytest
parse_log_pvlib = parse_log_pytest
parse_log_pyvista = parse_log_pytest
parse_log_sqlfluff = parse_log_pytest
parse_log_xarray = parse_log_pytest

# 使用带路径选项的参数化 pytest 格式（复用 parse_log_pytest_options）
parse_log_pydicom = parse_log_pytest_options
parse_log_requests = parse_log_pytest_options
parse_log_pylint = parse_log_pytest_options

# 使用带 ANSI 转义码的新版 pytest 格式（复用 parse_log_pytest_v2）
parse_log_astropy = parse_log_pytest_v2
parse_log_scikit = parse_log_pytest_v2
parse_log_sphinx = parse_log_pytest_v2


# ============================================================
# MAP_REPO_TO_PARSER
# 仓库名（"owner/repo" 格式）到解析函数的映射表。
# eval_utils.py 的 parse_eval_output() 通过 instance_id 解析出仓库名后，
# 在此表中查找对应的解析函数。
#
# 注意："dgm" 条目是 DGM 框架自测时使用的特殊条目（不是真实 GitHub 仓库），
# 使用标准 pytest 解析器即可（DGM 的测试用 pytest 运行）。
# ============================================================
MAP_REPO_TO_PARSER = {
    "astropy/astropy": parse_log_astropy,
    "django/django": parse_log_django,
    "marshmallow-code/marshmallow": parse_log_marshmallow,
    "matplotlib/matplotlib": parse_log_matplotlib,
    "mwaskom/seaborn": parse_log_seaborn,
    "pallets/flask": parse_log_flask,
    "psf/requests": parse_log_requests,
    "pvlib/pvlib-python": parse_log_pvlib,
    "pydata/xarray": parse_log_xarray,
    "pydicom/pydicom": parse_log_pydicom,
    "pylint-dev/astroid": parse_log_astroid,
    "pylint-dev/pylint": parse_log_pylint,
    "pytest-dev/pytest": parse_log_pytest,
    "pyvista/pyvista": parse_log_pyvista,
    "scikit-learn/scikit-learn": parse_log_scikit,
    "sqlfluff/sqlfluff": parse_log_sqlfluff,
    "sphinx-doc/sphinx": parse_log_sphinx,
    "sympy/sympy": parse_log_sympy,
    # DGM 框架自测条目（非真实 GitHub 仓库，使用标准 pytest 解析器）
    "dgm": parse_log_pytest,
}
