"""
judge/prompts.py
----------------
DGM 自改进的核心目标文件：存放评判 research proposal 合理性的 prompt 模板。

在整个适配架构中的定位：
  - 这是 DGM agent 每一轮 self-improve 最主要的修改对象
  - agent 通过修改下面的 SYSTEM_PROMPT_* / USER_TEMPLATE_* 字符串常量来优化判断策略
  - 两种 mode 给 DGM 提供了"均衡评审 vs 激进过滤"这一实验维度

与 SoundnessBench rigorbench/evaluation/prompt.py 的关系：
  - 直接提取了两套 prompt 常量（direct_bucket / direct_bucket_aggressive）
  - 去掉了 markdown 文件加载逻辑（load_prompt_from_file）：
    原版支持从外部 .md 文件热加载 prompt，但 DGM agent 直接编辑 Python 源码，
    不需要这层间接；去掉后逻辑更简单，也更容易被 agent 理解和修改。
"""

from __future__ import annotations
from typing import Literal

EvaluationMode = Literal["direct_bucket", "direct_bucket_aggressive"]

# ── direct_bucket（标准模式）──────────────────────────────────────────────────
# 定位：公平、均衡的同行评审人，不对 low/high 有先验偏好
# DGM 改进方向示例：调整 low/high 的判断标准描述，或改变 confidence 的语义
SYSTEM_PROMPT_DIRECT_BUCKET = """You are an expert ML researcher and peer reviewer. Classify the scientific rigor bucket of a research idea and your assessment confidence from 1 to 5 from its hypothesis and experiment description.

Output the assessment as a JSON object, including a detailed step-by-step justification for the rigor bucket selected."""

USER_TEMPLATE_DIRECT_BUCKET = """Classify this hypothesis-experiment pair into one rigor bucket:
- "low": "Weak scientific contribution. Hypothesis is vague or trivial, experiments lack controls or baselines, metrics are weak, or methodology has fundamental flaws.",
- "high": "Strong scientific contribution. Hypothesis is clear and meaningful. Experiments are rigorous, controlled, include appropriate baselines/ablations, and use suitable metrics.",

Confidence Score Scale:
- 1: You are unable to assess this paper and have alerted the ACs to seek an opinion from different reviewers.
- 2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
- 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
- 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
- 5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

HYPOTHESIS:
{hypothesis}

EXPERIMENT:
{experiment}

Output format:
{{
  "justification": "<Think step-by-step, provide detailed justification>",
  "rigor_bucket": <"low" or "high">,
  "confidence": <1-5 integer>
}}

Constraints:
- rigor_bucket must be a choice in ["low", "high"]
- confidence must be an integer in [1, 5]
"""

# ── direct_bucket_aggressive（激进模式）──────────────────────────────────────
# 定位：严格 area chair，默认倾向于 low，只有极强的证据才给 high
# 用于过滤明显 low-rigor 的 proposal；kappa 通常比 direct_bucket 更高但 recall 更低
SYSTEM_PROMPT_DIRECT_BUCKET_AGGRESSIVE = """You are a strict ML area chair applying an aggressive rigor filter. Classify scientific rigor from a hypothesis and experiment description.

Default to "low" unless the evidence clearly demonstrates strong scientific rigor with concrete controls, strong baselines, appropriate metrics, and a credible evaluation plan.

Output valid JSON only with a detailed step-by-step justification."""

USER_TEMPLATE_DIRECT_BUCKET_AGGRESSIVE = """Classify this hypothesis-experiment pair into one rigor bucket under an aggressive standard:
- "low": choose this unless there is clear, concrete, and compelling evidence of rigorous methodology.
- "high": only if the plan is explicitly strong on hypothesis clarity, experimental controls, baselines/ablations, metric validity, and methodological credibility.

Aggressive policy:
- Penalize missing controls, vague methods, missing or weak baselines, underspecified metrics, unclear evaluation protocol, or hand-wavy claims.
- If information is incomplete or ambiguous, prefer "low".
- Use "high" only when justification is unambiguous.

Confidence Score Scale:
- 1: You are unable to assess this paper and have alerted the ACs to seek an opinion from different reviewers.
- 2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math or other details were not carefully checked.
- 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math or other details were not carefully checked.
- 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
- 5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math or other details carefully.

HYPOTHESIS:
{hypothesis}

EXPERIMENT:
{experiment}

Output format:
{{
  "justification": "<Think step-by-step, provide detailed justification>",
  "rigor_bucket": <"low" or "high">,
  "confidence": <1-5 integer>
}}

Constraints:
- rigor_bucket must be a choice in ["low", "high"]
- confidence must be an integer in [1, 5]
"""


def build_scoring_prompt(
    hypothesis: str,
    experiment: str,
    mode: EvaluationMode = "direct_bucket",
) -> tuple[str, str]:
    """返回 (system_prompt, user_prompt) 用于单个样本的评分。

    与原版 rigorbench 的差异：去掉了 prompt_path 参数。
    原版支持从外部 markdown 文件加载 prompt（热替换），此处省略，
    因为 DGM agent 直接修改本文件的字符串常量，不需要外部文件这层间接。
    """
    if mode == "direct_bucket_aggressive":
        system = SYSTEM_PROMPT_DIRECT_BUCKET_AGGRESSIVE
        template = USER_TEMPLATE_DIRECT_BUCKET_AGGRESSIVE
    else:
        # 未识别的 mode 也回退到 direct_bucket，保持鲁棒性
        system = SYSTEM_PROMPT_DIRECT_BUCKET
        template = USER_TEMPLATE_DIRECT_BUCKET
    return system, template.format(
        hypothesis=hypothesis.strip() or "(none)",
        experiment=experiment.strip() or "(none)",
    )
