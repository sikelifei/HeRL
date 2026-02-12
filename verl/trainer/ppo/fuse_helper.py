# from __future__ import annotations

# from typing import Any, List, Optional, Sequence, Tuple


# def normalize_is_following_list(raw: Any) -> Optional[List[int]]:
#     if isinstance(raw, list):
#         return [int(bool(x)) for x in raw]
#     return None


# def get_numeric_score(score: Any, fallback: float = 0.0) -> float:
#     if isinstance(score, dict):
#         dense = score.get("dense")
#         if isinstance(dense, (int, float)):
#             return float(dense)
#         flat = score.get("score")
#         if isinstance(flat, bool):
#             return 1.0 if flat else 0.0
#         if isinstance(flat, (int, float)):
#             return float(flat)
#         sparse = score.get("sparse")
#         if isinstance(sparse, bool):
#             return 1.0 if sparse else 0.0
#         if isinstance(sparse, (int, float)):
#             return float(sparse)
#     elif isinstance(score, bool):
#         return 1.0 if score else 0.0
#     elif isinstance(score, (int, float)):
#         return float(score)
#     return float(fallback)


# def find_all_better_pairs(is_lists: Sequence[List[int]]) -> Tuple[int, List[dict]]:
#     if not is_lists:
#         return 0, []
#     single_best = max(sum(lst) for lst in is_lists)
#     better_pairs: List[dict] = []
#     for i in range(len(is_lists)):
#         for j in range(i + 1, len(is_lists)):
#             union = [max(a, b) for a, b in zip(is_lists[i], is_lists[j])]
#             union_ones = sum(union)
#             if union_ones > single_best:
#                 better_pairs.append({"i": i, "j": j, "union": union, "union_ones": union_ones})
#     return single_best, better_pairs


# def _extract_checkers_and_rewards(
#     constraint_results1: list[dict],
#     constraint_results2: list[dict],
# ) -> Tuple[List[str], List[int], List[int]]:
#     n = min(len(constraint_results1), len(constraint_results2))
#     constraint_results1 = constraint_results1[:n]
#     constraint_results2 = constraint_results2[:n]

#     checkers, reward1, reward2 = [], [], []
#     for c1, c2 in zip(constraint_results1, constraint_results2):
#         desc = c1.get("description") or c1.get("instruction_id") or "未命名约束"
#         if isinstance(desc, list):
#             desc = " / ".join(map(str, desc))
#         checkers.append(str(desc))
#         reward1.append(1 if c1.get("is_following") else 0)
#         reward2.append(1 if c2.get("is_following") else 0)
#     return checkers, reward1, reward2


# def build_fuse_prompt(
#     instruction: str,
#     response1: str,
#     score1: dict,
#     response2: str,
#     score2: dict,
# ) -> str:
#     constraint_results1 = score1.get("constraint_results") or []
#     constraint_results2 = score2.get("constraint_results") or []
#     checkers, reward1, reward2 = _extract_checkers_and_rewards(constraint_results1, constraint_results2)

#     lines: List[str] = []
#     for idx, (checker, r1, r2) in enumerate(zip(checkers, reward1, reward2), start=1):
#         union = 1 if (r1 or r2) else 0
#         goal = "必须满足" if union else "尽量满足"
#         status1 = "满足" if r1 else "不满足"
#         status2 = "满足" if r2 else "不满足"
#         lines.append(
#             f"[{idx}] 约束：{checker}\n"
#             f"    - 回复1：{status1} (标记: {r1})\n"
#             f"    - 回复2：{status2} (标记: {r2})\n"
#             f"    - 目标：{goal}（目标标记: {union}）"
#         )
#     constraints_block = "\n".join(lines) or "（当前样本无明确约束，仍需兼顾两条回复的优势。）"

#     return f"""
# 你是一名“约束优化整合助手”。现在有一个指令（instruction），以及两条由模型生成的回复（response1 和 response2）。每条回复都根据一组约束被评估为“满足(1)/不满足(0)”。
# 你的目标是：
# 1. 针对每条约束，目标标记 = max(reward1, reward2)：若为 1（必须满足）则必须满足，若为 0（尽量满足）则在不牺牲其它约束前提下尽量满足；
# 2. 生成一个新的融合回复（response_fused），使其在所有约束上的表现不低于两条回复中较好的那一条；
# 3. 只有回复1满足时优先复用/改写回复1，只有回复2满足时优先复用/改写回复2，两者都满足可任选更清晰的表述，两者都不满足则尝试补充或重写；
# 4. 满足约束的前提下尽量保留已有信息，避免编造；
# 5. **只输出融合后的正文，不要任何解释**。

# ========================
# [指令]
# {instruction}

# [回复1]
# {response1}

# [回复2]
# {response2}

# [约束对比与目标]
# {constraints_block}

# 请严格根据上述要求生成最终的融合回复（response_fused），并且只输出融合内容本身。
# ========================
# """.strip()


# __all__ = [
#     "build_fuse_prompt",
#     "find_all_better_pairs",
#     "get_numeric_score",
#     "normalize_is_following_list",
# ]


from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple


def normalize_is_following_list(raw: Any) -> Optional[List[int]]:
    if isinstance(raw, list):
        return [int(bool(x)) for x in raw]
    return None


def get_numeric_score(score: Any, fallback: float = 0.0) -> float:
    if isinstance(score, dict):
        dense = score.get("dense")
        if isinstance(dense, (int, float)):
            return float(dense)
        flat = score.get("score")
        if isinstance(flat, bool):
            return 1.0 if flat else 0.0
        if isinstance(flat, (int, float)):
            return float(flat)
        sparse = score.get("sparse")
        if isinstance(sparse, bool):
            return 1.0 if sparse else 0.0
        if isinstance(sparse, (int, float)):
            return float(sparse)
    elif isinstance(score, bool):
        return 1.0 if score else 0.0
    elif isinstance(score, (int, float)):
        return float(score)
    return float(fallback)


def find_all_better_pairs(is_lists: Sequence[List[int]]) -> Tuple[int, List[dict]]:
    if not is_lists:
        return 0, []
    single_best = max(sum(lst) for lst in is_lists)
    better_pairs: List[dict] = []
    for i in range(len(is_lists)):
        for j in range(i + 1, len(is_lists)):
            union = [max(a, b) for a, b in zip(is_lists[i], is_lists[j])]
            union_ones = sum(union)
            if union_ones > single_best:
                better_pairs.append(
                    {"i": i, "j": j, "union": union, "union_ones": union_ones}
                )
    return single_best, better_pairs


def _extract_checkers_and_rewards(
    constraint_results1: list[dict],
    constraint_results2: list[dict],
) -> Tuple[List[str], List[int], List[int]]:
    n = min(len(constraint_results1), len(constraint_results2))
    constraint_results1 = constraint_results1[:n]
    constraint_results2 = constraint_results2[:n]

    checkers: List[str] = []
    reward1: List[int] = []
    reward2: List[int] = []

    for c1, c2 in zip(constraint_results1, constraint_results2):
        desc = c1.get("description") or c1.get("instruction_id") or "Unnamed constraint"
        if isinstance(desc, list):
            desc = " / ".join(map(str, desc))
        checkers.append(str(desc))
        reward1.append(1 if c1.get("is_following") else 0)
        reward2.append(1 if c2.get("is_following") else 0)
    return checkers, reward1, reward2


def build_fuse_prompt(
    instruction: str,
    response1: str,
    score1: dict,
    response2: str,
    score2: dict,
) -> str:
    """
    Build a single English prompt string for a constraint-aware fusion assistant.
    This is intended to be used as the `user` content if you also provide a
    separate English `system` prompt, or standalone if you do not use system messages.
    """
    constraint_results1 = score1.get("constraint_results") or []
    constraint_results2 = score2.get("constraint_results") or []
    checkers, reward1, reward2 = _extract_checkers_and_rewards(
        constraint_results1, constraint_results2
    )

    lines: List[str] = []
    for idx, (checker, r1, r2) in enumerate(zip(checkers, reward1, reward2), start=1):
        union = 1 if (r1 or r2) else 0
        goal = "MUST satisfy" if union else "NICE TO HAVE"
        status1 = "satisfies" if r1 else "does NOT satisfy"
        status2 = "satisfies" if r2 else "does NOT satisfy"
        lines.append(
            f"[{idx}] Constraint: {checker}\n"
            f"    - Response 1: {status1} (label: {r1})\n"
            f"    - Response 2: {status2} (label: {r2})\n"
            f"    - Target: {goal} (target label: {union})"
        )

    constraints_block = (
        "\n".join(lines)
        or "(No explicit constraints for this sample. Still try to combine the strengths of both responses.)"
    )

    return f"""
You are a "constraint-aware fusion assistant".

There is:
- one instruction (instruction),
- two candidate responses (response1 and response2),
- and for a list of constraints, each response has been evaluated as "satisfies (1) / does not satisfy (0)".

Your goals:
1. For each constraint, the target label is defined as max(reward1, reward2):
   - If the target label is 1 ("MUST satisfy"), the final fused answer must satisfy this constraint.
   - If the target label is 0 ("NICE TO HAVE"), satisfy it when possible, as long as you do not violate any "MUST satisfy" constraints.
2. Produce a new fused answer (response_fused) whose performance on all constraints is:
   - at least as good as the better of the two original responses; and
   - as close as possible to satisfying all constraints whose target label is 1.
3. For each constraint:
   - If only response1 satisfies it, prefer to reuse or adapt the relevant content from response1.
   - If only response2 satisfies it, prefer to reuse or adapt the relevant content from response2.
   - If both satisfy it, choose whichever phrasing is clearer or more concise.
   - If neither satisfies it, try to rewrite or extend the content so that the fused answer satisfies the constraint (especially when the target label is 1).
4. While satisfying the constraints, preserve as much useful information from the original responses as possible, and avoid inventing unsupported facts.
5. Output rules:
   - You must output ONLY the fused answer text itself.
   - Do NOT output any explanations, reasoning steps, constraint lists, or meta text.
   - Do NOT include labels or prefixes such as "Answer:", "Final answer:", "Fusion result:", or "response_fused:".
   - Do NOT output any Markdown code fences (such as ``` or ```json).
   - The first line of your output must start directly with the fused answer content.

========================
[Instruction]
{instruction}

[Candidate response 1]
{response1}

[Candidate response 2]
{response2}

[Constraint comparison and targets]
{constraints_block}

Please strictly follow the above requirements and generate the final fused answer (response_fused),
and output only the fused answer content itself.
========================
""".strip()


__all__ = [
    "build_fuse_prompt",
    "find_all_better_pairs",
    "get_numeric_score",
    "normalize_is_following_list",
]
