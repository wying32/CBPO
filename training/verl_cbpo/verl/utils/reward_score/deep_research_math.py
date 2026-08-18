"""Rule-based reward for CBPO math-only training."""

from typing import Any, Dict, Optional

from verl.utils.reward_score.deep_research import (
    extract_answer,
    get_f1_score,
    last_boxed_only_string,
    remove_boxed,
    validate_format,
)
from verl.utils.reward_score.prime_math import grade_answer


def _is_equivalent(answer: str, ground_truth: Any) -> bool:
    references = ground_truth if isinstance(ground_truth, list) else [ground_truth]
    return any(grade_answer(answer, str(reference)) for reference in references)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return binary correctness using symbolic mathematical equivalence."""

    result = {
        "score": 0.0,
        "acc": 0.0,
        "reason": "",
        "answer": "",
        "f1_score": 0.0,
    }

    valid_template, reason = validate_format(solution_str)
    if not valid_template:
        result["score"] = -1.0
        result["reason"] = f"bad format: {reason}"
        return result

    answer_part = extract_answer(solution_str)
    if answer_part is None:
        result["score"] = -1.0
        result["reason"] = "cannot extract answer"
        return result

    try:
        boxed_answer = last_boxed_only_string(answer_part)
        if boxed_answer is None:
            raise ValueError("boxed answer not found")
        answer = remove_boxed(boxed_answer)
    except (AssertionError, TypeError, ValueError) as error:
        result["score"] = -1.0
        result["reason"] = f"find box error: {error}"
        return result

    result["answer"] = answer
    result["f1_score"] = get_f1_score(answer, ground_truth)

    try:
        correct = _is_equivalent(answer, ground_truth)
    except Exception as error:
        result["reason"] = f"math verifier error: {error}"
        return result

    if correct:
        result["score"] = 1.0
        result["acc"] = 1.0
        result["reason"] = "mathematically equivalent answer"
    else:
        result["reason"] = "wrong answer with valid format"
    return result
