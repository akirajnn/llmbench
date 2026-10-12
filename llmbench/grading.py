from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


@dataclass
class GradeResult:
    passed: bool
    detail: str
    # IFEval 風格的「寬鬆分數」(0~1)：預設等同 passed 的 0/1，
    # 只有 `checks`（複合條件）這種評分器才會給出介於中間的部分分數。
    score: float = field(default=None)  # type: ignore[assignment]

    def __post_init__(self):
        if self.score is None:
            self.score = 1.0 if self.passed else 0.0


def _normalize(text: str) -> str:
    """NFKC 正規化：把上標/下標數字、全形字元等轉成一般等價字元再比對。"""
    return unicodedata.normalize("NFKC", text)


def _extract_number(text: str) -> float | None:
    """取回應中「最後一個」數字當最終答案，而非第一個。

    模型即使被要求「只回答數字」，仍常附上運算過程（如 "12 × 7 = 84"），
    此時第一個數字通常是算式中的運算元，最後一個才是真正的答案。

    注意：這只是 fallback 的 heuristic。更嚴謹的做法是用 `gsm8k` 評分器，
    要求模型用 "#### <數字>" 明確標示最終答案（GSM8K 資料集的官方慣例），
    不需要靠猜的。
    """
    matches = re.findall(r"-?\d+(?:\.\d+)?", _normalize(text).replace(",", ""))
    if not matches:
        return None
    return float(matches[-1])


def grade_exact_match(response: str, expected: str, case_sensitive: bool = False) -> GradeResult:
    actual = _normalize(response).strip()
    target = _normalize(str(expected)).strip()
    if not case_sensitive:
        actual, target = actual.lower(), target.lower()
    passed = actual == target
    return GradeResult(passed, f"expected={target!r} actual={actual!r}")


def grade_contains(response: str, expected: str, case_sensitive: bool = False) -> GradeResult:
    haystack = _normalize(response)
    needle = _normalize(str(expected))
    if not case_sensitive:
        haystack, needle = haystack.lower(), needle.lower()
    passed = needle in haystack
    return GradeResult(passed, f"expected contains {needle!r} in response")


def grade_regex(response: str, expected: str) -> GradeResult:
    passed = re.search(expected, _normalize(response)) is not None
    return GradeResult(passed, f"pattern={expected!r}")


def grade_numeric_tolerance(response: str, expected: float, tolerance: float = 0) -> GradeResult:
    value = _extract_number(response)
    if value is None:
        return GradeResult(False, f"no number found in response: {response!r}")
    passed = abs(value - float(expected)) <= tolerance
    return GradeResult(passed, f"expected={expected} tolerance={tolerance} actual={value}")


def grade_multiple_choice(response: str, expected: str, choices: str = "ABCD") -> GradeResult:
    """MMLU 風格的選擇題評分：只抓獨立出現的選項字母，不做字串模糊比對。

    官方 MMLU/lm-evaluation-harness 是比較各選項的 log-likelihood，
    但透過一般對話 API（如 Ollama chat）拿不到 logprobs，因此採用公認的
    生成式替代方案：prompt 要求「只回答選項字母」，評分時用正則抓第一個
    獨立出現的合法字母（而非整段文字模糊比對），可避免例如選項文字剛好
    出現在解釋句子裡造成的誤判。
    """
    normalized = _normalize(response).strip()
    pattern = rf"\b([{choices}])\b"
    match = re.search(pattern, normalized, re.IGNORECASE)
    if match is None:
        return GradeResult(False, f"no valid choice letter ({choices}) found in response: {normalized!r}")
    picked = match.group(1).upper()
    passed = picked == str(expected).strip().upper()
    return GradeResult(passed, f"expected={expected} picked={picked}")


def grade_gsm8k(response: str, expected: float, tolerance: float = 0, delimiter: str = "####") -> GradeResult:
    """GSM8K 官方資料集的答案格式：推理過程後接 `#### <數字>` 標示最終答案。

    比起從整段文字猜「哪個數字是答案」，這個格式要求模型明確分隔推理與結論，
    評分時只看分隔符之後的內容，徹底消除歧義。若回應裡找不到分隔符，
    視為格式不合規，直接判失敗（而不是 fallback 回去猜數字）——
    這是刻意的設計，用來同時測「有沒有照格式回答」。
    """
    normalized = _normalize(response)
    if delimiter not in normalized:
        return GradeResult(False, f"delimiter {delimiter!r} not found in response (format non-compliant)")
    after = normalized.rsplit(delimiter, 1)[1]
    value = _extract_number(after)
    if value is None:
        return GradeResult(False, f"no number found after delimiter: {after!r}")
    passed = abs(value - float(expected)) <= tolerance
    return GradeResult(passed, f"expected={expected} tolerance={tolerance} actual={value}")


def grade_checks(response: str, checks: list[dict]) -> GradeResult:
    """IFEval 風格的複合條件評分：把一題拆成多個可驗證的子條件。

    回傳兩種分數：
    - strict（`passed`）：所有子條件都通過才算過
    - loose（`score`）：通過的子條件比例（部分分數）

    IFEval 論文本身就是分別報告 strict/loose 兩種準確率，而不是只給單一
    pass/fail，因為「幾乎符合格式」跟「完全不符合格式」在實務上代表不同
    的模型能力程度。
    """
    if not checks:
        raise ValueError("checks 評分器至少需要一個子條件")
    sub_results = [grade(response, sub_check) for sub_check in checks]
    passed_count = sum(1 for r in sub_results if r.passed)
    strict_passed = passed_count == len(sub_results)
    loose_score = passed_count / len(sub_results)
    detail = f"{passed_count}/{len(sub_results)} sub-checks passed: " + "; ".join(
        f"[{'✓' if r.passed else '✗'}] {r.detail}" for r in sub_results
    )
    result = GradeResult(strict_passed, detail)
    result.score = loose_score
    return result


GRADERS = {
    "exact_match": lambda response, grading: grade_exact_match(
        response, grading["expected"], grading.get("case_sensitive", False)
    ),
    "contains": lambda response, grading: grade_contains(
        response, grading["expected"], grading.get("case_sensitive", False)
    ),
    "regex": lambda response, grading: grade_regex(response, grading["expected"]),
    "numeric_tolerance": lambda response, grading: grade_numeric_tolerance(
        response, grading["expected"], grading.get("tolerance", 0)
    ),
    "multiple_choice": lambda response, grading: grade_multiple_choice(
        response, grading["expected"], grading.get("choices", "ABCD")
    ),
    "gsm8k": lambda response, grading: grade_gsm8k(
        response, grading["expected"], grading.get("tolerance", 0), grading.get("delimiter", "####")
    ),
    "checks": lambda response, grading: grade_checks(response, grading["checks"]),
}


def grade(response: str, grading: dict) -> GradeResult:
    grading_type = grading["type"]
    if grading_type not in GRADERS:
        raise ValueError(f"unknown grading type: {grading_type}")
    return GRADERS[grading_type](response, grading)
