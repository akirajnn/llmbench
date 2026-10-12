from __future__ import annotations

import math


def wilson_interval(p_hat: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval：小樣本二項比例的信賴區間。

    一般常態近似（Wald interval, p̂ ± z*sqrt(p̂(1-p̂)/n)）在 n 偏小或 p̂
    接近 0/1 時會失準（甚至算出 <0% 或 >100% 的區間）。Wilson interval
    是統計上公認更適合小樣本的方法。z=1.96 對應 95% 信心水準。

    來源：https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
    """
    if n == 0:
        return 0.0, 0.0
    denom = 1 + z**2 / n
    centre = p_hat + z**2 / (2 * n)
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    lo = (centre - margin) / denom
    hi = (centre + margin) / denom
    return max(0.0, lo), min(1.0, hi)


def pass_at_k(n: int, c: int, k: int) -> float:
    """pass@k 無偏估計式（Chen et al. 2021, Codex/HumanEval 論文）。

    naive 作法（重複抽樣 k 次，檢查「是否至少一次通過」）是有偏誤、
    高變異數的估計：等於每次都只抽 k 個樣本來看一次,抽到的樣本湊巧
    全錯就會低估。正確做法是獨立抽樣 n 次（n >= k），記錄通過數 c,
    再用以下公式反推「任選 k 個樣本，至少一個通過」的期望機率：

        pass@k = 1 - C(n-c, k) / C(n, k)

    這裡用數值穩定的連乘形式實作，避免直接算大數的組合數。
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))
