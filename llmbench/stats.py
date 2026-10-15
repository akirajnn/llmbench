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


def mcnemar_exact(b: int, c: int) -> float:
    """McNemar exact test（雙尾）：兩個模型答同一批題目時的配對檢定。

    b、c 是「不一致」的題數（A 對 B 錯、A 錯 B 對）；兩邊都對或都錯的題目
    不提供差異資訊。虛無假設下 b ~ Binomial(b+c, 0.5)，用精確二項分布算 p 值，
    小樣本也適用。比起把兩個模型的通過率當成獨立樣本比較（看 CI 是否重疊），
    配對檢定扣掉了「題目本身難易」的變異，同樣題數下檢定力高得多。
    """
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_diff_interval(b: int, c: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """配對比例差 (b - c) / n 與其 Wald 信賴區間，回傳 (diff, low, high)。

    n 是兩個模型都有作答的總題數（含一致的題目）。
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    diff = (b - c) / n
    se = math.sqrt(max(b + c - (b - c) ** 2 / n, 0.0)) / n
    return diff, max(-1.0, diff - z * se), min(1.0, diff + z * se)
