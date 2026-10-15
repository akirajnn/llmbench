from __future__ import annotations

import json
import statistics
import unicodedata
from pathlib import Path

from .runner import TaskResult
from .stats import mcnemar_exact, paired_diff_interval, pass_at_k, wilson_interval


def write_raw_jsonl(results: list[TaskResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")


def summarize(results: list[TaskResult]) -> dict[str, dict]:
    by_model: dict[str, list[TaskResult]] = {}
    for r in results:
        by_model.setdefault(r.model, []).append(r)

    summary = {}
    for model, model_results in by_model.items():
        total = len(model_results)
        passed = sum(1 for r in model_results if r.passed)
        errored = sum(1 for r in model_results if r.error)
        latencies = [r.latency_s for r in model_results]
        tps_values = [r.tokens_per_sec for r in model_results if r.tokens_per_sec > 0]
        slowest = max(model_results, key=lambda r: r.latency_s) if model_results else None
        pass_rate = passed / total if total else 0.0
        ci_lo, ci_hi = wilson_interval(pass_rate, total)
        avg_loose_score = statistics.mean(r.score for r in model_results) if model_results else 0.0

        # 依 task_id 分組，算每題的重複抽樣數（n）與通過數（c），
        # 用來算 pass@k。repeat=1 時每組 n=1，pass@k 只能算 k=1（跟
        # pass_rate 相同，不另外顯示）。
        by_task: dict[str, list[TaskResult]] = {}
        for r in model_results:
            by_task.setdefault(r.task_id, []).append(r)
        sample_counts = {len(v) for v in by_task.values()}
        repeat_n = max(sample_counts) if sample_counts else 1

        pass_at_k_value = None
        if repeat_n > 1:
            per_task_pass_at_k = [
                pass_at_k(len(v), sum(1 for r in v if r.passed), repeat_n) for v in by_task.values()
            ]
            pass_at_k_value = statistics.mean(per_task_pass_at_k) if per_task_pass_at_k else None

        summary[model] = {
            "total": total,
            "passed": passed,
            "pass_rate": pass_rate * 100,
            "ci_low": ci_lo * 100,
            "ci_high": ci_hi * 100,
            "avg_loose_score": avg_loose_score * 100,
            "errored": errored,
            "avg_latency_s": statistics.mean(latencies) if latencies else 0.0,
            "median_latency_s": statistics.median(latencies) if latencies else 0.0,
            "avg_tokens_per_sec": statistics.mean(tps_values) if tps_values else 0.0,
            "slowest_task_id": slowest.task_id if slowest else "",
            "slowest_latency_s": slowest.latency_s if slowest else 0.0,
            "repeat_n": repeat_n,
            "pass_at_k_value": pass_at_k_value * 100 if pass_at_k_value is not None else None,
        }
    return summary


def category_pass_rates(results: list[TaskResult]) -> dict[str, dict[str, tuple[int, int]]]:
    """回傳 {model: {category: (通過數, 總數)}}，另含 "ALL" 總計。"""
    table: dict[str, dict[str, list[int]]] = {}
    for r in results:
        for cat in (r.category or "uncategorized", "ALL"):
            cell = table.setdefault(r.model, {}).setdefault(cat, [0, 0])
            cell[0] += 1 if r.passed else 0
            cell[1] += 1
    return {m: {c: (v[0], v[1]) for c, v in cats.items()} for m, cats in table.items()}


def category_score_rows(results: list[TaskResult]) -> tuple[list[str], list[tuple[str, list[str], int]]]:
    """各分類得分表：欄位依題目順序排列，最後一欄為總分；模型依總分由高到低。

    出錯（連線失敗、timeout 等）的題目不計入分母，另外回傳錯誤數，
    避免把基礎設施問題算成模型答錯。
    """
    categories: list[str] = []
    table: dict[str, dict[str, list[int]]] = {}
    errors: dict[str, int] = {}
    for r in results:
        cat = r.category or "uncategorized"
        if cat not in categories:
            categories.append(cat)
        errors.setdefault(r.model, 0)
        if r.error:
            errors[r.model] += 1
            continue
        for key in (cat, "總分"):
            cell = table.setdefault(r.model, {}).setdefault(key, [0, 0])
            cell[0] += 1 if r.passed else 0
            cell[1] += 1
    columns = categories + ["總分"]

    def rate(cell: list[int]) -> float:
        return cell[0] / cell[1] if cell[1] else -1.0

    rows = []
    for model in sorted(errors, key=lambda m: -rate(table.get(m, {}).get("總分", [0, 0]))):
        cats = table.get(model, {})
        cells = [f"{p / n * 100:.1f}% ({p}/{n})" if (p, n) != (0, 0) else "—" for p, n in (cats.get(c, [0, 0]) for c in columns)]
        rows.append((model, cells, errors[model]))
    return columns, rows


def _display_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _pad(text: str, width: int, right: bool = False) -> str:
    gap = " " * max(0, width - _display_width(text))
    return gap + text if right else text + gap


def format_category_table(results: list[TaskResult]) -> str:
    columns, rows = category_score_rows(results)
    headers = ["model", *columns, "錯誤"]
    body = [[model, *cells, str(err)] for model, cells, err in rows]
    widths = [max(_display_width(row[i]) for row in [headers, *body]) for i in range(len(headers))]

    def fmt(row: list[str]) -> str:
        return "  ".join(_pad(c, w, right=i > 0) for i, (c, w) in enumerate(zip(row, widths)))

    lines = ["各分類得分（依總分排序）", fmt(headers), "-" * _display_width(fmt(headers))]
    lines += [fmt(row) for row in body]
    if any(err for _, _, err in rows):
        lines.append("錯誤題（連線失敗、timeout 等）不計入分母，修好後可只重跑這些題目。")
    return "\n".join(lines)


def relative_rows(results: list[TaskResult], baseline: str) -> tuple[list[str], list[tuple[str, list[str]]]]:
    """各分類下，每個模型的通過率與「相對 baseline 的比例」。

    相對比例 = 模型通過率 / baseline 通過率。baseline 通常放雲端模型，
    比例即「本地模型達到雲端模型的幾成」，比絕對分數更能跨題組解讀。
    baseline 在該分類通過率為 0 時無法計算，顯示「—」。
    """
    rates = category_pass_rates(results)
    categories = sorted({c for cats in rates.values() for c in cats if c != "ALL"}) + ["ALL"]
    base = rates.get(baseline, {})
    rows = []
    for model, cats in rates.items():
        cells = []
        for cat in categories:
            if cat not in cats:
                cells.append("—")
                continue
            p, n = cats[cat]
            rate = p / n if n else 0.0
            if model == baseline:
                cells.append(f"{rate*100:.0f}% (基準)")
                continue
            bp, bn = base.get(cat, (0, 0))
            brate = bp / bn if bn else 0.0
            rel = f"{rate/brate*100:.0f}%" if brate else "—"
            cells.append(f"{rate*100:.0f}% → {rel}")
        rows.append((model, cells))
    return categories, rows


def format_relative_table(results: list[TaskResult], baseline: str) -> str:
    categories, rows = relative_rows(results, baseline)
    width = max(18, *(len(c) + 2 for c in categories))
    header = f"{'model':<20}" + "".join(f"{c:>{width}}" for c in categories)
    lines = [f"各分類通過率 → 相對 baseline「{baseline}」的比例", header, "-" * len(header)]
    for model, cells in rows:
        lines.append(f"{model:<20}" + "".join(f"{c:>{width}}" for c in cells))
    lines.append("")
    lines.append("解讀：「80% → 89%」代表該模型通過率 80%，約達 baseline 的 89%。")
    return "\n".join(lines)


def paired_rows(results: list[TaskResult], baseline: str) -> list[dict]:
    """每個非 baseline 模型、每個分類（最後接 ALL）對 baseline 的配對比較。

    只用 sample_index == 0 配對：repeat > 1 時同一題的多次抽樣彼此相關，
    全部丟進 McNemar 會把題數灌水、p 值偏小。
    """
    outcome: dict[str, dict[str, tuple[str, bool]]] = {}
    for r in results:
        if r.sample_index == 0:
            outcome.setdefault(r.model, {})[r.task_id] = (r.category or "uncategorized", r.passed)
    base = outcome.get(baseline, {})
    categories = sorted({cat for cat, _ in base.values()}) + ["ALL"]

    rows = []
    for model, answers in outcome.items():
        if model == baseline:
            continue
        for cat in categories:
            shared = [
                (passed, base[task_id][1])
                for task_id, (task_cat, passed) in answers.items()
                if task_id in base and cat in (task_cat, "ALL")
            ]
            if not shared:
                continue
            b = sum(1 for m, bl in shared if m and not bl)
            c = sum(1 for m, bl in shared if not m and bl)
            diff, lo, hi = paired_diff_interval(b, c, len(shared))
            rows.append(
                {
                    "model": model,
                    "category": cat,
                    "n": len(shared),
                    "model_wins": b,
                    "baseline_wins": c,
                    "diff": diff * 100,
                    "ci_low": lo * 100,
                    "ci_high": hi * 100,
                    "p_value": mcnemar_exact(b, c),
                }
            )
    return rows


def _paired_cells(row: dict) -> list[str]:
    p = row["p_value"]
    p_text = "<0.001" if p < 0.001 else f"{p:.3f}"
    return [
        row["model"],
        row["category"],
        str(row["n"]),
        f"{row['model_wins']}/{row['baseline_wins']}",
        f"{row['diff']:+.1f}",
        f"[{row['ci_low']:+.1f}, {row['ci_high']:+.1f}]",
        p_text + (" *" if p < 0.05 else ""),
    ]


_PAIRED_HEADERS = ["model", "category", "n", "勝/負", "差距(點)", "95% CI", "McNemar p"]
_PAIRED_NOTE = (
    "勝/負＝模型對但 baseline 錯／模型錯但 baseline 對的題數；差距為負代表落後 baseline。"
    "* 表示 p < 0.05。只用第一次抽樣配對。"
)


def format_paired_table(results: list[TaskResult], baseline: str) -> str:
    rows = [_paired_cells(r) for r in paired_rows(results, baseline)]
    widths = [max(len(c) for c in col) + 2 for col in zip(_PAIRED_HEADERS, *rows)]

    def fmt(cells: list[str]) -> str:
        return "".join(f"{c:<{w}}" if i < 2 else f"{c:>{w}}" for i, (c, w) in enumerate(zip(cells, widths)))

    header = fmt(_PAIRED_HEADERS)
    lines = [f"配對檢定：各模型 vs baseline「{baseline}」", header, "-" * len(header)]
    lines += [fmt(cells) for cells in rows]
    lines += ["", _PAIRED_NOTE]
    return "\n".join(lines)


def rubric_summary(results: list[TaskResult], weights: dict[str, float]) -> dict[str, dict]:
    """評分準則題組的加權總分與重大錯誤統計。

    加權總分 = Σ(分類平均得分 × 分類權重) / Σ(有作答分類的權重)，
    得分用 0–1 的部分分數（rubric 總分 / 滿分），重大錯誤題為 0。
    """
    by_model: dict[str, dict[str, list[float]]] = {}
    critical: dict[str, list[TaskResult]] = {}
    for r in results:
        if r.category in weights:
            by_model.setdefault(r.model, {}).setdefault(r.category, []).append(r.score)
        if r.critical_failure:
            critical.setdefault(r.model, []).append(r)

    summary = {}
    for model, cats in by_model.items():
        total_weight = sum(weights[c] for c in cats)
        weighted = sum(statistics.mean(scores) * weights[c] for c, scores in cats.items()) / total_weight
        failures = critical.get(model, [])
        summary[model] = {
            "weighted_score": weighted * 100,
            "critical": len(failures),
            "high_risk_critical": sum(1 for r in failures if r.high_risk),
            "judge_errors": sum(1 for r in results if r.model == model and r.category in weights and r.error),
        }
    return summary


def critical_failure_rows(results: list[TaskResult]) -> list[TaskResult]:
    return sorted((r for r in results if r.critical_failure), key=lambda r: (not r.high_risk, r.model, r.task_id))


def format_rubric_table(results: list[TaskResult], weights: dict[str, float]) -> str:
    header = f"{'model':<20} {'加權總分':>8} {'重大錯誤':>8} {'其中高風險題':>12} {'評審錯誤':>8}"
    lines = ["評分準則題組（rubric）", header, "-" * 64]
    for model, s in rubric_summary(results, weights).items():
        lines.append(
            f"{model:<20} {s['weighted_score']:>11.1f}% {s['critical']:>11} "
            f"{s['high_risk_critical']:>17} {s['judge_errors']:>11}"
        )
    failures = critical_failure_rows(results)
    if failures:
        lines += ["", "重大錯誤明細（高風險題優先）："]
        lines += [
            f"  {'[高風險] ' if r.high_risk else ''}{r.model} {r.task_id}#{r.sample_index}: {' '.join(r.detail.split())}"
            for r in failures
        ]
    lines += ["", "分數由評審模型給出，正式報告前請抽樣人工複核；評審錯誤的題目以 0 分計入。"]
    return "\n".join(lines)


def _md_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def format_console_table(summary: dict[str, dict]) -> str:
    header = (
        f"{'model':<20} {'pass':>8} {'pass_rate (95% CI)':>22} {'avg_latency_s':>14} "
        f"{'median_s':>9} {'avg_tok/s':>10} {'errors':>7} {'slowest_task':>14}"
    )
    lines = [header, "-" * len(header)]
    for model, stats in summary.items():
        ci = f"[{stats['ci_low']:.0f}-{stats['ci_high']:.0f}%]"
        lines.append(
            f"{model:<20} {stats['passed']:>3}/{stats['total']:<4} "
            f"{stats['pass_rate']:>7.1f}% {ci:>13} {stats['avg_latency_s']:>13.2f}s "
            f"{stats['median_latency_s']:>8.2f}s {stats['avg_tokens_per_sec']:>9.1f} "
            f"{stats['errored']:>7} "
            f"{stats['slowest_task_id']}({stats['slowest_latency_s']:.1f}s)"
        )
        if stats["pass_at_k_value"] is not None:
            lines.append(
                f"{'':<20} pass@{stats['repeat_n']}={stats['pass_at_k_value']:.1f}%  "
                f"(每題重複抽樣 {stats['repeat_n']} 次；loose score={stats['avg_loose_score']:.1f}%)"
            )
    lines.append("")
    lines.append(
        "注意：95% CI 用 Wilson score interval 估計，樣本數偏小時（如 <30 題）"
        "區間仍會很寬，結論僅供參考方向，不宜當成嚴謹的統計結論。"
    )
    return "\n".join(lines)


def write_markdown_summary(
    summary: dict[str, dict],
    results: list[TaskResult],
    path: Path,
    baseline: str | None = None,
    category_weights: dict[str, float] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 評測結果",
        "",
        "| model | pass | pass_rate | 95% CI (Wilson) | loose score | pass@k | avg_latency_s | "
        "median_latency_s | avg_tokens/s | errors | slowest_task |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for model, stats in summary.items():
        ci = f"{stats['ci_low']:.0f}–{stats['ci_high']:.0f}%"
        pass_at_k_cell = (
            f"pass@{stats['repeat_n']}={stats['pass_at_k_value']:.1f}%"
            if stats["pass_at_k_value"] is not None
            else "n/a (repeat=1)"
        )
        lines.append(
            f"| {model} | {stats['passed']}/{stats['total']} | {stats['pass_rate']:.1f}% | {ci} | "
            f"{stats['avg_loose_score']:.1f}% | {pass_at_k_cell} | "
            f"{stats['avg_latency_s']:.2f} | {stats['median_latency_s']:.2f} | "
            f"{stats['avg_tokens_per_sec']:.1f} | {stats['errored']} | "
            f"{stats['slowest_task_id']} ({stats['slowest_latency_s']:.1f}s) |"
        )

    lines += [
        "",
        "> 95% CI 用 Wilson score interval（適合小樣本的二項比例信賴區間，"
        "見 [Wikipedia](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval)）。"
        "樣本數小時區間會偏寬，請勿過度解讀。",
        "",
    ]
    columns, rows = category_score_rows(results)
    lines += [
        "## 各分類得分",
        "",
        "| model | " + " | ".join(columns) + " | 錯誤 |",
        "|---|" + "---:|" * (len(columns) + 1),
    ]
    lines += [f"| {model} | " + " | ".join(cells) + f" | {err} |" for model, cells, err in rows]
    lines += ["", "> 錯誤題（連線失敗、timeout 等）不計入分母。", ""]
    if baseline:
        categories, rows = relative_rows(results, baseline)
        lines += [
            f"## 相對 baseline（{baseline}）的表現",
            "",
            "格式：該分類通過率 → 相對 baseline 的比例。",
            "",
            "| model | " + " | ".join(categories) + " |",
            "|---|" + "---|" * len(categories),
        ]
        lines += [f"| {model} | " + " | ".join(cells) + " |" for model, cells in rows]
        lines += [
            "",
            "### 配對檢定（McNemar）",
            "",
            "| " + " | ".join(_PAIRED_HEADERS) + " |",
            "|---" * len(_PAIRED_HEADERS) + "|",
        ]
        lines += ["| " + " | ".join(_paired_cells(r)) + " |" for r in paired_rows(results, baseline)]
        lines += ["", f"> {_PAIRED_NOTE}", ""]
    if category_weights:
        lines += [
            "## 評分準則題組（rubric）",
            "",
            "權重：" + "、".join(f"{c} {w:g}%" for c, w in category_weights.items()),
            "",
            "| model | 加權總分 | 重大錯誤 | 其中高風險題 | 評審錯誤 |",
            "|---|---|---|---|---|",
        ]
        lines += [
            f"| {model} | {s['weighted_score']:.1f}% | {s['critical']} | {s['high_risk_critical']} | {s['judge_errors']} |"
            for model, s in rubric_summary(results, category_weights).items()
        ]
        failures = critical_failure_rows(results)
        if failures:
            lines += ["", "### 重大錯誤明細", "", "| model | task_id | sample# | 高風險 | 評審說明 |", "|---|---|---|---|---|"]
            lines += [
                f"| {r.model} | {r.task_id} | {r.sample_index} | {'⚠️' if r.high_risk else ''} | {_md_cell(r.detail)} |"
                for r in failures
            ]
        lines += ["", "> 分數由評審模型給出，正式報告前請抽樣人工複核；評審錯誤的題目以 0 分計入。", ""]
    lines += [
        "## 各題明細",
        "",
        "| model | task_id | category | sample# | passed | loose score | detail |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        detail = _md_cell(r.error if r.error else r.detail)
        lines.append(
            f"| {r.model} | {r.task_id} | {r.category} | {r.sample_index} | "
            f"{'✅' if r.passed else '❌'} | {r.score:.2f} | {detail} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
