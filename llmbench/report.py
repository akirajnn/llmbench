from __future__ import annotations

import json
import statistics
from pathlib import Path

from .runner import TaskResult
from .stats import pass_at_k, wilson_interval


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


def write_markdown_summary(summary: dict[str, dict], results: list[TaskResult], path: Path) -> None:
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
        "## 各題明細",
        "",
        "| model | task_id | category | sample# | passed | loose score | detail |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        detail = r.error if r.error else r.detail
        lines.append(
            f"| {r.model} | {r.task_id} | {r.category} | {r.sample_index} | "
            f"{'✅' if r.passed else '❌'} | {r.score:.2f} | {detail} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
