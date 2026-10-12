from __future__ import annotations

import time
from dataclasses import dataclass

import ollama

from .grading import grade


@dataclass
class TaskResult:
    model: str
    task_id: str
    category: str
    prompt: str
    response: str
    passed: bool
    score: float
    detail: str
    latency_s: float
    eval_count: int
    tokens_per_sec: float
    sample_index: int = 0
    error: str | None = None


def run_task(model: str, task: dict, timeout: int = 120, think: bool | None = None, sample_index: int = 0) -> TaskResult:
    prompt = task["prompt"]
    start = time.monotonic()
    try:
        chat_kwargs = {} if think is None else {"think": think}
        result = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **chat_kwargs,
        )
    except Exception as exc:  # noqa: BLE001 - surface any Ollama/client error as a failed task
        latency = time.monotonic() - start
        return TaskResult(
            model=model,
            task_id=task["id"],
            category=task.get("category", ""),
            prompt=prompt,
            response="",
            passed=False,
            score=0.0,
            detail="",
            latency_s=latency,
            eval_count=0,
            tokens_per_sec=0.0,
            sample_index=sample_index,
            error=str(exc),
        )
    latency = time.monotonic() - start

    response_text = result["message"]["content"]
    eval_count = result.get("eval_count", 0) or 0
    eval_duration_ns = result.get("eval_duration", 0) or 0
    tokens_per_sec = (eval_count / (eval_duration_ns / 1e9)) if eval_duration_ns else 0.0

    grade_result = grade(response_text, task["grading"])

    return TaskResult(
        model=model,
        task_id=task["id"],
        category=task.get("category", ""),
        prompt=prompt,
        response=response_text,
        passed=grade_result.passed,
        score=grade_result.score,
        detail=grade_result.detail,
        latency_s=latency,
        eval_count=eval_count,
        tokens_per_sec=tokens_per_sec,
        sample_index=sample_index,
    )


def run_all(
    models: list[str],
    tasks: list[dict],
    timeout: int = 120,
    think: bool | None = None,
    repeat: int = 1,
) -> list[TaskResult]:
    """對每個 (model, task) 跑 `repeat` 次獨立取樣。

    repeat > 1 時才有統計意義：可以算 pass@k（同一題重複抽樣，
    估計「抽 k 次裡至少對一次」的機率，見 Codex/HumanEval 論文的
    unbiased estimator），而不是只看單次抽樣的 pass/fail。
    """
    results: list[TaskResult] = []
    for model in models:
        for task in tasks:
            for sample_index in range(repeat):
                results.append(
                    run_task(model, task, timeout=timeout, think=think, sample_index=sample_index)
                )
    return results
