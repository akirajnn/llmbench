from __future__ import annotations

import time
from dataclasses import dataclass

from .grading import GradeResult, grade
from .judge import grade_rubric
from .providers import ModelSpec, chat


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
    critical_failure: bool = False
    high_risk: bool = False


def run_task(
    spec: ModelSpec,
    task: dict,
    timeout: int = 120,
    think: bool | None = None,
    sample_index: int = 0,
    temperature: float | None = None,
    judge: ModelSpec | None = None,
) -> TaskResult:
    model = spec.name
    prompt = task["prompt"]
    start = time.monotonic()
    try:
        result = chat(spec, prompt, timeout=timeout, think=think, temperature=temperature)
    except Exception as exc:  # noqa: BLE001 - surface any provider/client error as a failed task
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
            high_risk=task.get("high_risk", False),
        )
    latency = time.monotonic() - start

    response_text = result.text
    eval_count = result.output_tokens
    tokens_per_sec = result.tokens_per_sec

    error = None
    if task["grading"]["type"] != "rubric":
        grade_result = grade(response_text, task["grading"])
    elif judge is None:
        raise ValueError("rubric 題需要評審模型，請在 config.yaml 設定 judge 或用 --judge 指定")
    else:
        try:
            grade_result = grade_rubric(prompt, response_text, task["grading"], judge, timeout)
        except Exception as exc:  # noqa: BLE001 - judge/network failure: keep the response and flag the task
            grade_result = GradeResult(False, "")
            error = f"judge: {exc}"

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
        error=error,
        critical_failure=grade_result.critical,
        high_risk=task.get("high_risk", False),
    )


def run_all(
    models: list[ModelSpec],
    tasks: list[dict],
    timeout: int = 120,
    think: bool | None = None,
    repeat: int = 1,
    temperature: float | None = None,
    judge: ModelSpec | None = None,
) -> list[TaskResult]:
    """對每個 (model, task) 跑 `repeat` 次獨立取樣。

    repeat > 1 時才有統計意義：可以算 pass@k（同一題重複抽樣，
    估計「抽 k 次裡至少對一次」的機率，見 Codex/HumanEval 論文的
    unbiased estimator），而不是只看單次抽樣的 pass/fail。
    """
    results: list[TaskResult] = []
    for spec in models:
        print(f"  執行中：{spec.name}（{spec.provider}）")
        for task in tasks:
            for sample_index in range(repeat):
                results.append(
                    run_task(
                        spec,
                        task,
                        timeout=timeout,
                        think=think,
                        sample_index=sample_index,
                        temperature=temperature,
                        judge=judge,
                    )
                )
    return results
