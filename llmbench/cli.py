from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import yaml

from .report import format_console_table, summarize, write_markdown_summary, write_raw_jsonl
from .runner import run_all


def load_tasks(task_file_paths: list[str], base_dir: Path) -> list[dict]:
    tasks: list[dict] = []
    for task_file in task_file_paths:
        path = base_dir / task_file
        with path.open(encoding="utf-8") as f:
            tasks.extend(yaml.safe_load(f))
    return tasks


def cmd_run(args: argparse.Namespace) -> None:
    base_dir = Path(args.config).parent if args.config else Path.cwd()

    if args.config:
        with open(args.config, encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    models = args.models.split(",") if args.models else config.get("models", [])
    task_files = args.tasks.split(",") if args.tasks else config.get("tasks", [])
    options = config.get("options", {})
    timeout = options.get("timeout", 120)

    if args.think is not None:
        think = args.think == "true"
    else:
        think = options.get("think")  # None = 使用模型預設

    repeat = args.repeat if args.repeat is not None else options.get("repeat", 1)

    if not models:
        raise SystemExit("沒有指定任何模型，請用 --models 或在 config.yaml 的 models 欄位設定")
    if not task_files:
        raise SystemExit("沒有指定任何任務檔，請用 --tasks 或在 config.yaml 的 tasks 欄位設定")

    tasks = load_tasks(task_files, base_dir)

    print(f"模型: {models}")
    print(f"任務數: {len(tasks)}")
    if repeat > 1:
        print(f"每題重複抽樣: {repeat} 次（用於計算 pass@{repeat}，見 Codex/HumanEval 論文）")
    print("開始執行...\n")

    results = run_all(models, tasks, timeout=timeout, think=think, repeat=repeat)

    summary = summarize(results)
    print("\n" + format_console_table(summary))

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base_dir / "results" / timestamp
    write_raw_jsonl(results, out_dir / "raw.jsonl")
    write_markdown_summary(summary, results, out_dir / "summary.md")
    print(f"\n詳細結果已寫入: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="llmbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="執行評測")
    run_parser.add_argument("--config", help="config.yaml 路徑")
    run_parser.add_argument("--models", help="逗號分隔的模型清單，覆蓋 config.yaml")
    run_parser.add_argument("--tasks", help="逗號分隔的任務檔路徑，覆蓋 config.yaml")
    run_parser.add_argument(
        "--think",
        choices=["true", "false"],
        default=None,
        help="是否開啟模型的 thinking 模式（支援的模型才有效），不指定則用模型預設值",
    )
    run_parser.add_argument(
        "--repeat",
        type=int,
        default=None,
        help="每題重複抽樣次數，>1 時會額外計算 pass@k 與信賴區間（預設 1，覆蓋 config.yaml）",
    )
    run_parser.set_defaults(func=cmd_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
