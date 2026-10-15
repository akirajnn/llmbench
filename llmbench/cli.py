from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import yaml

from .report import (
    format_category_table,
    format_console_table,
    format_paired_table,
    format_relative_table,
    format_rubric_table,
    summarize,
    write_markdown_summary,
    write_raw_jsonl,
)
from .importers import (
    TMMLU_REVISION,
    download_tmmluplus,
    load_tmmluplus,
    read_mc_csv,
    rubric_tasks,
    stratified_sample,
    tmmlu_group,
    to_task,
    write_tasks_yaml,
)
from .providers import ModelSpec
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

    model_entries = args.models.split(",") if args.models else config.get("models", [])
    models = [ModelSpec.parse(m) for m in model_entries]
    task_files = args.tasks.split(",") if args.tasks else config.get("tasks", [])
    options = config.get("options", {})
    timeout = options.get("timeout", 120)

    if args.think is not None:
        think = args.think == "true"
    else:
        think = options.get("think")  # None = 使用模型預設

    repeat = args.repeat if args.repeat is not None else options.get("repeat", 1)
    temperature = options.get("temperature")  # None = 用各模型預設值
    baseline = args.baseline or config.get("baseline")
    if baseline and baseline not in {m.name for m in models}:
        raise SystemExit(f"baseline「{baseline}」不在模型清單中")

    if not models:
        raise SystemExit("沒有指定任何模型，請用 --models 或在 config.yaml 的 models 欄位設定")
    if not task_files:
        raise SystemExit("沒有指定任何任務檔，請用 --tasks 或在 config.yaml 的 tasks 欄位設定")

    tasks = load_tasks(task_files, base_dir)
    judge = _resolve_judge(args.judge or config.get("judge"), models)
    if any(t["grading"]["type"] == "rubric" for t in tasks) and judge is None:
        raise SystemExit("任務中有 rubric 題，需要評審模型：請在 config.yaml 設定 judge 或用 --judge 指定")
    category_weights = {t["category"]: t["category_weight"] for t in tasks if "category_weight" in t}

    print(f"模型: {[m.name for m in models]}")
    print(f"任務數: {len(tasks)}")
    if judge:
        print(f"評審模型: {judge.name}（{judge.provider} / {judge.model}）")
        same_as = [m.name for m in models if (m.provider, m.model) == (judge.provider, judge.model)]
        if same_as:
            print(f"⚠ 評審模型同時也是受測模型（{', '.join(same_as)}），LLM 評審偏好自家回答，分數可能偏高")
    if repeat > 1:
        print(f"每題重複抽樣: {repeat} 次（用於計算 pass@{repeat}，見 Codex/HumanEval 論文）")
    print("開始執行...\n")

    results = run_all(
        models, tasks, timeout=timeout, think=think, repeat=repeat, temperature=temperature, judge=judge
    )

    summary = summarize(results)
    print("\n" + format_console_table(summary))
    print("\n" + format_category_table(results))
    if baseline:
        print("\n" + format_relative_table(results, baseline))
        print("\n" + format_paired_table(results, baseline))
    if category_weights:
        print("\n" + format_rubric_table(results, category_weights))

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base_dir / "results" / timestamp
    write_raw_jsonl(results, out_dir / "raw.jsonl")
    write_markdown_summary(
        summary, results, out_dir / "summary.md", baseline=baseline, category_weights=category_weights
    )
    print(f"\n詳細結果已寫入: {out_dir}")


def _resolve_judge(entry, models: list[ModelSpec]) -> ModelSpec | None:
    """字串先比對模型清單的 name，找不到才視為 Ollama 模型；dict 照 models 的格式解析。"""
    if not entry:
        return None
    if isinstance(entry, str):
        return next((m for m in models if m.name == entry), None) or ModelSpec.parse(entry)
    return ModelSpec.parse(entry)


def cmd_moex_to_csv(args: argparse.Namespace) -> None:
    from collections import Counter

    from .moex import build_rows, find_papers, parse_answers, parse_questions, write_csv

    out_dir = Path(args.out)
    papers = find_papers(Path(args.dir))
    if not papers:
        raise SystemExit(f"{args.dir} 底下找不到 <考試代碼>_<試題代號>.pdf")
    for exam, paper, pdf, answer_pdf in papers:
        questions, subject = parse_questions(pdf)
        answers = parse_answers(answer_pdf)
        rows = build_rows(questions, answers, exam, paper, subject)
        out = out_dir / f"{exam}_{paper}.csv"
        write_csv(rows, out)

        reasons = Counter(r for row in rows for r in row["skip"].split("；") if r)
        free = sum(1 for row in rows if row["answer"] == "一律給分")
        usable = sum(1 for row in rows if not row["skip"] and row["answer"] != "一律給分")
        print(f"{pdf.name}（{subject}，答案：{answer_pdf.name}）→ {out}")
        print(f"  解析 {len(questions)} 題，答案 {len(answers)} 題，可用 {usable} 題，一律給分 {free} 題")
        for reason, count in reasons.most_common():
            print(f"  標記剔除 {count} 題：{reason}")
        missing = sorted(set(answers) - {q.number for q in questions})
        if missing:
            print(f"  ⚠ 答案表有、但試題解析不到的題號：{missing}")
    print("\n請人工檢查 CSV（尤其是 skip 欄有標記的題目），確認後再用 import-exam 匯入。")


def cmd_import_rubric(args: argparse.Namespace) -> None:
    src = Path(args.src)
    with src.open(encoding="utf-8") as f:
        tasks = rubric_tasks(yaml.safe_load(f))
    header = (
        f"由 `llmbench import-rubric` 從 {src.name} 產生，共 {len(tasks)} 題。\n"
        "rubric 題需要評審模型（config.yaml 的 judge 或 --judge）。"
    )
    out = Path(args.out)
    write_tasks_yaml(tasks, out, header)
    by_cat: dict[str, int] = {}
    for t in tasks:
        by_cat[t["category"]] = by_cat.get(t["category"], 0) + 1
    print(f"輸出 {len(tasks)} 題 → {out}（高風險題 {sum(t['high_risk'] for t in tasks)} 題）")
    for cat, count in by_cat.items():
        weight = next((t.get("category_weight") for t in tasks if t["category"] == cat), None)
        print(f"  {cat}: {count} 題" + (f"，權重 {weight}%" if weight is not None else ""))


def _report_import(questions: list, sampled: list, skipped: list[str], out: Path, category_of) -> None:
    by_cat: dict[str, int] = {}
    for q in sampled:
        by_cat[category_of(q)] = by_cat.get(category_of(q), 0) + 1
    print(f"可用題數: {len(questions)}，剔除: {len(skipped)}，輸出: {len(sampled)} → {out}")
    for cat, count in sorted(by_cat.items()):
        print(f"  {cat}: {count}")
    if skipped:
        log = out.with_suffix(".skipped.txt")
        log.write_text("\n".join(skipped) + "\n", encoding="utf-8")
        print(f"剔除明細: {log}")


def cmd_import_tmmluplus(args: argparse.Namespace) -> None:
    src = Path(args.src)
    if args.download:
        download_tmmluplus(src, split=args.split)
    questions, skipped = load_tmmluplus(src, split=args.split)
    sampled = stratified_sample(questions, key=lambda q: q.subject, n=args.n, seed=args.seed)

    def category_of(q) -> str:
        return f"{args.category}_{tmmlu_group(q.subject)}" if args.split_by_group else args.category

    out = Path(args.out)
    header = (
        f"TMMLU+ {TMMLU_REVISION}（ikala/tmmluplus，MIT License）{args.split} split，\n"
        f"依學科比例分層抽樣 {len(sampled)} 題（seed={args.seed}），由 `llmbench import-tmmluplus` 產生。\n"
        "注意：TMMLU+ 自 2023 年公開，可能已在模型訓練資料中，分數可能偏高。"
    )
    write_tasks_yaml([to_task(q, category_of(q)) for q in sampled], out, header)
    _report_import(questions, sampled, skipped, out, category_of)


def cmd_import_exam(args: argparse.Namespace) -> None:
    questions: list = []
    skipped: list[str] = []
    for path in args.src.split(","):
        qs, sk = read_mc_csv(Path(path), id_prefix=args.id_prefix or Path(path).stem)
        questions += qs
        skipped += sk
    if args.min_year is not None:
        kept = []
        for q in questions:
            if q.year.isdigit() and int(q.year) >= args.min_year:
                kept.append(q)
            else:
                skipped.append(f"{q.id}: year={q.year or '(空白)'} 早於 --min-year {args.min_year}")
        questions = kept
    ids = [q.id for q in questions]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise SystemExit(f"題目 id 重複，請檢查 CSV 或改用 --id-prefix：{dupes[:10]}")

    n = args.n if args.n is not None else len(questions)
    sampled = stratified_sample(questions, key=lambda q: q.subject, n=n, seed=args.seed)
    out = Path(args.out)
    header = (
        f"由 `llmbench import-exam` 從 {args.src} 產生，共 {len(sampled)} 題（seed={args.seed}）。\n"
        "送往雲端模型前，請確認題目不含真實病歷或個資。"
    )
    write_tasks_yaml([to_task(q, args.category) for q in sampled], out, header)
    _report_import(questions, sampled, skipped, out, lambda q: args.category)


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
    run_parser.add_argument(
        "--baseline",
        default=None,
        help="作為錨點的模型名稱（通常是雲端模型），報表會算其他模型相對它的分數比例",
    )
    run_parser.add_argument(
        "--judge",
        default=None,
        help="rubric 題的評審模型：模型清單中的 name，或 Ollama 模型名稱（覆蓋 config.yaml 的 judge）",
    )
    run_parser.set_defaults(func=cmd_run)

    moex_parser = subparsers.add_parser(
        "moex-to-csv", help="把考選部試題／答案 PDF 轉成 CSV，人工檢查後再用 import-exam 匯入"
    )
    moex_parser.add_argument("--dir", required=True, help="放 <考試代碼>_<代號>.pdf、_ANS、_MOD 檔的資料夾")
    moex_parser.add_argument("--out", required=True, help="輸出 CSV 的資料夾")
    moex_parser.set_defaults(func=cmd_moex_to_csv)

    rubric_parser = subparsers.add_parser("import-rubric", help="把評分準則題組（benchmark + questions）轉成任務 YAML")
    rubric_parser.add_argument("--src", required=True, help="評分準則題組 YAML")
    rubric_parser.add_argument("--out", required=True, help="輸出的任務 YAML 路徑")
    rubric_parser.set_defaults(func=cmd_import_rubric)

    tmmlu_parser = subparsers.add_parser("import-tmmluplus", help="把 TMMLU+ 分層抽樣轉成任務 YAML")
    tmmlu_parser.add_argument("--src", required=True, help="放 <subject>_<split>.csv 的資料夾")
    tmmlu_parser.add_argument("--out", required=True, help="輸出的任務 YAML 路徑")
    tmmlu_parser.add_argument("--n", type=int, default=300, help="抽樣題數（預設 300）")
    tmmlu_parser.add_argument("--seed", type=int, default=42, help="抽樣亂數種子，固定才能重現同一份題組")
    tmmlu_parser.add_argument("--split", default="test", choices=["test", "val", "dev", "train"])
    tmmlu_parser.add_argument("--category", default="tmmluplus", help="報表分類名稱（預設 tmmluplus）")
    tmmlu_parser.add_argument(
        "--split-by-group",
        action="store_true",
        help="分類改為 <category>_<stem|social|humanities|other>，報表可分學科群比較",
    )
    tmmlu_parser.add_argument(
        "--download", action="store_true", help=f"先從 Hugging Face 下載 {TMMLU_REVISION} 的 CSV 到 --src"
    )
    tmmlu_parser.set_defaults(func=cmd_import_tmmluplus)

    exam_parser = subparsers.add_parser("import-exam", help="把選擇題 CSV（如國考題）轉成任務 YAML")
    exam_parser.add_argument("--src", required=True, help="逗號分隔的 CSV 路徑")
    exam_parser.add_argument("--out", required=True, help="輸出的任務 YAML 路徑")
    exam_parser.add_argument("--category", required=True, help="報表分類名稱，例如 醫療專業")
    exam_parser.add_argument("--n", type=int, default=None, help="依 subject 分層抽樣的題數（預設全部）")
    exam_parser.add_argument("--seed", type=int, default=42)
    exam_parser.add_argument("--min-year", type=int, default=None, help="只保留 year 欄 >= 此值的題目")
    exam_parser.add_argument("--id-prefix", default=None, help="CSV 沒有 id 欄時的題號前綴（預設為檔名）")
    exam_parser.set_defaults(func=cmd_import_exam)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
