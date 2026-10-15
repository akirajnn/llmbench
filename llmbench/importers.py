"""把外部題庫轉成 llmbench 的任務 YAML。

支援三種來源：
- 評分準則題組（benchmark + questions 格式的開放式情境題），轉成需評審模型
  評分的 rubric 任務。
- TMMLU+（iKala，MIT 授權）：每個學科一個 `<subject>_<split>.csv`，欄位
  question,A,B,C,D,answer。可依學科分層抽樣，避免全集約 2 萬題都送雲端。
- 通用選擇題 CSV（例如自行整理的醫師國考題）：欄位 question,A,B,C,D,answer，
  可選 id、year、subject、skip、E。answer 支援更正答案常見的多答案
  （如 "A,C"），「一律給分」的題目會被剔除——所有選項都算對的題目
  無法鑑別模型能力，留著只會灌水。

分層抽樣用「比例配置 + 最大餘數法」：每個學科抽出的題數與它在母體中的
比例一致，抽出的子集在學科組成上與全集相同，整體分數才能代表全集。
"""

from __future__ import annotations

import csv
import random
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

TMMLU_REVISION = "v1.1"
TMMLU_BASE_URL = "https://huggingface.co/datasets/ikala/tmmluplus/resolve"

# 學科 → (中文名, 子分類)，取自 iKala/ievals 的 ievals/subject.tsv（MIT License）。
TMMLU_SUBJECTS: dict[str, tuple[str, str]] = {
    "dentistry": ("牙醫學", "health"),
    "traditional_chinese_medicine_clinical_medicine": ("中醫臨床醫學", "health"),
    "clinical_psychology": ("臨床心理學", "psychology"),
    "technical": ("技術工相關", "other"),
    "culinary_skills": ("餐旅", "other"),
    "mechanical": ("機械與機電概論", "other"),
    "logic_reasoning": ("邏輯思維", "other"),
    "real_estate": ("房地產", "other"),
    "general_principles_of_law": ("法學大意", "law"),
    "finance_banking": ("金融與法規", "business"),
    "anti_money_laundering": ("洗錢防制", "law"),
    "ttqav2": ("台灣在地用語", "culture"),
    "marketing_management": ("行銷管理", "other"),
    "business_management": ("企業管理", "other"),
    "organic_chemistry": ("有機化學", "chemistry"),
    "advance_chemistry": ("化學", "chemistry"),
    "physics": ("物理", "physics"),
    "secondary_physics": ("高中物理", "physics"),
    "human_behavior": ("人類行為與社會", "psychology"),
    "national_protection": ("軍事", "politics"),
    "jce_humanities": ("指考人文科目", "philosophy"),
    "politic_science": ("政治", "politics"),
    "agriculture": ("農業", "other"),
    "official_document_management": ("機關文書", "other"),
    "financial_analysis": ("財務分析", "business"),
    "pharmacy": ("藥劑學", "biology"),
    "educational_psychology": ("教育心理", "psychology"),
    "statistics_and_machine_learning": ("統計與機器學習", "engineering"),
    "management_accounting": ("管理會計", "business"),
    "introduction_to_law": ("法律概論", "law"),
    "computer_science": ("資訊工程", "computer science"),
    "veterinary_pathology": ("獸醫病理學", "health"),
    "accounting": ("會計學", "business"),
    "fire_science": ("火災學", "other"),
    "optometry": ("視光學", "other"),
    "insurance_studies": ("保險學", "other"),
    "pharmacology": ("藥理學", "health"),
    "taxation": ("稅務", "law"),
    "education_(profession_level)": ("教育專業", "education"),
    "economics": ("經濟學", "economics"),
    "veterinary_pharmacology": ("獸醫藥理學", "health"),
    "nautical_science": ("航海", "other"),
    "occupational_therapy_for_psychological_disorders": ("心理障礙職能治療學", "psychology"),
    "trust_practice": ("信託實務", "law"),
    "geography_of_taiwan": ("台灣地理", "geography"),
    "physical_education": ("體育", "education"),
    "auditing": ("審計學", "business"),
    "administrative_law": ("行政法", "law"),
    "basic_medical_science": ("基礎醫學", "biology"),
    "macroeconomics": ("總經", "economics"),
    "trade": ("貿易", "business"),
    "chinese_language_and_literature": ("國文", "culture"),
    "tve_design": ("統測＿設計", "other"),
    "junior_science_exam": ("國中會考基測自然科", "biology"),
    "junior_math_exam": ("國中會考基測數學科", "math"),
    "junior_chinese_exam": ("國中會考基測國文", "culture"),
    "junior_social_studies": ("國中會考基測社會科", "other"),
    "tve_mathematics": ("統測數學", "math"),
    "tve_chinese_language": ("統測國文", "culture"),
    "tve_natural_sciences": ("統測自然科", "biology"),
    "junior_chemistry": ("國中理化", "chemistry"),
    "music": ("音樂科", "other"),
    "education": ("教育常識", "education"),
    "three_principles_of_people": ("三民主義", "culture"),
    "taiwanese_hokkien": ("閩南語", "culture"),
    "engineering_math": ("工程數學", "math"),
}

# 子分類 → 四大類，取自 ievals/settings.py 的 categories。
TMMLU_GROUPS: dict[str, str] = {
    **dict.fromkeys(["physics", "chemistry", "biology", "computer science", "math", "engineering"], "stem"),
    **dict.fromkeys(["history", "philosophy", "law"], "humanities"),
    **dict.fromkeys(["politics", "culture", "economics", "geography", "psychology", "education"], "social"),
    **dict.fromkeys(["other", "business", "health"], "other"),
}

_FREE_POINT_MARKERS = {"*", "#", "＊", "一律給分", "送分", "全部給分"}

MC_PROMPT = "請回答下列單選題，只回答選項字母（{letters}），不要有其他文字。\n\n{question}\n{options}\n"


@dataclass
class MCQuestion:
    id: str
    subject: str
    question: str
    options: dict[str, str]
    answers: list[str]
    year: str = ""
    extra: dict = field(default_factory=dict)


def parse_answer(raw: str, letters: str) -> list[str] | None:
    """解析答案欄。回傳 None 代表「一律給分」，應剔除。

    "C" → ["C"]；"A,C"、"A/C"、"A或C"、"AC" → ["A", "C"]。
    出現選項以外的字元時拋出 ValueError，避免錯的答案鍵默默進題庫。
    """
    text = raw.strip().upper()
    if text in _FREE_POINT_MARKERS or text == "":
        return None
    for sep in (",", "，", "、", "/", "或", " "):
        text = text.replace(sep, "")
    if not text or any(ch not in letters for ch in text):
        raise ValueError(f"無法解析的答案：{raw!r}")
    if set(text) == set(letters):
        return None
    return sorted(set(text))


def read_mc_csv(path: Path, subject: str = "", id_prefix: str = "") -> tuple[list[MCQuestion], list[str]]:
    """讀取選擇題 CSV，回傳 (題目, 被剔除的原因清單)。"""
    questions: list[MCQuestion] = []
    skipped: list[str] = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        letters = "".join(ch for ch in "ABCDE" if ch in (reader.fieldnames or []))
        for row_no, row in enumerate(reader, start=1):
            qid = (row.get("id") or "").strip() or f"{id_prefix or path.stem}_{row_no:04d}"
            if (row.get("skip") or "").strip():
                skipped.append(f"{qid}: skip 欄標記（{row['skip'].strip()}）")
                continue
            options = {ch: (row.get(ch) or "").strip() for ch in letters}
            if "E" in options and not options["E"]:
                del options["E"]
            if not (row.get("question") or "").strip() or not all(options.values()):
                skipped.append(f"{qid}: 題幹或選項空白")
                continue
            try:
                answers = parse_answer(row.get("answer") or "", "".join(options))
            except ValueError as exc:
                skipped.append(f"{qid}: {exc}")
                continue
            if answers is None:
                skipped.append(f"{qid}: 一律給分")
                continue
            questions.append(
                MCQuestion(
                    id=qid,
                    subject=(row.get("subject") or "").strip() or subject,
                    question=row["question"].strip(),
                    options=options,
                    answers=answers,
                    year=(row.get("year") or "").strip(),
                )
            )
    return questions, skipped


def download_tmmluplus(dest: Path, split: str = "test", revision: str = TMMLU_REVISION) -> None:
    """下載 TMMLU+ 指定 split 的 CSV；已存在的檔案不重抓。"""
    dest.mkdir(parents=True, exist_ok=True)
    for subject in TMMLU_SUBJECTS:
        target = dest / f"{subject}_{split}.csv"
        if target.exists():
            continue
        url = f"{TMMLU_BASE_URL}/{revision}/data/{urllib.request.quote(subject)}_{split}.csv"
        print(f"  下載 {target.name}")
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = resp.read()
        target.write_bytes(data)


def load_tmmluplus(src: Path, split: str = "test") -> tuple[list[MCQuestion], list[str]]:
    questions: list[MCQuestion] = []
    skipped: list[str] = []
    missing = [s for s in TMMLU_SUBJECTS if not (src / f"{s}_{split}.csv").exists()]
    if len(missing) == len(TMMLU_SUBJECTS):
        raise FileNotFoundError(f"{src} 底下找不到任何 *_{split}.csv，請確認路徑或加上 --download")
    for name in missing:
        skipped.append(f"{name}: 找不到 {name}_{split}.csv，整科略過")
    for subject in TMMLU_SUBJECTS:
        path = src / f"{subject}_{split}.csv"
        if path.exists():
            qs, sk = read_mc_csv(path, subject=subject, id_prefix=f"tmmlu_{subject}")
            questions += qs
            skipped += sk
    return questions, skipped


def tmmlu_group(subject: str) -> str:
    return TMMLU_GROUPS[TMMLU_SUBJECTS[subject][1]]


def stratified_sample(items: list, key, n: int, seed: int = 42) -> list:
    """依 key 分層，按比例配置抽 n 筆（最大餘數法），維持原始順序回傳。"""
    if n >= len(items):
        return list(items)
    strata: dict[str, list[int]] = defaultdict(list)
    for i, item in enumerate(items):
        strata[key(item)].append(i)

    quotas = {k: n * len(idx) / len(items) for k, idx in strata.items()}
    alloc = {k: int(q) for k, q in quotas.items()}
    # 依小數部分由大到小補足剩下的名額；同分時用分層名稱排序，確保結果可重現
    for k in sorted(quotas, key=lambda k: (-(quotas[k] - alloc[k]), k))[: n - sum(alloc.values())]:
        alloc[k] += 1

    rng = random.Random(seed)
    chosen: list[int] = []
    for k in sorted(strata):
        chosen += rng.sample(strata[k], alloc[k])
    return [items[i] for i in sorted(chosen)]


def to_task(q: MCQuestion, category: str) -> dict:
    letters = "".join(q.options)
    task = {
        "id": q.id,
        "category": category,
        "prompt": MC_PROMPT.format(
            letters="/".join(letters),
            question=q.question,
            options="\n".join(f"{k}. {v}" for k, v in q.options.items()),
        ),
        "grading": {
            "type": "multiple_choice",
            "expected": q.answers[0] if len(q.answers) == 1 else q.answers,
        },
    }
    if letters != "ABCD":
        task["grading"]["choices"] = letters
    return task


def rubric_tasks(benchmark_file: dict) -> list[dict]:
    """把「benchmark + questions」格式的評分準則題組轉成 rubric 任務。

    題組層級的設定（prompt 模板、及格門檻、評分說明、分類權重）攤平到每一題，
    讓產生的任務檔自給自足，不必再回頭讀原始題組。
    """
    meta = benchmark_file.get("benchmark", {})
    grading_meta = meta.get("grading", {})
    template = grading_meta.get("question_prompt_template", "{question}")
    if "{question}" not in template:
        raise ValueError("question_prompt_template 必須包含 {question}")
    weights = meta.get("category_weights_percent", {})
    threshold = grading_meta.get("pass_threshold_percent", 85) / 100
    guidance = grading_meta.get("evaluation_guidance")

    tasks = []
    for q in benchmark_file["questions"]:
        if not q.get("scoring"):
            raise ValueError(f"{q['id']} 缺少 scoring（各評分面向的上限）")
        grading = {
            "type": "rubric",
            "expected_points": q.get("expected_points", []),
            "critical_failures": q.get("critical_failures", []),
            "scoring": q["scoring"],
            "pass_threshold": threshold,
        }
        if guidance:
            grading["guidance"] = guidance
        task = {
            "id": q["id"],
            "category": q.get("category_zh") or q["category"],
            "prompt": template.replace("{question}", q["prompt"].strip()),
            "grading": grading,
            "high_risk": bool(q.get("high_risk", False)),
        }
        if q["category"] in weights:
            task["category_weight"] = weights[q["category"]]
        tasks.append(task)
    return tasks


class _LiteralDumper(yaml.SafeDumper):
    pass


def _str_presenter(dumper: yaml.SafeDumper, value: str):
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_LiteralDumper.add_representer(str, _str_presenter)


def write_tasks_yaml(tasks: list[dict], path: Path, header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.dump(tasks, Dumper=_LiteralDumper, allow_unicode=True, sort_keys=False, width=1000)
    comment = "".join(f"# {line}\n" if line else "#\n" for line in header.splitlines())
    path.write_text(comment + "\n" + body, encoding="utf-8")
