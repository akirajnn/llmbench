"""把考選部（MOEX）公告的試題／答案 PDF 轉成 import-exam 用的 CSV。

PDF 沒有結構化的題目資訊，只能依版面規則切分，因此設計成「先轉 CSV 給人
檢查，再匯入」兩段式，而不是直接產生任務檔。支援兩種版面：

- 選項用文字標記 "A."、題號為 "1." 開頭（例如醫師分階段考試）。
- 選項用字型私有區字元當圈選符號、題號單獨一行（例如公務人員初等考試）。

PDF 抽出的純文字會遺失圖片、表格框線與底線（克漏字空格、字詞底線），
這類題目的文字不完整，一律在 skip 欄標記原因，由人工決定是否保留。
"""

from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

# 初等考試等試卷用私有區字元畫圈選符號 Ⓐ–Ⓔ
PUA_OPTIONS = {"": "A", "": "B", "": "C", "": "D", "": "E"}

_NOISE = re.compile(r"^(代\s*號：|頁次：)")
_SECTION = re.compile(r"^[一二三四五六七八九十]+、|^說明：")
_GROUP_RANGE = re.compile(r"第\s*(\d+)\s*題?\s*(?:至|到|-|－|~|～)\s*第?\s*(\d+)\s*題")
_FIGURE_WORDS = re.compile(r"下圖|如圖|附圖|上圖|圖中|圖示|下表|如表|附表|上表|表中")
_DOTTED_NUMBER = re.compile(r"^(\d{1,3})\.\s*(.*)$", re.S)
_DOTTED_OPTION = re.compile(r"^([A-E])\.\s*(.*)$", re.S)

# 選項換行時，續行會對齊在圈選符號之後；比選項起點縮排不到這個距離的行
# 不是選項續行（通常是題組前導文字或下一個大題的說明）
_CONTINUATION_INDENT = 6
# 題幹中比第一行縮排超過這個距離的行，幾乎都是表格儲存格
_TABLE_INDENT = 14
# 頁首區（考試名稱、代號、頁次）的分隔線不算題目內容
_HEADER_BOTTOM = 80


@dataclass
class _Line:
    page: int
    x0: float
    x1: float
    y0: float
    y1: float
    text: str


@dataclass
class ExamQuestion:
    number: int
    stem: str = ""
    options: dict[str, str] = field(default_factory=dict)
    skip_reasons: list[str] = field(default_factory=list)
    # (page, y0, y1) 片段，用來判斷題目範圍內是否有圖片或線條
    spans: list[tuple[int, float, float]] = field(default_factory=list)
    stem_x: float | None = None
    option_x: float | None = None


# Symbol 字型把標點與數字放在私有區 U+F020–U+F040（例如 U+F02D 為連字號），
# 這段與 ASCII 一一對應；更後面的是希臘字母等，無法安全對應，留給 _PUA 標記
_SYMBOL_FONT = {cp: chr(cp - 0xF000) for cp in range(0xF020, 0xF041)}
_PUA = re.compile(r"[-]")


def _nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text.translate(_SYMBOL_FONT))


def _join(left: str, right: str) -> str:
    left, right = left.rstrip(), right.strip()
    if not left:
        return right
    # 英文單字跨行時補回空白；中文跨行直接相連
    if left[-1].isascii() and left[-1].isalnum() and right[:1].isascii() and right[:1].isalnum():
        return f"{left} {right}"
    return left + right


def _read_lines(doc) -> list[_Line]:
    lines = []
    for page_no, page in enumerate(doc):
        page_lines = []
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                text = "".join(span["text"] for span in line["spans"])
                if text.strip():
                    x0, y0, x1, y1 = line["bbox"]
                    page_lines.append(_Line(page_no, x0, x1, y0, y1, text))
        for row in _group_rows(page_lines, key=lambda ln: ln.y0):
            lines += _merge_row(sorted(row, key=lambda ln: ln.x0))
    return lines


def _merge_row(pieces: list[_Line]) -> list[_Line]:
    """把同一列被切成多段的文字接回一行。

    有些 PDF 把一行拆成多個文字區塊，後段起點還會和前段重疊一個字
    （例如 "trocha" + "anter"），合併時要去掉重疊的重複字元。
    兩欄選項（右欄以圈選符號開頭）與單獨一行的題號不合併。
    """
    merged: list[_Line] = []
    for piece in pieces:
        prev = merged[-1] if merged else None
        text = piece.text
        if prev is None or text.lstrip()[:1] in PUA_OPTIONS or prev.text.strip().isdigit():
            merged.append(piece)
            continue
        head = prev.text.rstrip()
        if piece.x0 < prev.x1:
            for k in (3, 2, 1):
                if len(text) > k and head.endswith(text[:k]):
                    text = text[k:]
                    break
            prev.text = head + text
        else:
            prev.text = _join(head, text) if piece.x0 - prev.x1 > 3 else head + text
        prev.x1 = max(prev.x1, piece.x1)
        prev.y1 = max(prev.y1, piece.y1)
    return merged


def _group_rows(items: list, key, tolerance: float = 3.0) -> list[list]:
    """依 y 座標分列：與該列第一個元素相差 tolerance 以內視為同一列。"""
    rows: list[list] = []
    for item in sorted(items, key=key):
        if rows and key(item) - key(rows[-1][0]) <= tolerance:
            rows[-1].append(item)
        else:
            rows.append([item])
    return rows


def _option_marker(text: str, dotted: bool) -> tuple[str, str] | None:
    stripped = text.lstrip()
    if dotted:
        m = _DOTTED_OPTION.match(stripped)
        return (m.group(1), m.group(2)) if m else None
    if stripped[:1] in PUA_OPTIONS:
        return PUA_OPTIONS[stripped[0]], stripped[1:]
    return None


def _question_start(line: _Line, expected: int, dotted: bool) -> str | None:
    """若此行是第 expected 題的開頭，回傳題號後的文字（可能為空字串）。

    只接受「下一個應出現的題號」，避免把表格裡的數字或題幹中的編號誤判為新題。
    """
    stripped = line.text.strip()
    if dotted:
        m = _DOTTED_NUMBER.match(stripped)
        return m.group(2) if m and int(m.group(1)) == expected else None
    return "" if stripped == str(expected) and line.x0 < 70 else None


def parse_questions(pdf_path: Path) -> tuple[list[ExamQuestion], str]:
    """回傳 (題目, 科目名稱)。"""
    import fitz  # 延遲載入：只有轉 PDF 時才需要 PyMuPDF

    doc = fitz.open(pdf_path)
    lines = _read_lines(doc)
    dotted = any(_DOTTED_NUMBER.match(ln.text.strip()) for ln in lines[:60])
    subject = _subject_name("\n".join(ln.text for ln in lines[:30]))

    questions: list[ExamQuestion] = []
    current: ExamQuestion | None = None
    current_option: str | None = None
    preamble: list[str] = []  # 題組前導文字（例如「請依下文回答第 46 題至第 50 題」）
    multi_select = False
    in_section_header = False  # 大題標題與說明（可能換行）到下一題之前都不收

    for line in lines:
        text = line.text
        stripped = text.strip()
        if _NOISE.match(stripped):
            continue
        if _SECTION.match(stripped):
            if "複選" in stripped:
                multi_select = True
            elif "單選" in stripped:
                multi_select = False
            current = None
            current_option = None
            in_section_header = True
            continue

        expected = (questions[-1].number + 1) if questions else 1
        start = _question_start(line, expected, dotted)
        if start is not None:
            in_section_header = False
            current = ExamQuestion(number=expected, stem=start.strip())
            current.spans.append((line.page, line.y0, line.y1))
            if multi_select:
                current.skip_reasons.append("複選題")
            if preamble:
                passage = ""
                for piece in preamble:
                    passage = _join(passage, piece)
                current.stem = f"{passage}\n{current.stem}" if current.stem else passage
                current.skip_reasons.append("題組前導文字，請確認是否正確併入")
                preamble = _carry_preamble(preamble, expected)
            questions.append(current)
            current_option = None
            continue

        if current is None:
            if questions and not in_section_header:  # 第一題之前是封面資訊，不收
                preamble.append(stripped)
            continue

        marker = _option_marker(text, dotted)
        expected_letter = chr(ord("A") + len(current.options))
        if marker and marker[0] == expected_letter:
            letter, body = marker
            current.options[letter] = body.strip()
            current_option = letter
            if current.option_x is None:
                current.option_x = line.x0
            current.spans.append((line.page, line.y0, line.y1))
            continue

        if current_option is not None:
            if current.option_x is not None and line.x0 >= current.option_x + _CONTINUATION_INDENT:
                current.options[current_option] = _join(current.options[current_option], text)
                current.spans.append((line.page, line.y0, line.y1))
            else:
                # 選項結束後出現未縮排的文字：視為下一組題目的前導文字
                current = None
                current_option = None
                preamble.append(stripped)
            continue

        if current.stem_x is None:
            current.stem_x = line.x0
        elif line.x0 > current.stem_x + _TABLE_INDENT and "疑似表格" not in current.skip_reasons:
            current.skip_reasons.append("疑似表格")
        current.stem = _join(current.stem, text)
        current.spans.append((line.page, line.y0, line.y1))

    _flag_graphics(doc, questions)
    for q in questions:
        if len(q.options) not in (4, 5):
            q.skip_reasons.append(f"選項數異常（{len(q.options)}）")
        if _FIGURE_WORDS.search(q.stem):
            q.skip_reasons.append("題幹提及圖表")
        # 題目保留原卷的全形標點，只轉換 Symbol 字型的私有區字元
        q.stem = q.stem.translate(_SYMBOL_FONT).strip()
        q.options = {k: v.translate(_SYMBOL_FONT).strip() for k, v in q.options.items()}
        if any(_PUA.search(t) for t in [q.stem, *q.options.values()]):
            q.skip_reasons.append("含無法辨識的特殊符號")
        q.skip_reasons = list(dict.fromkeys(q.skip_reasons))
    return questions, subject


def _carry_preamble(preamble: list[str], number: int) -> list[str]:
    """題組前導文字若註明「第 a 題至第 b 題」，要一路併入到第 b 題。"""
    m = _GROUP_RANGE.search(_nfkc("".join(preamble)))
    if m and int(m.group(1)) <= number < int(m.group(2)):
        return preamble
    return []


def _flag_graphics(doc, questions: list[ExamQuestion]) -> None:
    """題目範圍內有圖片或向量線條（表格框線、底線）時標記剔除。"""
    marks: dict[int, list[tuple[float, float, str]]] = {}
    for page_no, page in enumerate(doc):
        for info in page.get_image_info():
            y0, y1 = info["bbox"][1], info["bbox"][3]
            marks.setdefault(page_no, []).append((y0, y1, "含圖片"))
        for drawing in page.get_drawings():
            rect = drawing["rect"]
            if rect.y1 > _HEADER_BOTTOM:
                marks.setdefault(page_no, []).append((rect.y0, rect.y1, "含表格線或底線（純文字會遺失）"))

    for q in questions:
        for page_no, y0, y1 in q.spans:
            for my0, my1, reason in marks.get(page_no, []):
                if my0 <= y1 + 2 and my1 >= y0 - 2 and reason not in q.skip_reasons:
                    q.skip_reasons.append(reason)


def _subject_name(header: str) -> str:
    header = _nfkc(header)
    m = re.search(r"科目名稱:\s*(.+?)(?:\(包括|\(試題|\n|$)", header) or re.search(r"目:\s*(\S+)", header)
    return m.group(1).strip() if m else ""


def parse_answers(pdf_path: Path) -> dict[int, list[str] | None]:
    """解析標準答案（或更正答案）PDF。

    回傳 {題號: 可接受的答案字母}；None 代表一律給分。
    複選題的答案（如 "BCDE"）以單一字串元素表示，例如 ["BCDE"]。
    """
    import fitz

    page = fitz.open(pdf_path)[0]
    words = [(y0, x0, _nfkc(word)) for x0, y0, _, _, word, *_ in page.get_text("words")]
    ordered = [sorted((x, w) for _, x, w in row) for row in _group_rows(words, key=lambda w: w[0])]

    answers: dict[int, list[str] | None] = {}
    corrected: list[int] = []
    next_number = 1
    header: list[tuple[float, int]] = []
    for row in ordered:
        first = row[0][1]
        if first.startswith(("題號", "題序")):
            header = [(x, int(m.group(1))) for x, w in row if (m := re.fullmatch(r"第(\d+)題", w))]
            continue
        if not first.startswith("答案"):
            continue
        cells = [(x, w.removeprefix("答案")) for x, w in row]
        cells = [(x, w) for x, w in cells if w]
        for x, value in cells:
            if header:
                number = min(header, key=lambda h: abs(h[0] - x))[1]
            else:
                number = next_number
            next_number = number + 1
            if value == "#":
                corrected.append(number)
            else:
                answers[number] = [value]
        header = []

    notes = _nfkc(page.get_text()).split("備")[-1]
    resolved = _apply_notes(notes, answers)
    unresolved = [n for n in corrected if n not in resolved]
    if unresolved:
        raise ValueError(f"{pdf_path.name}: 第 {unresolved} 題標記更正，但備註無法自動解析，請人工處理：{notes.strip()!r}")
    return answers


def _apply_notes(notes: str, answers: dict[int, list[str] | None]) -> set[int]:
    resolved: set[int] = set()
    for m in re.finditer(r"第([\d、,]+)題(?:答案)?一律給分", notes):
        for n in re.split(r"[、,]", m.group(1)):
            answers[int(n)] = None
            resolved.add(int(n))
    for m in re.finditer(r"第(\d+)題(?:答案)?(?:答|更正為)([A-E或、,及和]+?)(?:者均給分|者給分|$|,|，|。)", notes):
        tokens = [t for t in re.split(r"[或、,及和]", m.group(2)) if t]
        singles = sorted({t for t in tokens if len(t) == 1})
        answers[int(m.group(1))] = singles or tokens
        resolved.add(int(m.group(1)))
    return resolved


def build_rows(
    questions: list[ExamQuestion], answers: dict[int, list[str] | None], exam: str, paper: str, subject: str
) -> list[dict]:
    year = exam[:3] if exam[:3].isdigit() else ""
    rows = []
    for q in questions:
        reasons = list(q.skip_reasons)
        if q.number not in answers:
            reasons.append("答案表找不到此題")
            answer = ""
        elif answers[q.number] is None:
            answer = "一律給分"
        else:
            accepted = answers[q.number]
            if len(accepted) == 1 and len(accepted[0]) > 1:
                if "複選題" not in reasons:
                    reasons.append("複選題")
            answer = ",".join(accepted)
        row = {
            "id": f"{exam}_{paper}_{q.number:03d}",
            "year": year,
            "subject": subject,
            "question": q.stem,
            **{letter: q.options.get(letter, "") for letter in "ABCD"},
            "answer": answer,
            "skip": "；".join(reasons),
        }
        if "E" in q.options:
            row["E"] = q.options["E"]
        rows.append(row)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "year", "subject", "question", "A", "B", "C", "D", "answer", "skip"]
    if any("E" in r for r in rows):
        fields.insert(8, "E")
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def find_papers(pdf_dir: Path) -> list[tuple[str, str, Path, Path]]:
    """找出 <考試代碼>_<試題代號>.pdf 與對應答案；有更正答案（MOD）時優先使用。"""
    papers = []
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        m = re.fullmatch(r"(\d+)_(\d+)", pdf.stem)
        if not m:
            continue
        exam, paper = m.groups()
        answer = next(
            (p for p in (pdf_dir / f"{exam}_MOD{paper}.pdf", pdf_dir / f"{exam}_ANS{paper}.pdf") if p.exists()),
            None,
        )
        if answer is None:
            raise FileNotFoundError(f"{pdf.name} 找不到對應的答案檔（{exam}_ANS{paper}.pdf 或 {exam}_MOD{paper}.pdf）")
        papers.append((exam, paper, pdf, answer))
    return papers
