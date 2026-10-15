import tempfile
import unittest
import unicodedata
from pathlib import Path

from llmbench.moex import ExamQuestion, _apply_notes, _join, _Line, _merge_row, build_rows

try:
    import fitz
except ImportError:  # PyMuPDF 只有 moex-to-csv 需要
    fitz = None

TESTPDF = Path(__file__).resolve().parent.parent / "testpdf"


def line(x0, x1, text, y=100.0):
    return _Line(page=0, x0=x0, x1=x1, y0=y, y1=y + 10, text=text)


class TestTextAssembly(unittest.TestCase):
    def test_merge_row_drops_overlapping_char(self):
        merged = _merge_row([line(53, 244, "30.大轉子（greater trocha"), line(239, 445, "anter）斷裂")])
        self.assertEqual([m.text for m in merged], ["30.大轉子（greater trochanter）斷裂"])

    def test_merge_row_keeps_two_column_options(self):
        merged = _merge_row([line(60, 200, "出口大於進口"), line(313, 400, "出口小於進口")])
        self.assertEqual(len(merged), 2)

    def test_merge_row_keeps_question_number_alone(self):
        merged = _merge_row([line(41, 47, "12"), line(60, 400, "下列何者正確？")])
        self.assertEqual([m.text for m in merged], ["12", "下列何者正確？"])

    def test_join_adds_space_only_between_ascii_words(self):
        self.assertEqual(_join("solitary ", "nucleus"), "solitary nucleus")
        self.assertEqual(_join("同時對氟喹諾酮類藥物", "（fluoroquinolones）"), "同時對氟喹諾酮類藥物（fluoroquinolones）")


class TestAnswers(unittest.TestCase):
    def test_correction_notes(self):
        notes = unicodedata.normalize("NFKC", "第66題答Ａ或Ｄ或AD者均給分，第95題一律給分")
        answers = {66: ["#"], 95: ["#"]}
        self.assertEqual(_apply_notes(notes, answers), {66, 95})
        self.assertEqual(answers, {66: ["A", "D"], 95: None})

    def test_build_rows_flags_multi_select_and_missing_answer(self):
        qs = [
            ExamQuestion(1, "題一", {k: k.lower() for k in "ABCDE"}),
            ExamQuestion(2, "題二", {k: k.lower() for k in "ABCD"}),
        ]
        rows = build_rows(qs, {1: ["BCDE"]}, "115010", "1501", "國文")
        self.assertEqual(rows[0]["answer"], "BCDE")
        self.assertIn("複選題", rows[0]["skip"])
        self.assertEqual(rows[0]["E"], "e")
        self.assertIn("答案表找不到此題", rows[1]["skip"])
        self.assertEqual((rows[1]["id"], rows[1]["year"]), ("115010_1501_002", "115"))


@unittest.skipIf(fitz is None, "PyMuPDF 未安裝")
class TestSyntheticPdf(unittest.TestCase):
    """用 PyMuPDF 產生「1. / A.」版面的小型試卷與答案表，跑完整解析流程。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, name: str, items: list[tuple[float, float, str]], lines: list[tuple] = ()) -> Path:
        doc = fitz.open()
        page = doc.new_page()
        for x, y, text in items:
            page.insert_text((x, y), text, fontname="china-t", fontsize=10)
        for p1, p2 in lines:
            page.draw_line(p1, p2)
        path = self.dir / name
        doc.save(path)
        return path

    def test_parse_paper_and_answers(self):
        from llmbench.moex import find_papers, parse_answers, parse_questions

        self._write(
            "115020_1301.pdf",
            [
                (57, 60, "科目名稱：醫學（一）（包括生物化學）"),
                (53, 100, "1.下列何者正確？"),
                (63, 120, "A.甲"), (63, 140, "B.乙"), (63, 160, "C.丙"),
                (63, 180, "D.很長的選項第一行"), (75, 195, "續行"),
                (53, 220, "2.依下表，何者錯誤？"),
                (63, 240, "A.一"), (63, 260, "B.二"), (63, 280, "C.三"), (63, 300, "D.四"),
            ],
            lines=[((63, 235), (300, 235))],
        )
        self._write(
            "115020_ANS1301.pdf",
            [(52, 100, "題號"), (52, 120, "答案Ｃ"), (108, 120, "＃"), (52, 300, "備註：第2題一律給分")],
        )
        [(exam, paper, pdf, ans)] = find_papers(self.dir)
        questions, subject = parse_questions(pdf)
        self.assertEqual(subject, "醫學(一)")
        self.assertEqual([q.number for q in questions], [1, 2])
        self.assertEqual(questions[0].options["D"], "很長的選項第一行續行")
        self.assertEqual(questions[0].skip_reasons, [])
        self.assertIn("題幹提及圖表", questions[1].skip_reasons)
        self.assertIn("含表格線或底線（純文字會遺失）", questions[1].skip_reasons)
        self.assertEqual(parse_answers(ans), {1: ["C"], 2: None})

    def test_unresolved_correction_raises(self):
        from llmbench.moex import parse_answers

        path = self._write("x_ANS1.pdf", [(52, 100, "題號"), (52, 120, "答案＃"), (52, 300, "備註：請見公告")])
        with self.assertRaises(ValueError):
            parse_answers(path)


@unittest.skipIf(fitz is None or not TESTPDF.exists(), "需要 PyMuPDF 與 testpdf/ 的真實考卷")
class TestRealPapers(unittest.TestCase):
    """回歸測試：115 年考選部試卷的題數、答案與剔除結果。"""

    def test_all_papers(self):
        from llmbench.moex import find_papers, parse_answers, parse_questions

        expected = {"1501": 45, "2501": 50, "3501": 50, "4501": 50, "1301": 100, "2301": 100}
        for exam, paper, pdf, ans in find_papers(TESTPDF):
            with self.subTest(paper=paper):
                questions, _ = parse_questions(pdf)
                answers = parse_answers(ans)
                self.assertEqual([q.number for q in questions], list(range(1, expected[paper] + 1)))
                self.assertEqual(set(answers), set(range(1, expected[paper] + 1)))
                for q in questions:
                    if not q.skip_reasons:
                        self.assertEqual(len(q.options), 4, f"{paper} #{q.number}")

    def test_medical_corrections_and_figures(self):
        from llmbench.moex import parse_answers, parse_questions

        answers = parse_answers(TESTPDF / "115020_MOD1301.pdf")
        self.assertEqual((answers[66], answers[95]), (["A", "D"], None))
        questions, _ = parse_questions(TESTPDF / "115020_1301.pdf")
        flagged = {q.number for q in questions if q.skip_reasons}
        self.assertEqual(flagged, {31, 62})
        self.assertIn("greater trochanter", questions[29].stem)


if __name__ == "__main__":
    unittest.main()
