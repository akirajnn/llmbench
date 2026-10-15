import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from llmbench.grading import grade
from llmbench.importers import (
    TMMLU_SUBJECTS,
    load_tmmluplus,
    parse_answer,
    read_mc_csv,
    stratified_sample,
    tmmlu_group,
    to_task,
    write_tasks_yaml,
)
from llmbench.providers import ChatResult, ModelSpec
from llmbench.report import paired_rows
from llmbench.runner import run_all
from llmbench.stats import mcnemar_exact, paired_diff_interval


class TestMcNemar(unittest.TestCase):
    def test_no_discordant_pairs(self):
        self.assertEqual(mcnemar_exact(0, 0), 1.0)

    def test_one_sided_extreme(self):
        # 10 題不一致全部偏向同一邊：p = 2 * 0.5^10
        self.assertAlmostEqual(mcnemar_exact(10, 0), 2 / 1024)

    def test_symmetric(self):
        self.assertEqual(mcnemar_exact(3, 9), mcnemar_exact(9, 3))

    def test_balanced_is_one(self):
        self.assertEqual(mcnemar_exact(5, 5), 1.0)

    def test_paired_diff_interval(self):
        diff, lo, hi = paired_diff_interval(b=5, c=15, n=100)
        self.assertAlmostEqual(diff, -0.1)
        self.assertLess(lo, -0.1)
        self.assertGreater(hi, -0.1)


class TestPairedRows(unittest.TestCase):
    def test_counts_discordant_pairs(self):
        tasks = [
            {"id": f"t{i}", "category": "med", "prompt": str(i), "grading": {"type": "contains", "expected": "ok"}}
            for i in range(4)
        ]
        # cloud 全對；local 對 t0、t1，錯 t2、t3
        def fake_chat(spec, prompt, timeout, think, temperature):
            ok = spec.name == "cloud" or prompt in ("0", "1")
            return ChatResult("ok" if ok else "no", 1, 1.0)

        with mock.patch("llmbench.runner.chat", fake_chat):
            results = run_all([ModelSpec(name="local"), ModelSpec(name="cloud")], tasks, repeat=2)
        rows = {r["category"]: r for r in paired_rows(results, "cloud")}
        self.assertEqual(set(rows), {"med", "ALL"})
        med = rows["med"]
        # repeat=2 只取第一次抽樣，n 不會被灌成 8
        self.assertEqual((med["n"], med["model_wins"], med["baseline_wins"]), (4, 0, 2))
        self.assertAlmostEqual(med["diff"], -50.0)
        self.assertAlmostEqual(med["p_value"], 0.5)


class TestMultiAnswerGrading(unittest.TestCase):
    def test_list_expected_accepts_any(self):
        grading = {"type": "multiple_choice", "expected": ["A", "C"]}
        self.assertTrue(grade("C", grading).passed)
        self.assertTrue(grade("A", grading).passed)
        self.assertFalse(grade("B", grading).passed)


class TestParseAnswer(unittest.TestCase):
    def test_single(self):
        self.assertEqual(parse_answer(" c ", "ABCD"), ["C"])

    def test_multiple(self):
        for raw in ("A,C", "C、A", "A或C", "AC", "A/C"):
            self.assertEqual(parse_answer(raw, "ABCD"), ["A", "C"], raw)

    def test_free_point(self):
        self.assertIsNone(parse_answer("一律給分", "ABCD"))
        self.assertIsNone(parse_answer("*", "ABCD"))
        self.assertIsNone(parse_answer("ABCD", "ABCD"))

    def test_invalid(self):
        with self.assertRaises(ValueError):
            parse_answer("E", "ABCD")


class TestReadCsvAndTask(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_exam_csv(self):
        path = self.dir / "exam.csv"
        path.write_text(
            "﻿id,year,subject,question,A,B,C,D,answer,skip\n"
            "q1,113,內科,題目一,a1,b1,c1,d1,B,\n"
            "q2,113,內科,題目二,a2,b2,c2,d2,A、C,\n"
            "q3,113,外科,題目三,a3,b3,c3,d3,一律給分,\n"
            "q4,113,外科,看圖題,a4,b4,c4,d4,D,看圖\n"
            "q5,113,外科,題目五,a5,,c5,d5,D,\n",
            encoding="utf-8",
        )
        questions, skipped = read_mc_csv(path)
        self.assertEqual([q.id for q in questions], ["q1", "q2"])
        self.assertEqual(questions[1].answers, ["A", "C"])
        self.assertEqual(len(skipped), 3)

        task = to_task(questions[1], "醫療專業")
        self.assertEqual(task["grading"], {"type": "multiple_choice", "expected": ["A", "C"]})
        self.assertIn("A. a2\nB. b2", task["prompt"])
        self.assertTrue(grade("答案是C", task["grading"]).passed)

    def test_exam_csv_with_optional_e_column(self):
        path = self.dir / "exam.csv"
        path.write_text(
            "id,question,A,B,C,D,E,answer\n"
            "q1,四選一,a,b,c,d,,C\n"
            "q2,五選一,a,b,c,d,e,E\n"
            "q3,四選項卻答E,a,b,c,d,,E\n",
            encoding="utf-8",
        )
        questions, skipped = read_mc_csv(path)
        self.assertEqual([list(q.options) for q in questions], [list("ABCD"), list("ABCDE")])
        self.assertEqual(to_task(questions[0], "x")["grading"], {"type": "multiple_choice", "expected": "C"})
        self.assertEqual(len(skipped), 1)

    def test_yaml_roundtrip(self):
        questions, _ = read_mc_csv(self._write_tmmlu("pharmacology", 3))
        out = self.dir / "out.yaml"
        tasks = [to_task(q, "x") for q in questions]
        write_tasks_yaml(tasks, out, "header line")
        self.assertTrue(out.read_text(encoding="utf-8").startswith("# header line\n"))
        self.assertEqual(yaml.safe_load(out.read_text(encoding="utf-8")), tasks)

    def test_load_tmmluplus(self):
        self._write_tmmlu("pharmacology", 3)
        self._write_tmmlu("physics", 2)
        questions, skipped = load_tmmluplus(self.dir)
        self.assertEqual(len(questions), 5)
        self.assertIn("tmmlu_pharmacology_0001", {q.id for q in questions})
        self.assertEqual(len(skipped), len(TMMLU_SUBJECTS) - 2)
        self.assertEqual(tmmlu_group("pharmacology"), "other")
        self.assertEqual(tmmlu_group("physics"), "stem")

    def test_load_tmmluplus_empty_dir(self):
        with self.assertRaises(FileNotFoundError):
            load_tmmluplus(self.dir)

    def _write_tmmlu(self, subject: str, n: int) -> Path:
        path = self.dir / f"{subject}_test.csv"
        rows = "".join(f"{subject} 題目{i},a,b,c,d,A\n" for i in range(n))
        path.write_text("question,A,B,C,D,answer\n" + rows, encoding="utf-8")
        return path


class TestStratifiedSample(unittest.TestCase):
    items = [("big", i) for i in range(60)] + [("mid", i) for i in range(30)] + [("small", i) for i in range(10)]

    def test_proportional_allocation(self):
        sampled = stratified_sample(self.items, key=lambda x: x[0], n=20)
        counts = {k: sum(1 for s in sampled if s[0] == k) for k in ("big", "mid", "small")}
        self.assertEqual(counts, {"big": 12, "mid": 6, "small": 2})

    def test_largest_remainder_fills_exact_n(self):
        self.assertEqual(len(stratified_sample(self.items, key=lambda x: x[0], n=7)), 7)

    def test_deterministic_and_order_preserving(self):
        a = stratified_sample(self.items, key=lambda x: x[0], n=15, seed=1)
        self.assertEqual(a, stratified_sample(self.items, key=lambda x: x[0], n=15, seed=1))
        self.assertEqual(a, sorted(a, key=self.items.index))

    def test_n_larger_than_population(self):
        self.assertEqual(stratified_sample(self.items, key=lambda x: x[0], n=999), self.items)


if __name__ == "__main__":
    unittest.main()
