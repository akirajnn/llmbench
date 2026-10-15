import json
import unittest
from pathlib import Path
from unittest import mock

import yaml

from llmbench.grading import grade
from llmbench.importers import rubric_tasks
from llmbench.judge import JudgeError, build_judge_prompt, grade_rubric, parse_verdict
from llmbench.providers import ChatResult, ModelSpec
from llmbench.report import critical_failure_rows, rubric_summary
from llmbench.runner import run_all, run_task

SCORING = {"point_coverage": 3, "responsibility_and_procedure": 1, "uncertainty_and_safety": 1}
GRADING = {
    "type": "rubric",
    "expected_points": ["確認需求範圍", "釐清權責"],
    "critical_failures": ["無條件保證七天上線"],
    "scoring": SCORING,
    "pass_threshold": 0.85,
}
JUDGE = ModelSpec(name="judge", provider="openai", model="j")


def verdict(pc=3, rp=1, us=1, critical=False, reason=""):
    return json.dumps(
        {
            "point_coverage": pc,
            "responsibility_and_procedure": rp,
            "uncertainty_and_safety": us,
            "critical_failure": critical,
            "critical_failure_reason": reason,
            "rationale": "ok",
        },
        ensure_ascii=False,
    )


def fake_judge(*texts):
    replies = iter(texts)
    return mock.patch("llmbench.judge.chat", lambda *a, **k: ChatResult(next(replies), 1, 1.0))


class TestParseVerdict(unittest.TestCase):
    def test_accepts_code_fence(self):
        v = parse_verdict("```json\n" + verdict(2.5) + "\n```", SCORING)
        self.assertEqual(v["point_coverage"], 2.5)

    def test_rejects_out_of_range(self):
        with self.assertRaises(JudgeError):
            parse_verdict(verdict(pc=4), SCORING)

    def test_rejects_missing_dimension(self):
        with self.assertRaises(JudgeError):
            parse_verdict('{"point_coverage": 3, "critical_failure": false}', SCORING)

    def test_rejects_non_bool_critical(self):
        with self.assertRaises(JudgeError):
            parse_verdict(verdict().replace("false", '"no"'), SCORING)

    def test_rejects_non_json(self):
        with self.assertRaises(JudgeError):
            parse_verdict("我覺得滿分", SCORING)


class TestGradeRubric(unittest.TestCase):
    def test_full_score_passes(self):
        with fake_judge(verdict()):
            r = grade_rubric("題目", "回答", GRADING, JUDGE, 10)
        self.assertTrue(r.passed)
        self.assertEqual(r.score, 1.0)

    def test_below_threshold_fails_with_partial_score(self):
        with fake_judge(verdict(pc=2)):
            r = grade_rubric("題目", "回答", GRADING, JUDGE, 10)
        self.assertFalse(r.passed)
        self.assertAlmostEqual(r.score, 0.8)

    def test_critical_failure_zeroes_score(self):
        with fake_judge(verdict(critical=True, reason="保證七天上線")):
            r = grade_rubric("題目", "回答", GRADING, JUDGE, 10)
        self.assertEqual((r.passed, r.score, r.critical), (False, 0.0, True))
        self.assertIn("保證七天上線", r.detail)

    def test_retries_once_on_bad_format(self):
        with fake_judge("not json", verdict()):
            self.assertTrue(grade_rubric("題目", "回答", GRADING, JUDGE, 10).passed)

    def test_raises_after_two_bad_outputs(self):
        with fake_judge("not json", "still not json"), self.assertRaises(JudgeError):
            grade_rubric("題目", "回答", GRADING, JUDGE, 10)

    def test_empty_response_skips_judge(self):
        with mock.patch("llmbench.judge.chat") as chat:
            r = grade_rubric("題目", "  ", GRADING, JUDGE, 10)
        chat.assert_not_called()
        self.assertFalse(r.passed)

    def test_prompt_contains_rubric(self):
        prompt = build_judge_prompt("題目X", "回答Y", GRADING)
        for text in ("題目X", "回答Y", "確認需求範圍", "無條件保證七天上線", '"point_coverage": <0 到 3 的數字>'):
            self.assertIn(text, prompt)

    def test_grade_refuses_rubric(self):
        with self.assertRaises(ValueError):
            grade("x", GRADING)


class TestRunnerWithJudge(unittest.TestCase):
    task = {"id": "ADM-1", "category": "行政程序", "prompt": "q", "grading": GRADING, "high_risk": True}

    def test_rubric_without_judge_raises(self):
        with mock.patch("llmbench.runner.chat", return_value=ChatResult("a", 1, 1.0)):
            with self.assertRaises(ValueError):
                run_task(ModelSpec(name="m"), self.task)

    def test_judge_error_keeps_response(self):
        with mock.patch("llmbench.runner.chat", return_value=ChatResult("回答", 1, 1.0)), fake_judge("x", "y"):
            r = run_task(ModelSpec(name="m"), self.task, judge=JUDGE)
        self.assertEqual(r.response, "回答")
        self.assertTrue(r.error.startswith("judge:"))
        self.assertTrue(r.high_risk)


class TestRubricImportAndReport(unittest.TestCase):
    def test_import_real_benchmark_file(self):
        path = Path(__file__).resolve().parent.parent / "tw_hospital_admin_benchmark_v1.yaml"
        if not path.exists():
            self.skipTest("tw_hospital_admin_benchmark_v1.yaml 不存在")
        tasks = rubric_tasks(yaml.safe_load(path.read_text(encoding="utf-8")))
        self.assertEqual(len(tasks), 48)
        first = tasks[0]
        self.assertEqual((first["id"], first["category"], first["category_weight"]), ("ADM-01-001", "行政程序", 20))
        self.assertIn("題目：科室要求七天上線電子簽核", first["prompt"])
        self.assertNotIn("{question}", first["prompt"])
        self.assertEqual(first["grading"]["pass_threshold"], 0.85)

    def test_template_without_placeholder_rejected(self):
        bad = {"benchmark": {"grading": {"question_prompt_template": "no slot"}}, "questions": []}
        with self.assertRaises(ValueError):
            rubric_tasks(bad)

    def test_weighted_score_and_critical_list(self):
        tasks = [
            {"id": "a1", "category": "A", "prompt": "題目：a1", "grading": GRADING, "high_risk": False},
            {"id": "b1", "category": "B", "prompt": "題目：b1", "grading": GRADING, "high_risk": True},
        ]

        # 模型在 A 類滿分、B 類觸發重大錯誤
        def judge_chat(spec, prompt, **kwargs):
            return ChatResult(verdict() if "題目：a1" in prompt else verdict(critical=True, reason="r"), 1, 1.0)

        def model_chat(spec, prompt, **kwargs):
            return ChatResult(f"answer to {prompt}", 1, 1.0)

        with mock.patch("llmbench.runner.chat", model_chat), mock.patch("llmbench.judge.chat", judge_chat):
            results = run_all([ModelSpec(name="m")], tasks, judge=JUDGE)

        s = rubric_summary(results, {"A": 75, "B": 25})["m"]
        self.assertAlmostEqual(s["weighted_score"], 75.0)
        self.assertEqual((s["critical"], s["high_risk_critical"]), (1, 1))
        self.assertEqual([r.task_id for r in critical_failure_rows(results)], ["b1"])


if __name__ == "__main__":
    unittest.main()
