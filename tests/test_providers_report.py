import unittest
from unittest import mock

from llmbench import providers
from llmbench.grading import grade_multiple_choice
from llmbench.providers import ChatResult, ModelSpec
from llmbench.report import category_score_rows, format_category_table, relative_rows
from llmbench.runner import TaskResult, run_all


class TestModelSpec(unittest.TestCase):
    def test_string_is_ollama(self):
        spec = ModelSpec.parse("qwen2.5:7b")
        self.assertEqual((spec.name, spec.provider, spec.model), ("qwen2.5:7b", "ollama", "qwen2.5:7b"))

    def test_dict_cloud(self):
        spec = ModelSpec.parse({"name": "c", "provider": "anthropic", "model": "m"})
        self.assertEqual((spec.name, spec.provider), ("c", "anthropic"))

    def test_unknown_provider(self):
        with self.assertRaises(ValueError):
            ModelSpec.parse({"provider": "nope", "model": "m"})

    def test_missing_api_key(self):
        spec = ModelSpec.parse({"provider": "openai", "model": "m", "api_key_env": "LLMBENCH_NO_SUCH_KEY"})
        with self.assertRaises(RuntimeError):
            providers.chat(spec, "hi", timeout=5, think=None, temperature=0)


class TestMultipleChoiceChinese(unittest.TestCase):
    def test_chinese_prefix(self):
        self.assertTrue(grade_multiple_choice("答案是C", "C").passed)

    def test_english_article_not_choice(self):
        self.assertEqual(grade_multiple_choice("I think a good answer is C", "C").detail, "expected=C picked=C")


class TestRelative(unittest.TestCase):
    def test_relative_to_baseline(self):
        tasks = [
            {"id": "t1", "category": "math", "prompt": "1", "grading": {"type": "contains", "expected": "ok"}},
            {"id": "t2", "category": "math", "prompt": "2", "grading": {"type": "contains", "expected": "ok"}},
        ]
        answers = {("cloud", "1"): "ok", ("cloud", "2"): "ok", ("local", "1"): "ok", ("local", "2"): "no"}

        def fake_chat(spec, prompt, timeout, think, temperature):
            return ChatResult(answers[(spec.name, prompt)], 1, 1.0)

        with mock.patch("llmbench.runner.chat", fake_chat):
            results = run_all([ModelSpec(name="local"), ModelSpec(name="cloud")], tasks)
        categories, rows = relative_rows(results, "cloud")
        self.assertEqual(categories, ["math", "ALL"])
        self.assertEqual(dict(rows)["local"], ["50% → 50%", "50% → 50%"])
        self.assertEqual(dict(rows)["cloud"], ["100% (基準)", "100% (基準)"])


class TestCategoryScores(unittest.TestCase):
    def test_sorted_by_total_and_errors_excluded(self):
        def result(model, task_id, category, passed, error=None):
            return TaskResult(model, task_id, category, "", "", passed, float(passed), "", 0.0, 0, 0.0, error=error)

        results = [
            result("weak", "t1", "醫學", False),
            result("weak", "t2", "國文", True),
            result("strong", "t1", "醫學", True),
            result("strong", "t2", "國文", True),
            result("strong", "t3", "國文", False, error="Failed to connect"),
        ]
        columns, rows = category_score_rows(results)
        self.assertEqual(columns, ["醫學", "國文", "總分"])
        self.assertEqual([m for m, _, _ in rows], ["strong", "weak"])
        self.assertEqual(rows[0][1:], (["100.0% (1/1)", "100.0% (1/1)", "100.0% (2/2)"], 1))
        self.assertEqual(rows[1][1], ["0.0% (0/1)", "100.0% (1/1)", "50.0% (1/2)"])
        self.assertIn("strong", format_category_table(results))


if __name__ == "__main__":
    unittest.main()
