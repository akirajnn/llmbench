import unittest

from llmbench.grading import grade


class TestGrading(unittest.TestCase):
    def test_exact_match_pass(self):
        result = grade("12", {"type": "exact_match", "expected": "12"})
        self.assertTrue(result.passed)

    def test_exact_match_fail(self):
        result = grade("13", {"type": "exact_match", "expected": "12"})
        self.assertFalse(result.passed)

    def test_exact_match_case_insensitive(self):
        result = grade("Yes", {"type": "exact_match", "expected": "yes"})
        self.assertTrue(result.passed)

    def test_contains_pass(self):
        result = grade("台灣最高峰是玉山沒錯", {"type": "contains", "expected": "玉山"})
        self.assertTrue(result.passed)

    def test_contains_fail(self):
        result = grade("台灣最高峰是阿里山", {"type": "contains", "expected": "玉山"})
        self.assertFalse(result.passed)

    def test_regex_pass(self):
        result = grade("today is 2024-01-15", {"type": "regex", "expected": r"\d{4}-\d{2}-\d{2}"})
        self.assertTrue(result.passed)

    def test_regex_fail(self):
        result = grade("today is Jan 15", {"type": "regex", "expected": r"\d{4}-\d{2}-\d{2}"})
        self.assertFalse(result.passed)

    def test_numeric_tolerance_exact(self):
        result = grade("391", {"type": "numeric_tolerance", "expected": 391, "tolerance": 0})
        self.assertTrue(result.passed)

    def test_numeric_tolerance_within_range(self):
        result = grade("about 366 days", {"type": "numeric_tolerance", "expected": 365, "tolerance": 2})
        self.assertTrue(result.passed)

    def test_numeric_tolerance_out_of_range(self):
        result = grade("400", {"type": "numeric_tolerance", "expected": 391, "tolerance": 0})
        self.assertFalse(result.passed)

    def test_numeric_tolerance_picks_last_number_as_final_answer(self):
        result = grade("12 × 7 = 84", {"type": "numeric_tolerance", "expected": 84, "tolerance": 0})
        self.assertTrue(result.passed)

    def test_numeric_tolerance_no_number(self):
        result = grade("I don't know", {"type": "numeric_tolerance", "expected": 391, "tolerance": 0})
        self.assertFalse(result.passed)

    def test_contains_unicode_subscript_normalized(self):
        result = grade("H₂O", {"type": "contains", "expected": "H2O"})
        self.assertTrue(result.passed)

    def test_unknown_type_raises(self):
        with self.assertRaises(ValueError):
            grade("x", {"type": "nonexistent"})

    # --- multiple_choice (MMLU 風格) ---

    def test_multiple_choice_pass(self):
        result = grade("答案是 B", {"type": "multiple_choice", "expected": "B"})
        self.assertTrue(result.passed)

    def test_multiple_choice_fail(self):
        result = grade("答案是 C", {"type": "multiple_choice", "expected": "B"})
        self.assertFalse(result.passed)

    def test_multiple_choice_no_letter_found(self):
        result = grade("我不知道", {"type": "multiple_choice", "expected": "B"})
        self.assertFalse(result.passed)

    def test_multiple_choice_does_not_match_letter_inside_word(self):
        # "A" 不應該被 "Apple" 這種單字裡的字母誤判為選到 A
        result = grade("Apple 是答案", {"type": "multiple_choice", "expected": "A"})
        self.assertFalse(result.passed)

    # --- gsm8k（#### 分隔符）---

    def test_gsm8k_pass(self):
        result = grade(
            "小明有 3 顆蘋果，又買了 4 顆。\n#### 7",
            {"type": "gsm8k", "expected": 7},
        )
        self.assertTrue(result.passed)

    def test_gsm8k_missing_delimiter_fails(self):
        result = grade("答案是 7", {"type": "gsm8k", "expected": 7})
        self.assertFalse(result.passed)

    def test_gsm8k_wrong_answer_fails(self):
        result = grade("推理過程...\n#### 8", {"type": "gsm8k", "expected": 7})
        self.assertFalse(result.passed)

    # --- checks（IFEval 風格 strict/loose）---

    def test_checks_all_pass_strict_and_loose(self):
        result = grade(
            "2024-01-15",
            {
                "type": "checks",
                "checks": [
                    {"type": "regex", "expected": r"\d{4}-\d{2}-\d{2}"},
                    {"type": "contains", "expected": "2024"},
                ],
            },
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.score, 1.0)

    def test_checks_partial_pass_loose_score(self):
        result = grade(
            "2024-01-15",
            {
                "type": "checks",
                "checks": [
                    {"type": "regex", "expected": r"\d{4}-\d{2}-\d{2}"},
                    {"type": "contains", "expected": "不存在的字串"},
                ],
            },
        )
        self.assertFalse(result.passed)
        self.assertAlmostEqual(result.score, 0.5)


if __name__ == "__main__":
    unittest.main()
