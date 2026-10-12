import unittest

from llmbench.stats import pass_at_k, wilson_interval


class TestWilsonInterval(unittest.TestCase):
    def test_zero_trials(self):
        self.assertEqual(wilson_interval(0.0, 0), (0.0, 0.0))

    def test_perfect_score_interval_not_zero_width(self):
        # 100% 通過但樣本數少時，Wilson interval 不會錯誤地給出寬度為 0 的區間
        lo, hi = wilson_interval(1.0, 5)
        self.assertLess(lo, 1.0)
        self.assertEqual(hi, 1.0)

    def test_interval_contains_point_estimate(self):
        lo, hi = wilson_interval(0.5, 20)
        self.assertLessEqual(lo, 0.5)
        self.assertGreaterEqual(hi, 0.5)

    def test_more_samples_narrows_interval(self):
        lo_small, hi_small = wilson_interval(0.5, 10)
        lo_large, hi_large = wilson_interval(0.5, 1000)
        self.assertGreater(hi_small - lo_small, hi_large - lo_large)


class TestPassAtK(unittest.TestCase):
    def test_all_passed_is_one(self):
        self.assertEqual(pass_at_k(n=10, c=10, k=3), 1.0)

    def test_none_passed_is_zero(self):
        self.assertEqual(pass_at_k(n=10, c=0, k=3), 0.0)

    def test_k_equals_n_with_some_failures(self):
        # k == n 代表「把全部樣本都算進去」，只要 c > 0 必定是 1.0
        self.assertEqual(pass_at_k(n=5, c=1, k=5), 1.0)

    def test_matches_naive_combinatorial_formula(self):
        import math

        n, c, k = 10, 4, 3
        naive = 1 - math.comb(n - c, k) / math.comb(n, k)
        self.assertAlmostEqual(pass_at_k(n, c, k), naive, places=9)


if __name__ == "__main__":
    unittest.main()
