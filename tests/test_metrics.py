"""用定义反例、单位变换和配对设计检查公开演示，不验证论文原始结论。"""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metrics import decompose, paired_tests, weighted_gower, bh_adjust


class MethodTests(unittest.TestCase):
    def test_nestedness_is_not_richness_difference(self):
        # 无共享分类单元且丰富度不同：Podani 丰富度差异不为零，Baselga 嵌套结果为零。
        x = [[1, 1, 0, 0, 0], [0, 0, 1, 1, 1]]
        total, repl, diff = decompose(x, coefficient="jaccard")
        self.assertAlmostEqual(total[0, 1], 1)
        self.assertAlmostEqual(diff[0, 1], .2)
        _, turn, nested = decompose(x, partition="baselga")
        self.assertAlmostEqual(turn[0, 1], 1)
        self.assertAlmostEqual(nested[0, 1], 0)

    def test_subset_and_scale_invariance(self):
        x = np.array([[1, 0, 0], [1, 1, 1]])
        _, turn, nested = decompose(x, partition="baselga")
        self.assertAlmostEqual(turn[0, 1], 0)
        self.assertAlmostEqual(nested[0, 1], .5)
        for a, b in zip(decompose(x), decompose(x * 1000)):
            np.testing.assert_allclose(a, b)

    def test_unknown_is_not_zero(self):
        d, audit, _ = weighted_gower([[0, np.nan], [1, 0], [np.nan, 1]], ["a", "b"])
        self.assertEqual(audit["undefined_pair_count"], 1)
        self.assertEqual(audit["fill_value"], 1)
        self.assertEqual(d[0, 1], 1)

    def test_empty_and_invalid_input_fail(self):
        for x in ([[0, 0], [1, 0]], [[-1, 2], [1, 0]], [[np.nan, 2], [1, 0]]):
            with self.assertRaises(ValueError):decompose(x)
        with self.assertRaises(ValueError):decompose([[2, 1], [1, 0]], partition="baselga")

    def test_paired_exact_swap_invariance(self):
        x = np.array([[0.], [1.], [2.], [5.], [6.], [7.]])
        d = np.abs(x - x.T)
        a = paired_tests(d, 3)
        order = [3, 4, 5, 0, 1, 2]
        b = paired_tests(d[np.ix_(order, order)], 3)
        self.assertEqual(a["n_permutations"], 8)
        self.assertEqual(a["p_permanova"], .25)
        self.assertAlmostEqual(a["p_permanova"], b["p_permanova"])
        self.assertAlmostEqual(a["p_permdisp"], b["p_permdisp"])
        self.assertTrue(0 <= a["R2"] <= 1)

    def test_bh_known_values(self):
        np.testing.assert_allclose(bh_adjust([.01, .04, .03]), [.03, .04, .04])


if __name__ == "__main__":unittest.main()
