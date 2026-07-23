import unittest

import numpy as np

from sensitivity_analysis import (
    bootstrap_prcc,
    convergence_diagnostics,
    latin_hypercube,
    nonlinear_permutation_importance,
    partial_rank_correlations,
    scale_design,
    variance_decomposition,
)


class SensitivityStatisticsTests(unittest.TestCase):
    def test_latin_hypercube_uses_every_stratum_once_per_parameter(self):
        design = latin_hypercube(20, 4, seed=11)
        self.assertEqual(design.shape, (20, 4))
        self.assertTrue(np.all((design >= 0.0) & (design <= 1.0)))
        for column in range(design.shape[1]):
            strata = np.floor(design[:, column] * 20).astype(int)
            self.assertEqual(sorted(strata.tolist()), list(range(20)))

    def test_scaling_respects_physical_bounds(self):
        unit = np.array([[0.0, 0.5], [1.0, 1.0]])
        scaled = scale_design(unit, [(10.0, 20.0), (-2.0, 2.0)])
        np.testing.assert_allclose(scaled, [[10.0, 0.0], [20.0, 2.0]])

    def test_prcc_recovers_conditional_directions(self):
        rng = np.random.default_rng(4)
        x = rng.uniform(size=(300, 3))
        y = 3.0 * x[:, 0] - 2.0 * x[:, 1] + rng.normal(0, 0.05, 300)
        result = partial_rank_correlations(x, y)
        self.assertGreater(result[0]["coefficient"], 0.9)
        self.assertLess(result[1]["coefficient"], -0.9)
        self.assertLess(abs(result[2]["coefficient"]), 0.2)

    def test_bootstrap_prcc_returns_ordered_intervals(self):
        rng = np.random.default_rng(5)
        x = rng.uniform(size=(120, 3))
        y = x[:, 0] + rng.normal(0, 0.1, 120)
        result = bootstrap_prcc(x, y, n_bootstrap=50, seed=6)
        self.assertEqual(len(result), 3)
        for row in result:
            self.assertLessEqual(row["ci_low"], row["coefficient"])
            self.assertGreaterEqual(row["ci_high"], row["coefficient"])

    def test_variance_decomposition_separates_parameter_and_seed_noise(self):
        outputs = np.array([[0.0, 0.1], [10.0, 10.1], [20.0, 20.1]])
        result = variance_decomposition(outputs)
        self.assertGreater(result["parameter_variance_share"], 0.99)

    def test_convergence_diagnostic_reaches_full_sample_reference(self):
        rng = np.random.default_rng(8)
        x = rng.uniform(size=(100, 4))
        y = 2 * x[:, 0] - x[:, 1] + rng.normal(0, 0.05, 100)
        result = convergence_diagnostics(x, y)
        self.assertEqual(result["rows"][-1]["max_abs_change"], 0.0)
        self.assertEqual(result["rows"][-1]["rank_stability"], 1.0)

    def test_nonlinear_importance_is_rejected_without_predictive_skill(self):
        rng = np.random.default_rng(9)
        x = rng.uniform(size=(60, 4))
        y = rng.normal(size=60)
        result = nonlinear_permutation_importance(x, y, seed=10)
        self.assertEqual(result["status"], "rejected_no_predictive_skill")
        np.testing.assert_allclose(result["importance"], 0.0)


if __name__ == "__main__":
    unittest.main()
