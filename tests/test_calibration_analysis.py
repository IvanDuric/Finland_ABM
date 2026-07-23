import unittest

import numpy as np

from calibration_analysis import (
    calibration_design,
    identifiability_diagnostics,
    standardized_rmse,
    waste_rate_percent,
)


class CalibrationAnalysisTests(unittest.TestCase):
    def test_design_respects_ranges_and_integer_parameters(self):
        design = calibration_design(40, [(0.1, 0.9, "float"), (1, 8, "int")], seed=4)
        self.assertTrue(np.all((design[:, 0] >= 0.1) & (design[:, 0] <= 0.9)))
        self.assertTrue(np.all((design[:, 1] >= 1) & (design[:, 1] <= 8)))
        self.assertTrue(np.all(design[:, 1] == np.rint(design[:, 1])))

    def test_standardized_rmse_uses_declared_scales(self):
        value = standardized_rmse([110, 0.8], [100, 0.9], [10, 0.1])
        self.assertAlmostEqual(value, 1.0)

    def test_waste_rate_uses_physical_throughput(self):
        self.assertAlmostEqual(waste_rate_percent([80, 10], [5, 5]), 10.0)

    def test_recovery_accepts_identifiable_two_parameter_mapping(self):
        rng = np.random.default_rng(9)
        design = rng.uniform(0, 1, size=(160, 2))
        means = np.column_stack((design[:, 0], design[:, 1]))
        replicates = np.stack([
            means + rng.normal(0, 0.005, means.shape) for _ in range(3)
        ], axis=1)

        result = identifiability_diagnostics(["a", "b"], design, replicates)

        self.assertTrue(result["recommendation_allowed"])
        self.assertTrue(all(row["individually_recoverable"] for row in result["parameters"]))

    def test_recovery_rejects_parameter_absent_from_targets(self):
        rng = np.random.default_rng(10)
        design = rng.uniform(0, 1, size=(120, 2))
        means = design[:, [0]]
        replicates = np.stack([means, means, means], axis=1)

        result = identifiability_diagnostics(["visible", "hidden"], design, replicates)

        rows = {row["parameter"]: row for row in result["parameters"]}
        self.assertFalse(result["recommendation_allowed"])
        self.assertTrue(rows["visible"]["individually_recoverable"])
        self.assertFalse(rows["hidden"]["individually_recoverable"])
        self.assertFalse(result["rank_adequate"])


if __name__ == "__main__":
    unittest.main()
