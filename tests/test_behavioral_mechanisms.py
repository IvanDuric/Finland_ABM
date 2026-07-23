import unittest

from behavioral_mechanisms import (
    effective_hoarding_multiplier,
    normalized_tpb_weights,
    tpb_intention,
)


class BehavioralMechanismTests(unittest.TestCase):
    def test_tpb_weights_form_a_convex_combination(self):
        weights = normalized_tpb_weights()
        self.assertAlmostEqual(sum(weights), 1.0)
        self.assertTrue(all(0.0 <= weight <= 1.0 for weight in weights))

    def test_tpb_intention_does_not_clip_high_equal_inputs(self):
        self.assertAlmostEqual(tpb_intention(0.9, 0.9, 0.9), 0.9)
        self.assertAlmostEqual(tpb_intention(1.0, 1.0, 1.0), 1.0)
        self.assertAlmostEqual(tpb_intention(0.0, 0.0, 0.0), 0.0)

    def test_invalid_tpb_weights_fail_loudly(self):
        with self.assertRaises(ValueError):
            normalized_tpb_weights((0.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            normalized_tpb_weights((0.4, -0.1, 0.7))

    def test_hoarding_is_continuous_and_bounded(self):
        values = [
            effective_hoarding_multiplier(3.0, 0.75, panic)
            for panic in (0.0, 0.2, 0.4, 0.6, 1.0)
        ]
        self.assertEqual(values[0], 1.0)
        self.assertTrue(all(left < right for left, right in zip(values, values[1:])))
        self.assertLessEqual(values[-1], 3.0)
        self.assertAlmostEqual(values[-1], 2.5)

    def test_no_propensity_means_no_hoarding_amplification(self):
        self.assertEqual(effective_hoarding_multiplier(3.0, 0.0, 1.0), 1.0)


if __name__ == "__main__":
    unittest.main()
