from __future__ import annotations

import unittest

from evals.authoring.pi99.scoring import load_key, score_pi99, validate_key


class Pi99ScoringTests(unittest.TestCase):
    def test_published_key_has_expected_dimensions(self) -> None:
        key = load_key()
        self.assertEqual(validate_key(key), {"tertiary": 22, "secondary": 3, "primary": 1})
        self.assertEqual(len(key["item_ids"]), 99)
        self.assertEqual(len(set(key["item_ids"])), 99)
        self.assertEqual(len(key["reverse_marked_item_ids"]), 39)
        self.assertEqual(len(key["subscales"]["secondary"]["Safe"]), 29)
        self.assertEqual(len(key["subscales"]["secondary"]["Enticing"]), 28)
        self.assertEqual(len(key["subscales"]["secondary"]["Alive"]), 14)
        self.assertEqual(len(key["subscales"]["primary"]["Good"]), 71)

    def test_synthetic_polarity_responses_expose_published_source_anomaly(self) -> None:
        key = load_key()
        reverse = set(key["reverse_marked_item_ids"])
        responses = {
            item_id: 0 if item_id in reverse else 5 for item_id in key["item_ids"]
        }
        scores = score_pi99(responses, key=key)
        for name, value in scores.items():
            if name not in {"Intentional", "Alive"}:
                self.assertEqual(value, 5, name)
        self.assertEqual(scores["Intentional"], 4)
        self.assertAlmostEqual(scores["Alive"], 65 / 14)

    def test_invalid_or_missing_responses_fail_closed(self) -> None:
        key = load_key()
        responses = {item_id: 3 for item_id in key["item_ids"]}
        responses.pop("ab1")
        with self.assertRaisesRegex(KeyError, "ab1"):
            score_pi99(responses, key=key)
        responses["ab1"] = 6
        with self.assertRaisesRegex(ValueError, "outside"):
            score_pi99(responses, key=key)


if __name__ == "__main__":
    unittest.main()
