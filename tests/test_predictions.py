from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, log_loss, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYTICS_ROOT = REPO_ROOT
sys.path.insert(0, str(ANALYTICS_ROOT / "src"))

from cs2_sql_analytics.build_database import build_database


class HoldoutPredictionsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.db_path = Path(cls.temp_dir.name) / "cs2.sqlite"
        build_database(REPO_ROOT, cls.db_path)
        cls.connection = sqlite3.connect(cls.db_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.connection.close()
        cls.temp_dir.cleanup()

    def test_holdout_predictions_reproduce_frozen_metrics(self) -> None:
        rows = self.connection.execute(
            """
            SELECT y_true, predicted_class, probability
            FROM fact_prediction_holdout
            ORDER BY model_match_id
            """
        ).fetchall()
        self.assertEqual(len(rows), 294)

        y_true = np.array([row[0] for row in rows])
        y_pred = np.array([row[1] for row in rows])
        probability = np.array([row[2] for row in rows])

        self.assertAlmostEqual(roc_auc_score(y_true, probability), 0.6828385899814471, places=12)
        self.assertAlmostEqual(accuracy_score(y_true, y_pred), 0.6292517006802721, places=12)
        self.assertAlmostEqual(log_loss(y_true, probability), 0.6387933840597699, places=12)
        self.assertAlmostEqual(brier_score_loss(y_true, probability), 0.22430145922903105, places=12)
        self.assertEqual(confusion_matrix(y_true, y_pred).tolist(), [[103, 51], [58, 82]])

    def test_prediction_descriptors_are_complete_and_consistent(self) -> None:
        row = self.connection.execute(
            """
            SELECT
                COUNT(*),
                COUNT(DISTINCT model_match_id),
                SUM(high_confidence),
                SUM(high_confidence_error),
                MIN(probability),
                MAX(probability),
                SUM(CASE WHEN correct NOT IN (0, 1) THEN 1 ELSE 0 END),
                SUM(CASE WHEN environment NOT IN ('LAN', 'ONLINE') THEN 1 ELSE 0 END),
                SUM(CASE WHEN elo_band_sort NOT BETWEEN 1 AND 4 THEN 1 ELSE 0 END),
                SUM(CASE WHEN probability_band_sort NOT BETWEEN 1 AND 10 THEN 1 ELSE 0 END)
            FROM fact_prediction_holdout
            """
        ).fetchone()
        self.assertEqual(row[:4], (294, 294, 24, 4))
        self.assertGreaterEqual(row[4], 0.0)
        self.assertLessEqual(row[5], 1.0)
        self.assertEqual(row[6:], (0, 0, 0, 0))

    def test_predictions_match_only_holdout_model_rows(self) -> None:
        missing_or_extra = self.connection.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT model_match_id FROM fact_model_match WHERE season_usage = 'holdout'
                EXCEPT
                SELECT model_match_id FROM fact_prediction_holdout
            )
            """
        ).fetchone()[0]
        non_holdout = self.connection.execute(
            """
            SELECT COUNT(*)
            FROM fact_prediction_holdout p
            JOIN fact_model_match m USING (model_match_id)
            WHERE m.season_usage <> 'holdout'
            """
        ).fetchone()[0]
        self.assertEqual(missing_or_extra, 0)
        self.assertEqual(non_holdout, 0)


if __name__ == "__main__":
    unittest.main()
