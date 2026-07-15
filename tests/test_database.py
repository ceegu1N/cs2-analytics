from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYTICS_ROOT = REPO_ROOT
sys.path.insert(0, str(ANALYTICS_ROOT / "src"))

from cs2_sql_analytics.build_database import build_database


class DatabaseBuildTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.db_path = Path(cls.temp_dir.name) / "cs2_analytics.sqlite"
        build_database(repo_root=REPO_ROOT, database_path=cls.db_path)
        cls.connection = sqlite3.connect(cls.db_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.connection.close()
        cls.temp_dir.cleanup()

    def scalar(self, sql: str):
        return self.connection.execute(sql).fetchone()[0]

    def test_expected_tables_exist(self) -> None:
        names = {
            row[0]
            for row in self.connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        self.assertTrue(
            {"dim_team", "dim_date", "fact_team_match", "fact_match", "fact_model_match"}
            <= names
        )

    def test_source_granularities_are_preserved(self) -> None:
        self.assertEqual(self.scalar("SELECT COUNT(*) FROM fact_team_match"), 20_527)
        self.assertEqual(self.scalar("SELECT COUNT(*) FROM fact_match"), 14_028)
        self.assertEqual(self.scalar("SELECT COUNT(*) FROM fact_model_match"), 2_840)
        self.assertEqual(
            self.scalar("SELECT COUNT(DISTINCT match_id) FROM fact_match"), 14_028
        )
        self.assertEqual(
            self.scalar("SELECT COUNT(DISTINCT model_match_id) FROM fact_model_match"),
            2_840,
        )

    def test_training_and_holdout_counts_match_the_tcc(self) -> None:
        counts = dict(
            self.connection.execute(
                "SELECT season_usage, COUNT(*) FROM fact_model_match GROUP BY season_usage"
            )
        )
        self.assertEqual(counts, {"holdout": 294, "train": 2_546})

    def test_dates_cover_the_operational_source(self) -> None:
        bounds = self.connection.execute(
            """
            SELECT MIN(d.date_iso), MAX(d.date_iso)
            FROM fact_team_match f
            JOIN dim_date d ON d.date_key = f.match_date_key
            """
        ).fetchone()
        self.assertEqual(bounds, ("2023-09-27", "2026-03-22"))

    def test_canonical_match_keeps_scores_on_the_correct_side(self) -> None:
        row = self.connection.execute(
            """
            SELECT ta.team_key, tb.team_key, f.team_a_score, f.team_b_score, f.team_a_win
            FROM fact_match f
            JOIN dim_team ta ON ta.team_id = f.team_a_id
            JOIN dim_team tb ON tb.team_id = f.team_b_id
            WHERE f.match_id = 2391131
            """
        ).fetchone()
        self.assertEqual(row, ("3dmax", "natus vincere", 1, 2, 0))

    def test_foreign_keys_and_uniqueness_are_valid(self) -> None:
        self.assertEqual(list(self.connection.execute("PRAGMA foreign_key_check")), [])
        duplicates = self.scalar(
            """
            SELECT COUNT(*) FROM (
                SELECT match_id, team_id, COUNT(*) AS n
                FROM fact_team_match
                GROUP BY match_id, team_id
                HAVING n > 1
            )
            """
        )
        self.assertEqual(duplicates, 0)


if __name__ == "__main__":
    unittest.main()
