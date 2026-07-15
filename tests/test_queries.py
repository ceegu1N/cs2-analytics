from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYTICS_ROOT = REPO_ROOT
sys.path.insert(0, str(ANALYTICS_ROOT / "src"))

from cs2_sql_analytics.build_database import build_database
from cs2_sql_analytics.export_power_bi import export_power_bi_tables
from cs2_sql_analytics.export_queries import export_queries


class QueryExecutionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        root = Path(cls.temp_dir.name)
        cls.db_path = root / "cs2.sqlite"
        cls.output_dir = root / "results"
        build_database(REPO_ROOT, cls.db_path)
        cls.connection = sqlite3.connect(cls.db_path)
        cls.query_paths = sorted((ANALYTICS_ROOT / "sql" / "queries").glob("*.sql"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.connection.close()
        cls.temp_dir.cleanup()

    def test_nine_guided_queries_exist_and_execute(self) -> None:
        self.assertEqual(len(self.query_paths), 9)
        for path in self.query_paths:
            with self.subTest(query=path.name):
                frame = pd.read_sql_query(path.read_text(encoding="utf-8"), self.connection)
                self.assertFalse(frame.empty)
                self.assertGreater(len(frame.columns), 1)

    def test_overview_uses_unique_matches(self) -> None:
        path = ANALYTICS_ROOT / "sql" / "queries" / "01_visao_geral.sql"
        row = pd.read_sql_query(path.read_text(encoding="utf-8"), self.connection).iloc[0]
        self.assertEqual(int(row["partidas_unicas"]), 14_028)
        self.assertEqual(int(row["perspectivas_de_equipe"]), 20_527)
        self.assertEqual(int(row["partidas_modelagem"]), 2_840)

    def test_elo_query_reports_rates_in_valid_range(self) -> None:
        path = ANALYTICS_ROOT / "sql" / "queries" / "07_faixas_elo.sql"
        frame = pd.read_sql_query(path.read_text(encoding="utf-8"), self.connection)
        self.assertGreaterEqual(len(frame), 4)
        self.assertTrue(frame["taxa_vitoria_maior_elo"].between(0, 1).all())

    def test_export_uses_pt_br_friendly_csv(self) -> None:
        exported = export_queries(
            database_path=self.db_path,
            query_dir=ANALYTICS_ROOT / "sql" / "queries",
            output_dir=self.output_dir,
        )
        self.assertEqual(len(exported), 9)
        overview = pd.read_csv(
            self.output_dir / "01_visao_geral.csv",
            sep=";",
            decimal=",",
            encoding="utf-8-sig",
        )
        self.assertEqual(int(overview.loc[0, "partidas_unicas"]), 14_028)

    def test_power_bi_export_preserves_table_row_counts(self) -> None:
        destinations = export_power_bi_tables(
            database_path=self.db_path,
            output_dir=Path(self.temp_dir.name) / "power_bi",
        )
        self.assertEqual(len(destinations), 6)
        counts = {
            path.stem: len(
                pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig")
            )
            for path in destinations
        }
        self.assertEqual(counts["fact_team_match"], 20_527)
        self.assertEqual(counts["fact_match"], 14_028)
        self.assertEqual(counts["fact_model_match"], 2_840)
        self.assertEqual(counts["fact_prediction_holdout"], 294)
        dim_team_text = (Path(self.temp_dir.name) / "power_bi" / "dim_team.csv").read_text(
            encoding="utf-8-sig"
        )
        self.assertIn(";3dmax;3DMAX;4914\n", dim_team_text)


if __name__ == "__main__":
    unittest.main()
