from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYTICS_ROOT = REPO_ROOT
sys.path.insert(0, str(ANALYTICS_ROOT / "src"))

from cs2_sql_analytics.build_conclusions import build_conclusions
from cs2_sql_analytics.build_dashboard import build_dashboard
from cs2_sql_analytics.build_database import build_database


class GeneratedOutputsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        root = Path(cls.temp_dir.name)
        cls.db_path = root / "cs2.sqlite"
        cls.dashboard_dir = root / "dashboard"
        cls.conclusions_path = root / "conclusoes.md"
        build_database(REPO_ROOT, cls.db_path)
        cls.dashboard_paths = build_dashboard(cls.db_path, cls.dashboard_dir)
        build_conclusions(cls.db_path, cls.conclusions_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def test_five_dashboard_pages_and_html_are_generated(self) -> None:
        pngs = [path for path in self.dashboard_paths if path.suffix == ".png"]
        html = [path for path in self.dashboard_paths if path.suffix == ".html"]
        self.assertEqual(len(pngs), 5)
        self.assertEqual(len(html), 1)
        for path in pngs:
            with self.subTest(path=path.name):
                self.assertGreater(path.stat().st_size, 40_000)
                with Image.open(path) as image:
                    self.assertGreaterEqual(image.width, 1_600)
                self.assertGreaterEqual(image.height, 900)
        self.assertTrue(any(path.name == "05_desempenho_modelo.png" for path in pngs))
        html_text = html[0].read_text(encoding="utf-8")
        for path in pngs:
            self.assertIn(path.name, html_text)

    def test_team_dashboard_is_interactive_and_exposes_every_team_on_hover(self) -> None:
        html_path = next(path for path in self.dashboard_paths if path.suffix == ".html")
        html_text = html_path.read_text(encoding="utf-8")

        self.assertIn('id="teams-interactive-chart"', html_text)
        self.assertIn("data-nearest", html_text)
        self.assertIn("<svg", html_text)
        self.assertIn("Partidas representadas", html_text)
        self.assertIn("Taxa de vitória", html_text)
        self.assertIn("02_equipes.png", html_text)
        lines_with_trailing_whitespace = [
            line_number
            for line_number, line in enumerate(html_text.splitlines(), start=1)
            if line != line.rstrip()
        ]
        self.assertEqual(lines_with_trailing_whitespace, [])

        connection = sqlite3.connect(self.db_path)
        try:
            team_names = [
                row[0]
                for row in connection.execute(
                    """
                    SELECT t.team_name
                    FROM fact_team_match AS f
                    JOIN dim_team AS t ON t.team_id = f.team_id
                    GROUP BY t.team_id, t.team_name
                    HAVING COUNT(*) >= 50
                    ORDER BY AVG(f.win) DESC, COUNT(*) DESC, t.team_name
                    """
                )
            ]
        finally:
            connection.close()

        self.assertEqual(len(team_names), 95)
        for team_name in team_names:
            with self.subTest(team=team_name):
                self.assertIn(team_name, html_text)

    def test_high_confidence_errors_use_neutral_prediction_labels(self) -> None:
        html_path = next(path for path in self.dashboard_paths if path.suffix == ".html")
        html_text = html_path.read_text(encoding="utf-8")

        self.assertNotIn("Equipe favorita", html_text)
        self.assertNotIn("modelo apostou", html_text)
        self.assertIn("Equipe analisada", html_text)
        self.assertIn("Previsão", html_text)
        self.assertIn("Confiança", html_text)
        self.assertIn("Vitória", html_text)
        self.assertIn("Derrota", html_text)

    def test_power_bi_team_page_lists_all_eligible_teams_with_smaller_bubbles(self) -> None:
        power_bi_root = ANALYTICS_ROOT / "power_bi"
        table_path = (
            power_bi_root
            / "CS2_Analytics.Report"
            / "definition"
            / "pages"
            / "1ae4ea902a37e63b"
            / "visuals"
            / "tabela_equipes"
            / "visual.json"
        )
        theme_path = (
            power_bi_root
            / "CS2_Analytics.Report"
            / "StaticResources"
            / "RegisteredResources"
            / "sqlbi.json"
        )

        table = json.loads(table_path.read_text(encoding="utf-8-sig"))
        filters = table["filterConfig"]["filters"]
        self.assertNotIn("TopN", {item.get("type") for item in filters})

        minimum_sample = next(
            item
            for item in filters
            if item.get("field", {}).get("Measure", {}).get("Property")
            == "Partidas representadas"
        )
        comparison = minimum_sample["filter"]["Where"][0]["Condition"]["Comparison"]
        self.assertEqual(comparison["ComparisonKind"], 2)
        self.assertEqual(comparison["Right"]["Literal"]["Value"], "50L")

        theme = json.loads(theme_path.read_text(encoding="utf-8-sig"))
        bubble_size = theme["visualStyles"]["scatterChart"]["*"]["bubbles"][0]["bubbleSize"]
        self.assertEqual(bubble_size, -20)

        connection = sqlite3.connect(self.db_path)
        try:
            eligible_teams = connection.execute(
                """
                SELECT COUNT(*)
                FROM (
                    SELECT team_id
                    FROM fact_team_match
                    GROUP BY team_id
                    HAVING COUNT(*) >= 50
                )
                """
            ).fetchone()[0]
        finally:
            connection.close()
        self.assertEqual(eligible_teams, 95)

    def test_five_real_nontechnical_conclusions_are_generated(self) -> None:
        text = self.conclusions_path.read_text(encoding="utf-8")
        numbered = [line for line in text.splitlines() if line[:2] in {"1.", "2.", "3.", "4.", "5."}]
        self.assertEqual(len(numbered), 5)
        self.assertIn("14.028", text)
        self.assertIn("ELO", text)
        self.assertNotIn("TODO", text)
        self.assertNotIn("mapa específico", text)


if __name__ == "__main__":
    unittest.main()
