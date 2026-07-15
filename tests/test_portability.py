from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cs2_sql_analytics.export_power_bi import configure_power_bi_data_root


class PowerBiPortabilityTest(unittest.TestCase):
    def test_data_root_is_rewritten_for_the_current_clone(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            clone_root = Path(temporary_directory) / "clone"
            expression_path = (
                clone_root
                / "power_bi"
                / "CS2_Analytics.SemanticModel"
                / "definition"
                / "expressions.tmdl"
            )
            expression_path.parent.mkdir(parents=True)
            expression_path.write_text(
                'expression DataRoot = "C:\\OLD_PROJECT\\output\\power_bi_tables" '
                'meta [IsParameterQuery=true, Type="Text", IsParameterQueryRequired=true]\n',
                encoding="utf-8",
            )

            configured_path = configure_power_bi_data_root(
                project_root=clone_root,
                expression_path=expression_path,
            )

            text = configured_path.read_text(encoding="utf-8")
            expected_data_root = str(
                (clone_root / "output" / "power_bi_tables").resolve()
            )
            self.assertIn(expected_data_root, text)
            self.assertNotIn("OLD_PROJECT", text)
            self.assertIn("IsParameterQueryRequired=true", text)


if __name__ == "__main__":
    unittest.main()
