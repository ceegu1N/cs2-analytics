from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .config import ANALYTICS_ROOT, DATABASE_PATH


POWER_BI_TABLES = (
    "dim_team",
    "dim_date",
    "fact_team_match",
    "fact_match",
    "fact_model_match",
    "fact_prediction_holdout",
)


def configure_power_bi_data_root(
    project_root: Path | str = ANALYTICS_ROOT,
    expression_path: Path | str | None = None,
) -> Path:
    project_root = Path(project_root).resolve()
    if expression_path is None:
        expression_path = (
            project_root
            / "power_bi"
            / "CS2_Analytics.SemanticModel"
            / "definition"
            / "expressions.tmdl"
        )
    expression_path = Path(expression_path)
    if not expression_path.exists():
        raise FileNotFoundError(
            f"Parâmetro DataRoot do Power BI não encontrado: {expression_path}"
        )

    lines = expression_path.read_text(encoding="utf-8-sig").splitlines()
    if not lines or not lines[0].startswith("expression DataRoot ="):
        raise ValueError(
            f"Formato inesperado no parâmetro DataRoot: {expression_path}"
        )

    data_root = (project_root / "output" / "power_bi_tables").resolve()
    lines[0] = (
        f'expression DataRoot = "{data_root}" '
        'meta [IsParameterQuery=true, Type="Text", IsParameterQueryRequired=true]'
    )
    expression_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return expression_path


def export_power_bi_tables(
    database_path: Path | str = DATABASE_PATH,
    output_dir: Path | str = ANALYTICS_ROOT / "output" / "power_bi_tables",
) -> list[Path]:
    database_path = Path(database_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    destinations: list[Path] = []
    connection = sqlite3.connect(database_path)
    try:
        for table_name in POWER_BI_TABLES:
            frame = pd.read_sql_query(f"SELECT * FROM {table_name}", connection)
            if table_name == "dim_team":
                frame["hltv_team_id"] = frame["hltv_team_id"].astype("Int64")
            destination = output_dir / f"{table_name}.csv"
            frame.to_csv(
                destination,
                index=False,
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
            )
            destinations.append(destination)
    finally:
        connection.close()
    return destinations


if __name__ == "__main__":
    paths = export_power_bi_tables()
    print(f"{len(paths)} tabelas exportadas para {paths[0].parent}")
