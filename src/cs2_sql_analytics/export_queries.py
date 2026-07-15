from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .config import DATABASE_PATH, QUERY_DIR, QUERY_OUTPUT_DIR


def export_queries(
    database_path: Path | str = DATABASE_PATH,
    query_dir: Path | str = QUERY_DIR,
    output_dir: Path | str = QUERY_OUTPUT_DIR,
) -> list[Path]:
    database_path = Path(database_path)
    query_dir = Path(query_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    query_paths = sorted(query_dir.glob("*.sql"))
    if not query_paths:
        raise FileNotFoundError(f"Nenhuma consulta SQL encontrada em {query_dir}")

    exported: list[Path] = []
    connection = sqlite3.connect(database_path)
    try:
        for query_path in query_paths:
            sql = query_path.read_text(encoding="utf-8")
            frame = pd.read_sql_query(sql, connection)
            if frame.empty:
                raise ValueError(f"A consulta {query_path.name} não retornou linhas.")
            destination = output_dir / f"{query_path.stem}.csv"
            frame.to_csv(
                destination,
                index=False,
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
            )
            exported.append(destination)
    finally:
        connection.close()
    return exported


if __name__ == "__main__":
    paths = export_queries()
    print(f"{len(paths)} resultados exportados para {paths[0].parent}")
