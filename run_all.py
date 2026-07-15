from __future__ import annotations

import argparse
import sys
from pathlib import Path


ANALYTICS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ANALYTICS_ROOT / "src"))

from cs2_sql_analytics.build_conclusions import build_conclusions
from cs2_sql_analytics.build_dashboard import build_dashboard
from cs2_sql_analytics.build_database import build_database
from cs2_sql_analytics.config import DATABASE_PATH, QUERY_DIR, QUERY_OUTPUT_DIR, REPO_ROOT
from cs2_sql_analytics.export_power_bi import (
    configure_power_bi_data_root,
    export_power_bi_tables,
)
from cs2_sql_analytics.export_queries import export_queries


def run(rebuild: bool = False) -> None:
    if rebuild or not DATABASE_PATH.exists():
        print("[1/5] Construindo o banco SQLite...")
        build_database(repo_root=REPO_ROOT, database_path=DATABASE_PATH)
    else:
        print("[1/5] Reutilizando o banco existente. Use --rebuild para recriá-lo.")

    print("[2/5] Executando e exportando as consultas SQL...")
    query_paths = export_queries(DATABASE_PATH, QUERY_DIR, QUERY_OUTPUT_DIR)

    print("[3/5] Exportando as tabelas para o Power BI...")
    power_bi_paths = export_power_bi_tables(DATABASE_PATH)
    power_bi_expression = configure_power_bi_data_root(REPO_ROOT)

    print("[4/5] Gerando os painéis estáticos e o HTML interativo...")
    dashboard_paths = build_dashboard(DATABASE_PATH)

    print("[5/5] Escrevendo conclusões não técnicas...")
    conclusions_path = build_conclusions(DATABASE_PATH)

    print("\nConcluído.")
    print(f"Banco: {DATABASE_PATH}")
    print(f"Consultas exportadas: {len(query_paths)}")
    print(f"Tabelas para Power BI: {len(power_bi_paths)}")
    print(f"Caminho do Power BI configurado em: {power_bi_expression}")
    print(f"Painéis: {len([p for p in dashboard_paths if p.suffix == '.png'])}")
    print(f"Abra no navegador: {ANALYTICS_ROOT / 'output' / 'dashboard' / 'index.html'}")
    print(f"Conclusões: {conclusions_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Constrói a extensão SQL e os artefatos analíticos do TCC."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Recria o banco SQLite a partir dos CSVs antes de gerar os resultados.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(rebuild=parse_args().rebuild)
