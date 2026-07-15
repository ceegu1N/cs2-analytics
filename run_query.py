from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd


ANALYTICS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ANALYTICS_ROOT / "src"))

from cs2_sql_analytics.config import DATABASE_PATH, QUERY_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa uma das consultas didáticas e mostra o resultado no terminal."
    )
    parser.add_argument("query", nargs="?", help="Nome do arquivo SQL, por exemplo 02_desempenho_equipes.sql")
    parser.add_argument("--list", action="store_true", help="Lista as consultas disponíveis.")
    parser.add_argument("--limit", type=int, default=20, help="Máximo de linhas exibidas no terminal.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    available = sorted(QUERY_DIR.glob("*.sql"))
    if args.list or not args.query:
        print("Consultas disponíveis:")
        for path in available:
            print(f"- {path.name}")
        if not args.query:
            return

    query_path = QUERY_DIR / args.query
    if query_path not in available:
        raise SystemExit(f"Consulta não encontrada: {args.query}. Use --list.")
    if not DATABASE_PATH.exists():
        raise SystemExit("Banco ainda não existe. Rode: python run_all.py --rebuild")

    connection = sqlite3.connect(DATABASE_PATH)
    try:
        frame = pd.read_sql_query(query_path.read_text(encoding="utf-8"), connection)
    finally:
        connection.close()
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(frame.head(args.limit).to_string(index=False))
    if len(frame) > args.limit:
        print(f"\nExibindo {args.limit} de {len(frame)} linhas.")


if __name__ == "__main__":
    main()
