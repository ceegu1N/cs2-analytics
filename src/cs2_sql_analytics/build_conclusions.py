from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import ANALYTICS_ROOT, DATABASE_PATH, QUERY_DIR


def _read(connection: sqlite3.Connection, filename: str) -> pd.DataFrame:
    sql = (QUERY_DIR / filename).read_text(encoding="utf-8")
    return pd.read_sql_query(sql, connection)


def _integer(value: int | float) -> str:
    return f"{int(value):,}".replace(",", ".")


def _decimal(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _date(value: str) -> str:
    return datetime.strptime(value, "%Y-%m-%d").strftime("%d/%m/%Y")


def build_conclusions(
    database_path: Path | str = DATABASE_PATH,
    output_path: Path | str = ANALYTICS_ROOT / "output" / "conclusoes_nao_tecnicas.md",
) -> Path:
    database_path = Path(database_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(database_path)
    try:
        overview = _read(connection, "01_visao_geral.sql").iloc[0]
        formats = _read(connection, "05_formatos_serie.sql")
        context = _read(connection, "04_contexto_lan_online.sql").set_index("ambiente")
        elo = _read(connection, "07_faixas_elo.sql").set_index("faixa_elo")
        split = _read(connection, "08_holdout_treino.sql").set_index("recorte")
    finally:
        connection.close()

    dominant_format = formats.sort_values("partidas_unicas", ascending=False).iloc[0]
    lines = [
        "# Cinco conclusões para um leitor não técnico",
        "",
        (
            f"1. A base operacional cobre {_integer(overview.partidas_unicas)} partidas únicas entre "
            f"{_date(overview.primeira_data)} e {_date(overview.ultima_data)}. As "
            f"{_integer(overview.perspectivas_de_equipe)} linhas de equipe não são o número de jogos: "
            "uma mesma partida pode aparecer pelo ponto de vista de duas equipes."
        ),
        (
            f"2. O formato mais frequente é {dominant_format.formato}, com "
            f"{_decimal(dominant_format.participacao_pct)}% das partidas únicas. Isso mostra por que "
            "o formato da série precisa ser mantido como contexto da análise."
        ),
        (
            f"3. No recorte de modelagem, as partidas LAN tiveram diferença média absoluta de ELO de "
            f"{_decimal(context.loc['LAN', 'diferenca_elo_media'])} pontos; nas partidas online, a média foi "
            f"{_decimal(context.loc['ONLINE', 'diferenca_elo_media'])}. O ambiente e o equilíbrio dos confrontos "
            "não devem ser tratados como se fossem a mesma coisa."
        ),
        (
            f"4. Quando a diferença absoluta de ELO ficou abaixo de 50 pontos, a equipe com maior ELO venceu "
            f"{_decimal(100 * elo.loc['0-49', 'taxa_vitoria_maior_elo'])}% das vezes. Na faixa de 200 pontos ou mais, "
            f"essa taxa foi de {_decimal(100 * elo.loc['200+', 'taxa_vitoria_maior_elo'])}%. A associação fica mais "
            "forte quando a separação pré-jogo é maior, sem que isso prove causalidade."
        ),
        (
            f"5. O treino reúne {_integer(split.loc['train', 'partidas'])} partidas e o holdout, "
            f"{_integer(split.loc['holdout', 'partidas'])}. A diferença média absoluta de ELO foi de "
            f"{_decimal(split.loc['train', 'diferenca_elo_media'])} no treino e "
            f"{_decimal(split.loc['holdout', 'diferenca_elo_media'])} no holdout; por isso, comparar os recortes "
            "também exige observar o contexto dos confrontos."
        ),
        "",
        "Essas conclusões são descritivas. Elas ajudam a formular perguntas e não substituem a avaliação preditiva do TCC.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


if __name__ == "__main__":
    print(build_conclusions())
