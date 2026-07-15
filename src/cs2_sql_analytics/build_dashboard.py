from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_hex

from .config import DASHBOARD_DIR, DATABASE_PATH, QUERY_DIR
from .dashboard_html import (
    PALETTE_LIGHT,
    load_dashboard_data,
    luminance,
    pt_int,
    pt_num,
    pt_pct,
    rank_auc,
    render_index_html,
)


COLORS = {
    "ink": PALETTE_LIGHT["ink"],
    "ink2": PALETTE_LIGHT["ink2"],
    "muted": PALETTE_LIGHT["muted"],
    "s1": PALETTE_LIGHT["s1"],
    "s2": PALETTE_LIGHT["s2"],
    "accent": PALETTE_LIGHT["ramp"][2],
    "paper": PALETTE_LIGHT["page"],
    "grid": PALETTE_LIGHT["grid"],
}
RAMP = PALETTE_LIGHT["ramp"]
CMAP_SEQ = LinearSegmentedColormap.from_list("cs2_seq", PALETTE_LIGHT["seq"])


def _query(connection: sqlite3.Connection, filename: str) -> pd.DataFrame:
    sql = (QUERY_DIR / filename).read_text(encoding="utf-8")
    return pd.read_sql_query(sql, connection)


def _style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor("white")
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(COLORS["grid"])
    axis.tick_params(colors=COLORS["muted"], labelsize=9)
    axis.grid(axis="y", color=COLORS["grid"], linewidth=0.7, alpha=0.65)
    axis.set_axisbelow(True)


def _title(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.suptitle(
        title,
        x=0.055,
        y=0.98,
        ha="left",
        va="top",
        fontsize=24,
        fontweight="bold",
        color=COLORS["accent"],
    )
    fig.text(0.055, 0.905, subtitle, ha="left", fontsize=11, color=COLORS["ink2"])


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=120, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _overview_page(connection: sqlite3.Connection, destination: Path) -> None:
    overview = _query(connection, "01_visao_geral.sql").iloc[0]
    monthly = _query(connection, "03_evolucao_mensal.sql")
    formats = _query(connection, "05_formatos_serie.sql")

    fig = plt.figure(figsize=(16, 9), facecolor=COLORS["paper"])
    grid = fig.add_gridspec(2, 2, left=0.055, right=0.97, bottom=0.08, top=0.85, hspace=0.34, wspace=0.22)
    _title(fig, "CS2 Analytics | Visão geral", "Contagens globais usam uma linha por partida; perspectivas de equipe são analisadas separadamente.")

    cards = fig.add_subplot(grid[0, 0])
    cards.axis("off")
    card_values = [
        ("Partidas únicas", pt_int(overview.partidas_unicas)),
        ("Perspectivas de equipe", pt_int(overview.perspectivas_de_equipe)),
        ("Recorte de modelagem", pt_int(overview.partidas_modelagem)),
        ("Equipes representadas", pt_int(overview.equipes_representadas)),
    ]
    for index, (label, value) in enumerate(card_values):
        row, column = divmod(index, 2)
        x = 0.02 + column * 0.5
        y = 0.68 - row * 0.47
        cards.text(x, y, value, fontsize=30, fontweight="bold", color=COLORS["accent"])
        cards.text(x, y - 0.13, label, fontsize=11, color=COLORS["ink2"])
    cards.text(
        0.02,
        0.02,
        f"Período operacional: {overview.primeira_data} a {overview.ultima_data}",
        fontsize=10,
        color=COLORS["ink"],
    )

    axis = fig.add_subplot(grid[0, 1])
    _style_axis(axis)
    axis.plot(monthly["mes"], monthly["partidas_unicas"], color=COLORS["s1"], linewidth=2.5, marker="o", markersize=3)
    axis.fill_between(np.arange(len(monthly)), monthly["partidas_unicas"], color=COLORS["s1"], alpha=0.1)
    axis.set_title("Volume mensal de partidas únicas", loc="left", fontsize=14, fontweight="bold", color=COLORS["ink"])
    axis.set_ylabel("Partidas")
    ticks = np.arange(0, len(monthly), max(1, len(monthly) // 8))
    axis.set_xticks(ticks, monthly.iloc[ticks]["mes"], rotation=35, ha="right")

    axis = fig.add_subplot(grid[1, 0])
    _style_axis(axis)
    context_labels = ["LAN", "Online"]
    context_values = [int(overview.partidas_lan), int(overview.partidas_online)]
    bars = axis.bar(context_labels, context_values, color=COLORS["s1"], width=0.45)
    axis.set_title("Ambiente das partidas", loc="left", fontsize=14, fontweight="bold", color=COLORS["ink"])
    axis.set_ylabel("Partidas únicas")
    axis.bar_label(bars, labels=[pt_int(value) for value in context_values], padding=4, color=COLORS["ink"])

    axis = fig.add_subplot(grid[1, 1])
    _style_axis(axis)
    bars = axis.bar(formats["formato"], formats["participacao_pct"], color=COLORS["s1"], width=0.45)
    axis.set_title("Distribuição dos formatos de série", loc="left", fontsize=14, fontweight="bold", color=COLORS["ink"])
    axis.set_ylabel("Participação (%)")
    axis.bar_label(bars, labels=[pt_pct(value) for value in formats["participacao_pct"]], padding=4, color=COLORS["ink"])
    _save(fig, destination)


def _teams_page(connection: sqlite3.Connection, destination: Path) -> None:
    all_teams = _query(connection, "02_desempenho_equipes.sql")
    ranked_teams = all_teams.head(20).sort_values("taxa_vitoria")
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), facecolor=COLORS["paper"], gridspec_kw={"left": 0.07, "right": 0.97, "bottom": 0.09, "top": 0.84, "wspace": 0.25})
    _title(fig, "CS2 Analytics | Equipes", "A taxa considera apenas as partidas em que a equipe aparece como lado representado na fonte operacional.")

    axis = axes[0]
    _style_axis(axis)
    bars = axis.barh(ranked_teams["equipe"], 100 * ranked_teams["taxa_vitoria"], color=COLORS["s1"], height=0.62)
    axis.set_title("Maior taxa de vitória (mínimo de 50 partidas)", loc="left", fontsize=14, fontweight="bold", color=COLORS["ink"])
    axis.set_xlabel("Taxa de vitória (%)")
    axis.bar_label(bars, labels=[pt_pct(100 * value) for value in ranked_teams["taxa_vitoria"]], padding=3, fontsize=8, color=COLORS["ink2"])

    axis = axes[1]
    _style_axis(axis)
    axis.scatter(all_teams["partidas_representadas"], 100 * all_teams["taxa_vitoria"], s=65, color=COLORS["s1"], alpha=0.85, edgecolors="white", linewidths=1)
    axis.set_title(f"Volume e taxa de vitória ({len(all_teams)} equipes)", loc="left", fontsize=14, fontweight="bold", color=COLORS["ink"])
    axis.set_xlabel("Partidas representadas")
    axis.set_ylabel("Taxa de vitória (%)")
    for row in all_teams.nlargest(6, "partidas_representadas").itertuples():
        axis.annotate(row.equipe, (row.partidas_representadas, 100 * row.taxa_vitoria), xytext=(5, 4), textcoords="offset points", fontsize=8, color=COLORS["ink"])
    _save(fig, destination)


def _context_page(connection: sqlite3.Connection, destination: Path) -> None:
    monthly = _query(connection, "03_evolucao_mensal.sql")
    context = _query(connection, "04_contexto_lan_online.sql")
    fig = plt.figure(figsize=(16, 9), facecolor=COLORS["paper"])
    grid = fig.add_gridspec(2, 2, left=0.06, right=0.97, bottom=0.09, top=0.84, hspace=0.35, wspace=0.25)
    _title(fig, "CS2 Analytics | Tempo e contexto", "LAN e online descrevem o ambiente; não são identificadores de mapas.")

    axis = fig.add_subplot(grid[0, :])
    _style_axis(axis)
    x = np.arange(len(monthly))
    axis.stackplot(x, monthly["partidas_lan"], monthly["partidas_online"], labels=["LAN", "Online"], colors=[COLORS["s1"], COLORS["s2"]], alpha=0.85)
    axis.set_title("Composição mensal por ambiente", loc="left", fontsize=14, fontweight="bold", color=COLORS["ink"])
    ticks = np.arange(0, len(monthly), max(1, len(monthly) // 9))
    axis.set_xticks(ticks, monthly.iloc[ticks]["mes"], rotation=30, ha="right")
    axis.set_ylabel("Partidas únicas")
    axis.legend(frameon=False, ncol=2, loc="upper left")

    axis = fig.add_subplot(grid[1, 0])
    _style_axis(axis)
    bars = axis.bar(context["ambiente"], context["diferenca_elo_media"], color=COLORS["s1"], width=0.45)
    axis.set_title("Diferença média absoluta de ELO", loc="left", fontsize=14, fontweight="bold", color=COLORS["ink"])
    axis.set_ylabel("Pontos de ELO")
    axis.bar_label(bars, labels=[pt_num(value) for value in context["diferenca_elo_media"]], padding=4, color=COLORS["ink"])

    axis = fig.add_subplot(grid[1, 1])
    _style_axis(axis)
    values = 100 * context["taxa_vitoria_maior_elo"]
    bars = axis.bar(context["ambiente"], values, color=COLORS["s1"], width=0.45)
    axis.set_title("Vitória da equipe com maior ELO", loc="left", fontsize=14, fontweight="bold", color=COLORS["ink"])
    axis.set_ylabel("Taxa (%)")
    axis.set_ylim(0, max(75, values.max() + 8))
    axis.axhline(50, color=COLORS["muted"], linewidth=1, linestyle=(0, (5, 4)))
    axis.bar_label(bars, labels=[pt_pct(value) for value in values], padding=4, color=COLORS["ink"])
    _save(fig, destination)


def _model_page(connection: sqlite3.Connection, destination: Path) -> None:
    elo = _query(connection, "07_faixas_elo.sql")
    split = _query(connection, "08_holdout_treino.sql")
    fig = plt.figure(figsize=(16, 9), facecolor=COLORS["paper"])
    grid = fig.add_gridspec(1, 4, left=0.07, right=0.97, bottom=0.1, top=0.82, wspace=0.45)
    _title(fig, "CS2 Analytics | ELO e recortes", "Análise descritiva da base de modelagem; associação não implica causalidade.")

    axis = fig.add_subplot(grid[0, :2])
    _style_axis(axis)
    rates = 100 * elo["taxa_vitoria_maior_elo"]
    bars = axis.bar(elo["faixa_elo"], rates, color=RAMP, width=0.45)
    axis.set_title("Vantagem de ELO e resultado", loc="left", fontsize=14, fontweight="bold", color=COLORS["ink"])
    axis.set_xlabel("Diferença absoluta de ELO")
    axis.set_ylabel("Vitórias da equipe com maior ELO (%)")
    axis.set_ylim(0, 100)
    axis.axhline(50, color=COLORS["muted"], linewidth=1, linestyle=(0, (5, 4)))
    axis.bar_label(
        bars,
        labels=[f"{pt_pct(rate)}\n(n={pt_int(count)})" for rate, count in zip(rates, elo["partidas"], strict=True)],
        padding=4,
        fontsize=9,
        color=COLORS["ink2"],
    )

    names = {"train": "Treino", "holdout": "Holdout"}
    recortes = [names.get(value, value) for value in split["recorte"]]
    panels = [
        ("Diferença média de ELO", split["diferenca_elo_media"], pt_num),
        ("Partidas em LAN (%)", split["partidas_lan_pct"], pt_pct),
    ]
    for offset, (panel_title, values, fmt) in enumerate(panels):
        axis = fig.add_subplot(grid[0, 2 + offset])
        _style_axis(axis)
        bars = axis.bar(recortes, values, color=COLORS["s1"], width=0.45)
        axis.set_title(panel_title, loc="left", fontsize=12, fontweight="bold", color=COLORS["ink"])
        axis.set_ylim(0, float(values.max()) * 1.25)
        axis.bar_label(bars, labels=[fmt(value) for value in values], padding=3, fontsize=9, color=COLORS["ink2"])
    _save(fig, destination)


def _prediction_page(connection: sqlite3.Connection, destination: Path) -> None:
    predictions = pd.read_sql_query(
        "SELECT * FROM fact_prediction_holdout ORDER BY match_date_key, model_match_id",
        connection,
    )
    monthly = (
        predictions.assign(month=predictions["match_date_iso"].str[:7])
        .groupby("month", as_index=False)
        .agg(matches=("model_match_id", "size"), accuracy=("correct", "mean"))
    )
    context = (
        predictions.groupby("environment", as_index=False)
        .agg(matches=("model_match_id", "size"), accuracy=("correct", "mean"))
        .sort_values("environment")
    )
    elo = (
        predictions.groupby(["elo_band_sort", "elo_band"], as_index=False)
        .agg(matches=("model_match_id", "size"), accuracy=("correct", "mean"))
        .sort_values("elo_band_sort")
    )
    calibration = (
        predictions.groupby(["probability_band_sort", "probability_band"], as_index=False)
        .agg(
            matches=("model_match_id", "size"),
            predicted=("probability", "mean"),
            observed=("y_true", "mean"),
        )
        .sort_values("probability_band_sort")
    )
    errors = (
        predictions.loc[predictions["high_confidence_error"].eq(1)]
        .sort_values(["confidence", "match_date_key"], ascending=[False, True])
        .copy()
    )
    auc = rank_auc(predictions["y_true"], predictions["probability"])

    fig = plt.figure(figsize=(16, 9), facecolor=COLORS["paper"])
    outer = fig.add_gridspec(
        3,
        12,
        height_ratios=[0.9, 2.4, 2.2],
        left=0.055,
        right=0.97,
        bottom=0.07,
        top=0.84,
        hspace=0.48,
        wspace=0.75,
    )
    _title(
        fig,
        "CS2 Analytics | Desempenho do modelo",
        f"Resultado principal no holdout de {len(predictions)} partidas; filtros de contexto servem para diagnóstico descritivo.",
    )

    cards = [
        ("Partidas", pt_int(len(predictions))),
        ("ROC-AUC", pt_num(auc, 4)),
        ("Acurácia", pt_pct(100 * predictions["correct"].mean(), 2)),
        ("Erros com confiança >= 80%", pt_int(errors.shape[0])),
    ]
    for index, (label, value) in enumerate(cards):
        axis = fig.add_subplot(outer[0, index * 3 : (index + 1) * 3])
        axis.axis("off")
        axis.text(0.02, 0.58, value, fontsize=26, fontweight="bold", color=COLORS["accent"])
        axis.text(0.02, 0.18, label, fontsize=10, color=COLORS["ink2"])

    monthly_grid = outer[1, :7].subgridspec(2, 1, height_ratios=[1.5, 1], hspace=0.5)
    x = np.arange(len(monthly))

    axis = fig.add_subplot(monthly_grid[0])
    _style_axis(axis)
    axis.plot(x, 100 * monthly["accuracy"], color=COLORS["s1"], linewidth=2.5, marker="o")
    axis.set_ylim(0, 100)
    axis.set_xticks(x, monthly["month"])
    axis.set_ylabel("Acurácia (%)")
    axis.set_title("Acurácia e volume por mês", loc="left", fontsize=13, fontweight="bold", color=COLORS["ink"])
    axis.axhline(50, color=COLORS["muted"], linewidth=1, linestyle=(0, (5, 4)))
    for pos, value in zip(x, monthly["accuracy"], strict=True):
        axis.annotate(pt_pct(100 * value), (pos, 100 * value), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8, color=COLORS["ink2"])

    axis = fig.add_subplot(monthly_grid[1])
    _style_axis(axis)
    bars = axis.bar(x, monthly["matches"], color=COLORS["s1"], width=0.4)
    axis.set_xticks(x, monthly["month"])
    axis.set_ylabel("Partidas")
    axis.set_ylim(0, monthly["matches"].max() * 1.3)
    axis.bar_label(bars, labels=monthly["matches"].astype(str), padding=2, fontsize=8, color=COLORS["ink2"])

    axis = fig.add_subplot(outer[1, 8:12])
    axis.set_title("Matriz de confusão", loc="left", fontsize=13, fontweight="bold", color=COLORS["ink"], pad=10)
    matrix = pd.crosstab(predictions["y_true"], predictions["predicted_class"]).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    values = matrix.to_numpy()
    image = axis.imshow(values, cmap=CMAP_SEQ, vmin=0, vmax=values.max())
    axis.set_xticks([0, 1], ["Derrota", "Vitória"])
    axis.set_yticks([0, 1], ["Derrota", "Vitória"])
    axis.set_xlabel("Previsto")
    axis.set_ylabel("Real")
    for row in range(2):
        for column in range(2):
            value = int(values[row, column])
            cell_color = to_hex(CMAP_SEQ(value / values.max()))
            text_color = COLORS["ink"] if luminance(cell_color) > 0.4 else "white"
            axis.text(column, row, str(value), ha="center", va="center", fontsize=18, fontweight="bold", color=text_color)
    axis.figure.colorbar(image, ax=axis, fraction=0.045, pad=0.04)

    axis = fig.add_subplot(outer[2, :4])
    _style_axis(axis)
    labels = [*context["environment"], *elo["elo_band"].map(lambda value: f"ELO {value}")]
    rates = [*(100 * context["accuracy"]), *(100 * elo["accuracy"])]
    positions = np.arange(len(labels))
    bars = axis.barh(positions, rates, color=COLORS["s1"], height=0.55)
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(0, 100)
    axis.set_xlabel("Acurácia (%)")
    axis.axvline(50, color=COLORS["muted"], linewidth=1, linestyle=(0, (5, 4)))
    axis.set_title("Desempenho por contexto", loc="left", fontsize=13, fontweight="bold", color=COLORS["ink"])
    axis.bar_label(bars, labels=[pt_pct(value) for value in rates], padding=3, fontsize=8, color=COLORS["ink2"])

    axis = fig.add_subplot(outer[2, 4:8])
    _style_axis(axis)
    x = np.arange(len(calibration))
    axis.plot(x, 100 * calibration["predicted"], color=COLORS["s1"], marker="o", linewidth=2, label="Probabilidade média")
    axis.plot(x, 100 * calibration["observed"], color=COLORS["s2"], marker="o", linewidth=2, label="Vitória observada")
    axis.plot([0, len(x) - 1], [5, 95], color=COLORS["muted"], linestyle=(0, (5, 4)), linewidth=1)
    axis.set_xticks(x, calibration["probability_band"], rotation=35, ha="right", fontsize=7)
    axis.set_ylim(0, 100)
    axis.set_ylabel("Percentual")
    axis.set_title("Previsto versus observado", loc="left", fontsize=13, fontweight="bold", color=COLORS["ink"])
    axis.legend(frameon=False, fontsize=8, loc="upper left")

    axis = fig.add_subplot(outer[2, 8:12])
    axis.axis("off")
    axis.set_title("Erros de alta confiança", loc="left", fontsize=13, fontweight="bold", color=COLORS["ink"], pad=8)
    table_rows = [
        [
            row.match_date_iso[5:],
            row.team_name,
            row.opponent_name,
            pt_pct(100 * row.probability),
        ]
        for row in errors.itertuples()
    ]
    table = axis.table(
        cellText=table_rows,
        colLabels=["Data", "Equipe", "Adversário", "P(vitória)"],
        cellLoc="left",
        colLoc="left",
        bbox=[0, 0.05, 1, 0.82],
        colWidths=[0.13, 0.28, 0.35, 0.20],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor(COLORS["grid"])
        cell.set_facecolor("white" if row else PALETTE_LIGHT["seq"][0])
        if row == 0:
            cell.set_text_props(weight="bold", color=COLORS["ink"])

    _save(fig, destination)


def _write_html(output_dir: Path, pages: list[Path], data: dict) -> Path:
    content = render_index_html(data, [page.name for page in pages])
    path = output_dir / "index.html"
    path.write_text(content, encoding="utf-8")
    return path


def build_dashboard(
    database_path: Path | str = DATABASE_PATH,
    output_dir: Path | str = DASHBOARD_DIR,
) -> list[Path]:
    database_path = Path(database_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pages = [
        output_dir / "01_visao_geral.png",
        output_dir / "02_equipes.png",
        output_dir / "03_contexto_tempo.png",
        output_dir / "04_modelo_elo.png",
        output_dir / "05_desempenho_modelo.png",
    ]
    connection = sqlite3.connect(database_path)
    try:
        _overview_page(connection, pages[0])
        _teams_page(connection, pages[1])
        _context_page(connection, pages[2])
        _model_page(connection, pages[3])
        _prediction_page(connection, pages[4])
        data = load_dashboard_data(connection)
    finally:
        connection.close()
    return [*pages, _write_html(output_dir, pages, data)]


if __name__ == "__main__":
    generated = build_dashboard()
    print("Painéis gerados:")
    for path in generated:
        print(f"- {path}")
