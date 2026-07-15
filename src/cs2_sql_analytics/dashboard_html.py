"""Gera o dashboard HTML interativo em SVG proprio, sem dependencias externas.

Todos os graficos sao SVGs calculados em Python e embutidos em um unico
index.html com tema escuro/claro, tooltips e visao em tabela por grafico.
"""
from __future__ import annotations

import html
import json
import math
import sqlite3

import numpy as np
import pandas as pd

from .config import QUERY_DIR


PALETTE_LIGHT = {
    "s1": "#2a78d6", "s2": "#008300", "s3": "#e87ba4", "s4": "#eda100",
    "surface": "#fcfcfb", "page": "#f9f9f7", "ink": "#0b0b0b",
    "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
    "baseline": "#c3c2b7", "border": "rgba(11,11,11,0.10)",
    "ramp": ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
    "seq": ["#cde2fb", "#9ec5f4", "#6da7ec", "#2a78d6", "#184f95"],
}
PALETTE_DARK = {
    "s1": "#3987e5", "s2": "#008300", "s3": "#d55181", "s4": "#c98500",
    "surface": "#1a1a19", "page": "#0d0d0d", "ink": "#ffffff",
    "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
    "baseline": "#383835", "border": "rgba(255,255,255,0.10)",
    "ramp": ["#184f95", "#2a78d6", "#6da7ec", "#b7d3f6"],
    "seq": ["#184f95", "#256abf", "#3987e5", "#6da7ec", "#b7d3f6"],
}


def pt_int(value: float | int) -> str:
    return f"{int(value):,}".replace(",", ".")


def pt_num(value: float, decimals: int = 1) -> str:
    text = f"{value:,.{decimals}f}"
    return text.replace(",", "@").replace(".", ",").replace("@", ".")


def pt_pct(value: float, decimals: int = 1) -> str:
    return pt_num(value, decimals) + "%"


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def luminance(hex_color: str) -> float:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (0, 2, 4))

    def channel(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def rank_auc(y_true: pd.Series, probability: pd.Series) -> float:
    """ROC-AUC pela estatistica de Mann-Whitney, com rank medio nos empates."""
    frame = pd.DataFrame({"y": y_true, "p": probability})
    frame["rank"] = frame["p"].rank(method="average")
    positives = frame[frame["y"] == 1]
    n_pos = len(positives)
    n_neg = len(frame) - n_pos
    return float((positives["rank"].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def load_dashboard_data(connection: sqlite3.Connection) -> dict:
    def read(filename: str) -> pd.DataFrame:
        sql = (QUERY_DIR / filename).read_text(encoding="utf-8")
        return pd.read_sql_query(sql, connection)

    data: dict = {
        "overview": read("01_visao_geral.sql").iloc[0],
        "teams": read("02_desempenho_equipes.sql"),
        "monthly": read("03_evolucao_mensal.sql"),
        "context": read("04_contexto_lan_online.sql"),
        "formats": read("05_formatos_serie.sql"),
        "elo": read("07_faixas_elo.sql"),
        "split": read("08_holdout_treino.sql"),
    }
    predictions = pd.read_sql_query(
        "SELECT * FROM fact_prediction_holdout ORDER BY match_date_key, model_match_id",
        connection,
    )
    data["pred"] = predictions
    data["pred_monthly"] = (
        predictions.assign(mes=predictions["match_date_iso"].str[:7])
        .groupby("mes", as_index=False)
        .agg(partidas=("model_match_id", "size"), acuracia=("correct", "mean"))
    )
    data["pred_context"] = (
        predictions.groupby("environment", as_index=False)
        .agg(partidas=("model_match_id", "size"), acuracia=("correct", "mean"))
        .sort_values("environment")
    )
    data["pred_elo"] = (
        predictions.groupby(["elo_band_sort", "elo_band"], as_index=False)
        .agg(partidas=("model_match_id", "size"), acuracia=("correct", "mean"))
        .sort_values("elo_band_sort")
    )
    data["calibration"] = (
        predictions.groupby(["probability_band_sort", "probability_band"], as_index=False)
        .agg(
            partidas=("model_match_id", "size"),
            prevista=("probability", "mean"),
            observada=("y_true", "mean"),
        )
        .sort_values("probability_band_sort")
    )
    data["errors"] = (
        predictions.loc[predictions["high_confidence_error"].eq(1)]
        .sort_values(["confidence", "match_date_key"], ascending=[False, True])
        .copy()
    )
    matrix = pd.crosstab(predictions["y_true"], predictions["predicted_class"])
    data["confusion"] = matrix.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    data["auc"] = rank_auc(predictions["y_true"], predictions["probability"])
    data["accuracy"] = float(predictions["correct"].mean())
    return data


# ------------------------------------------------------------------ blocos

def nice_ticks(vmax: float, target: int = 5) -> list[float]:
    if vmax <= 0:
        return [0.0, 1.0]
    raw = vmax / target
    magnitude = 10 ** math.floor(math.log10(raw))
    step = magnitude
    for mult in (1, 2, 2.5, 5, 10):
        step = mult * magnitude
        if vmax / step <= target:
            break
    count = math.ceil(vmax / step)
    return [round(i * step, 10) for i in range(count + 1)]


def json_script(payload: dict, kind: str) -> str:
    text = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return f'<script type="application/json" data-{kind}>{text}</script>'


def bar_h(x: float, y: float, w: float, h: float, fill: str, extra: str = "") -> str:
    r = min(4.0, w / 2, h / 2)
    d = (
        f"M{x:.1f},{y:.1f} L{x + w - r:.1f},{y:.1f} "
        f"Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} "
        f"L{x + w:.1f},{y + h - r:.1f} "
        f"Q{x + w:.1f},{y + h:.1f} {x + w - r:.1f},{y + h:.1f} "
        f"L{x:.1f},{y + h:.1f} Z"
    )
    return f'<path d="{d}" fill="{fill}" {extra}/>'


def col_v(x: float, y: float, w: float, h: float, fill: str, extra: str = "") -> str:
    r = min(4.0, w / 2, h / 2)
    d = (
        f"M{x:.1f},{y + h:.1f} L{x:.1f},{y + r:.1f} "
        f"Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} "
        f"L{x + w - r:.1f},{y:.1f} "
        f"Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} "
        f"L{x + w:.1f},{y + h:.1f} Z"
    )
    return f'<path d="{d}" fill="{fill}" {extra}/>'


def tip_attr(title: str, rows: list[tuple[str, str]], key: str = "") -> str:
    rows_txt = ";;".join(f"{value}::{label}" for label, value in rows)
    key_attr = f' data-tip-key="{key}"' if key else ""
    return f'data-tip-title="{esc(title)}" data-tip-rows="{esc(rows_txt)}"{key_attr} tabindex="0"'


def grid_lines(fr: dict, ticks: list[float], fmt) -> str:
    parts = []
    vmax = ticks[-1] or 1
    for tick in ticks:
        y = fr["y0"] + fr["ph"] * (1 - tick / vmax)
        parts.append(
            f'<line x1="{fr["x0"]}" x2="{fr["x0"] + fr["pw"]}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>'
        )
        parts.append(
            f'<text x="{fr["x0"] - 8}" y="{y + 4:.1f}" class="tick" text-anchor="end">{fmt(tick)}</text>'
        )
    return "".join(parts)


def legend(items: list[tuple[str, str]], swatch: str = "rect") -> str:
    chips = []
    for label, color in items:
        mark = (
            f'<span class="key-line" style="background:{color}"></span>'
            if swatch == "line"
            else f'<span class="key-rect" style="background:{color}"></span>'
        )
        chips.append(f'<span class="key">{mark}{esc(label)}</span>')
    return f'<div class="legend">{"".join(chips)}</div>'


def card(span: int, title: str, note: str, body: str, table: str = "") -> str:
    return (
        f'<figure class="card span{span}"><figcaption><h3>{esc(title)}</h3>'
        f'<p class="note">{esc(note)}</p></figcaption>{body}{table}</figure>'
    )


def details_table(caption: str, headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f'<th scope="col">{esc(h)}</th>' for h in headers)
    body = "\n".join(
        "<tr>" + "".join(f"<td>{esc(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return (
        f'<details class="table-view"><summary>{esc(caption)}</summary>'
        f'<div class="table-scroll"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>\n{body}\n</tbody></table></div></details>"
    )


def kpi_tile(label: str, value: str, note: str = "") -> str:
    note_html = f'<span class="kpi-note">{esc(note)}</span>' if note else ""
    return (
        f'<div class="kpi"><span class="kpi-value">{esc(value)}</span>'
        f'<span class="kpi-label">{esc(label)}</span>{note_html}</div>'
    )


# ---------------------------------------------------------------- graficos

def chart_monthly_line(monthly: pd.DataFrame) -> str:
    w, h, ml, mr, mt, mb = 1120, 300, 56, 20, 16, 36
    fr = {"x0": ml, "y0": mt, "pw": w - ml - mr, "ph": h - mt - mb}
    values = monthly["partidas_unicas"].tolist()
    labels = monthly["mes"].tolist()
    n = len(values)
    ticks = nice_ticks(max(values))
    vmax = ticks[-1]
    xs = [ml + fr["pw"] * i / (n - 1) for i in range(n)]
    ys = [mt + fr["ph"] * (1 - v / vmax) for v in values]
    line = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(zip(xs, ys)))
    area = f"{line} L{xs[-1]:.1f},{mt + fr['ph']:.1f} L{xs[0]:.1f},{mt + fr['ph']:.1f} Z"
    tick_step = max(1, n // 8)
    xticks = "".join(
        f'<text x="{xs[i]:.1f}" y="{h - 12}" class="tick" text-anchor="{"end" if xs[i] > w - 44 else "middle"}">{esc(labels[i])}</text>'
        for i in range(0, n, tick_step)
    )
    payload = {
        "xs": [round(x, 1) for x in xs],
        "labels": labels,
        "series": [{
            "name": "Partidas únicas", "color": "var(--s1)",
            "values": [pt_int(v) for v in values],
            "ys": [round(y, 1) for y in ys],
        }],
    }
    svg = f"""<div class="chart" data-xh-chart>
<svg viewBox="0 0 {w} {h}" role="img" aria-label="Volume mensal de partidas únicas de {esc(labels[0])} a {esc(labels[-1])}">
{grid_lines(fr, ticks, pt_int)}
<path d="{area}" fill="var(--s1)" opacity="0.1"/>
<path d="{line}" fill="none" stroke="var(--s1)" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>
<circle cx="{xs[-1]:.1f}" cy="{ys[-1]:.1f}" r="4.5" fill="var(--s1)" stroke="var(--surface)" stroke-width="2"/>
{xticks}
<line x1="{ml}" x2="{w - mr}" y1="{mt + fr['ph']}" y2="{mt + fr['ph']}" class="axis"/>
<line class="xh-line" x1="0" x2="0" y1="{mt}" y2="{mt + fr['ph']}"/>
<g class="xh-dots"><circle class="xh-dot" r="4.5" fill="var(--s1)" stroke="var(--surface)" stroke-width="2"/></g>
</svg>
{json_script(payload, 'xh')}
</div>"""
    table = details_table(
        "Ver dados em tabela", ["Mês", "Partidas únicas", "LAN", "Online"],
        [[m.mes, pt_int(m.partidas_unicas), pt_int(m.partidas_lan), pt_int(m.partidas_online)]
         for m in monthly.itertuples()],
    )
    return card(12, "Volume mensal de partidas únicas",
                "Passe o mouse sobre o gráfico para ler o total de cada mês.", svg, table)


def chart_stacked_share(title: str, note: str, rows: list[tuple[str, int, str]],
                        total_label: str, total_value: int) -> str:
    w, h = 545, 96
    ml, mr, bar_y, bar_height = 8, 8, 46, 24
    pw = w - ml - mr
    total = sum(value for _, value, _ in rows)
    parts, x = [], float(ml)
    for index, (label, value, color) in enumerate(rows):
        seg_w = pw * value / total
        inset_l = 0 if index == 0 else 1
        inset_r = 1 if index < len(rows) - 1 else 0
        sx, sw = x + inset_l, max(seg_w - inset_l - inset_r, 0.5)
        pct = 100 * value / total
        tip = tip_attr(label, [("Participação", pt_pct(pct)), (total_label, pt_int(value))], color)
        if index == len(rows) - 1:
            parts.append(bar_h(sx, bar_y, sw, bar_height, color, f'class="mark" {tip}'))
        else:
            parts.append(f'<rect x="{sx:.1f}" y="{bar_y}" width="{sw:.1f}" height="{bar_height}" fill="{color}" class="mark" {tip}/>')
        label_text = f"{esc(label)} · {pt_pct(pct)}"
        if seg_w > 12 + 7.2 * len(label_text):
            parts.append(
                f'<text x="{sx + sw / 2:.1f}" y="{bar_y + bar_height / 2 + 4}" text-anchor="middle" '
                f'class="seg-label" style="fill:var(--on-{color[6:-1]})">{label_text}</text>'
            )
        x += seg_w
    svg = f"""<div class="chart">
<svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(title)}">
{''.join(parts)}
<text x="{ml}" y="{bar_y - 14}" class="tick">{esc(total_label)}: {pt_int(total_value)}</text>
</svg>
</div>"""
    chips = legend([(label, color) for label, _, color in rows])
    table = details_table(
        "Ver dados em tabela", ["Categoria", total_label, "Participação"],
        [[label, pt_int(value), pt_pct(100 * value / total)] for label, value, _ in rows],
    )
    return card(6, title, note, chips + svg, table)


def chart_teams_bar(teams: pd.DataFrame) -> str:
    top = teams.head(20).reset_index(drop=True)
    band, bar_th = 29, 16
    ml, mr, mt, mb = 118, 56, 8, 30
    w = 545
    h = mt + band * len(top) + mb
    pw = w - ml - mr
    parts = [
        f'<line x1="{ml + pw * p / 100:.1f}" x2="{ml + pw * p / 100:.1f}" y1="{mt}" y2="{mt + band * len(top)}" class="grid"/>'
        f'<text x="{ml + pw * p / 100:.1f}" y="{h - 10}" class="tick" text-anchor="middle">{p}%</text>'
        for p in (0, 25, 50, 75, 100)
    ]
    for i, row in top.iterrows():
        y = mt + i * band + (band - bar_th) / 2
        rate = 100 * row["taxa_vitoria"]
        bw = pw * rate / 100
        tip = tip_attr(row["equipe"], [
            ("Taxa de vitória", pt_pct(rate)),
            ("Partidas representadas", pt_int(row["partidas_representadas"])),
            ("Vitórias", pt_int(row["vitorias"])), ("Derrotas", pt_int(row["derrotas"])),
        ], "var(--s1)")
        parts.append(f'<rect x="0" y="{mt + i * band:.1f}" width="{w}" height="{band}" fill="transparent" class="row-hit" {tip}/>')
        parts.append(bar_h(ml, y, bw, bar_th, "var(--s1)", 'class="mark" pointer-events="none"'))
        parts.append(f'<text x="{ml - 8}" y="{y + bar_th - 4:.1f}" class="cat" text-anchor="end" pointer-events="none">{esc(row["equipe"])}</text>')
        parts.append(f'<text x="{ml + bw + 6:.1f}" y="{y + bar_th - 4:.1f}" class="val" pointer-events="none">{pt_pct(rate)}</text>')
    svg = f"""<div class="chart">
<svg viewBox="0 0 {w} {h}" role="img" aria-label="Vinte equipes com maior taxa de vitória">
{''.join(parts)}
</svg>
</div>"""
    table = details_table(
        "Ver as 95 equipes em tabela",
        ["#", "Equipe", "Partidas", "Vitórias", "Derrotas", "Taxa de vitória"],
        [[str(i + 1), r["equipe"], pt_int(r["partidas_representadas"]), pt_int(r["vitorias"]),
          pt_int(r["derrotas"]), pt_pct(100 * r["taxa_vitoria"], 2)]
         for i, r in teams.reset_index(drop=True).iterrows()],
    )
    return card(6, "Maior taxa de vitória (mínimo de 50 partidas)",
                "Top 20 do recorte de 95 equipes elegíveis; passe o mouse para ver vitórias e derrotas.", svg, table)


def chart_teams_scatter(teams: pd.DataFrame) -> str:
    w, h, ml, mr, mt, mb = 545, 610, 52, 16, 14, 40
    fr = {"x0": ml, "y0": mt, "pw": w - ml - mr, "ph": h - mt - mb}
    xmax_ticks = nice_ticks(teams["partidas_representadas"].max())
    xmax = xmax_ticks[-1]
    ymin, ymax = 25.0, 85.0

    def sx(v: float) -> float:
        return ml + fr["pw"] * v / xmax

    def sy(v: float) -> float:
        return mt + fr["ph"] * (1 - (v - ymin) / (ymax - ymin))

    parts = []
    for pct in range(30, 90, 10):
        parts.append(f'<line x1="{ml}" x2="{w - mr}" y1="{sy(pct):.1f}" y2="{sy(pct):.1f}" class="grid"/>')
        parts.append(f'<text x="{ml - 8}" y="{sy(pct) + 4:.1f}" class="tick" text-anchor="end">{pct}%</text>')
    for tick in xmax_ticks:
        parts.append(f'<text x="{sx(tick):.1f}" y="{h - 14}" class="tick" text-anchor="middle">{pt_int(tick)}</text>')
    points = []
    for _, row in teams.iterrows():
        rate = 100 * row["taxa_vitoria"]
        px, py = sx(row["partidas_representadas"]), sy(rate)
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="var(--s1)" stroke="var(--surface)" stroke-width="2" class="dot"/>')
        points.append({
            "x": round(px, 1), "y": round(py, 1), "title": row["equipe"],
            "rows": [[pt_pct(rate), "Taxa de vitória"],
                     [pt_int(row["partidas_representadas"]), "Partidas representadas"],
                     [pt_int(row["vitorias"]), "Vitórias"], [pt_int(row["derrotas"]), "Derrotas"]],
        })
    highlighted = teams.nlargest(3, "partidas_representadas")
    if teams.iloc[0]["equipe"] not in set(highlighted["equipe"]):
        highlighted = pd.concat([highlighted, teams.iloc[[0]]])
    for _, row in highlighted.iterrows():
        px, py = sx(row["partidas_representadas"]), sy(100 * row["taxa_vitoria"])
        if px > ml + fr["pw"] * 0.82:
            parts.append(f'<text x="{px - 9:.1f}" y="{py + 4:.1f}" class="anno" text-anchor="end">{esc(row["equipe"])}</text>')
        else:
            parts.append(f'<text x="{px + 9:.1f}" y="{py + 4:.1f}" class="anno">{esc(row["equipe"])}</text>')
    payload = {"points": points, "radius": 34}
    svg = f"""<div class="chart" id="teams-interactive-chart" data-nearest-chart>
<svg viewBox="0 0 {w} {h}" role="img" aria-label="Dispersão de volume de partidas por taxa de vitória das 95 equipes">
{''.join(parts)}
<line x1="{ml}" x2="{w - mr}" y1="{mt + fr['ph']}" y2="{mt + fr['ph']}" class="axis"/>
<text x="{ml + fr['pw'] / 2:.1f}" y="{h - 1}" class="tick" text-anchor="middle">Partidas representadas</text>
<circle class="nearest-halo" r="8" fill="none" stroke="var(--s1)" stroke-width="2" opacity="0"/>
</svg>
{json_script(payload, 'nearest')}
</div>"""
    return card(6, "Volume e taxa de vitória (95 equipes)",
                "Cada ponto é uma equipe; aproxime o cursor para identificar qualquer uma delas.", svg)


def chart_stacked_area(monthly: pd.DataFrame) -> str:
    w, h, ml, mr, mt, mb = 1120, 300, 56, 20, 16, 36
    fr = {"x0": ml, "y0": mt, "pw": w - ml - mr, "ph": h - mt - mb}
    labels = monthly["mes"].tolist()
    lan = monthly["partidas_lan"].tolist()
    online = monthly["partidas_online"].tolist()
    totals = [a + b for a, b in zip(lan, online)]
    ticks = nice_ticks(max(totals))
    vmax = ticks[-1]
    n = len(labels)
    xs = [ml + fr["pw"] * i / (n - 1) for i in range(n)]
    y_base = mt + fr["ph"]
    ys_lan = [mt + fr["ph"] * (1 - v / vmax) for v in lan]
    ys_tot = [mt + fr["ph"] * (1 - v / vmax) for v in totals]
    line_lan = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(zip(xs, ys_lan)))
    line_tot = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(zip(xs, ys_tot)))
    area_lan = f"{line_lan} L{xs[-1]:.1f},{y_base:.1f} L{xs[0]:.1f},{y_base:.1f} Z"
    area_online = f"{line_tot} L{xs[-1]:.1f},{ys_lan[-1]:.1f} " + " ".join(
        f"L{x:.1f},{y:.1f}" for x, y in zip(reversed(xs), reversed(ys_lan))
    ) + " Z"
    tick_step = max(1, n // 8)
    xticks = "".join(
        f'<text x="{xs[i]:.1f}" y="{h - 12}" class="tick" text-anchor="{"end" if xs[i] > w - 44 else "middle"}">{esc(labels[i])}</text>'
        for i in range(0, n, tick_step)
    )
    payload = {
        "xs": [round(x, 1) for x in xs], "labels": labels,
        "series": [
            {"name": "LAN", "color": "var(--s1)", "values": [pt_int(v) for v in lan],
             "ys": [round(y, 1) for y in ys_lan]},
            {"name": "Online", "color": "var(--s2)", "values": [pt_int(v) for v in online],
             "ys": [round(y, 1) for y in ys_tot]},
            {"name": "Total", "color": "var(--muted)", "values": [pt_int(v) for v in totals],
             "ys": [round(y, 1) for y in ys_tot]},
        ],
    }
    svg = f"""<div class="chart" data-xh-chart>
<svg viewBox="0 0 {w} {h}" role="img" aria-label="Composição mensal de partidas por ambiente, LAN e online">
{grid_lines(fr, ticks, pt_int)}
<path d="{area_lan}" fill="var(--s1)" opacity="0.12"/>
<path d="{area_online}" fill="var(--s2)" opacity="0.12"/>
<path d="{line_lan}" fill="none" stroke="var(--s1)" stroke-width="2" stroke-linejoin="round"/>
<path d="{line_tot}" fill="none" stroke="var(--s2)" stroke-width="2" stroke-linejoin="round"/>
{xticks}
<line x1="{ml}" x2="{w - mr}" y1="{y_base}" y2="{y_base}" class="axis"/>
<line class="xh-line" x1="0" x2="0" y1="{mt}" y2="{y_base}"/>
<g class="xh-dots"><circle class="xh-dot" r="4.5" fill="var(--s1)" stroke="var(--surface)" stroke-width="2"/>
<circle class="xh-dot" r="4.5" fill="var(--s2)" stroke="var(--surface)" stroke-width="2"/>
<circle class="xh-dot" r="0" fill="none"/></g>
</svg>
{json_script(payload, 'xh')}
</div>"""
    chips = legend([("LAN (faixa inferior)", "var(--s1)"), ("Online (faixa superior, empilhada)", "var(--s2)")])
    table = details_table(
        "Ver dados em tabela", ["Mês", "LAN", "Online", "Total"],
        [[m.mes, pt_int(m.partidas_lan), pt_int(m.partidas_online), pt_int(m.partidas_lan + m.partidas_online)]
         for m in monthly.itertuples()],
    )
    return card(12, "Composição mensal por ambiente",
                "Áreas empilhadas: a faixa superior soma LAN + online. LAN e online descrevem o ambiente da partida.",
                chips + svg, table)


def chart_two_cols(title: str, note: str, cats: list[str], values: list[float], fmt,
                   ref: float | None = None, aria: str = "",
                   table_headers: list[str] | None = None,
                   table_rows: list[list[str]] | None = None) -> str:
    w, h, ml, mr, mt, mb = 545, 240, 52, 16, 18, 34
    fr = {"x0": ml, "y0": mt, "pw": w - ml - mr, "ph": h - mt - mb}
    ticks = nice_ticks(max(values) * 1.15)
    vmax = ticks[-1]
    band = fr["pw"] / len(cats)
    col_w = 24.0
    parts = [grid_lines(fr, ticks, lambda t: pt_int(t) if vmax > 10 else pt_num(t))]
    for i, (cat, val) in enumerate(zip(cats, values)):
        cx = ml + band * i + band / 2
        ch = fr["ph"] * val / vmax
        y = mt + fr["ph"] - ch
        tip = tip_attr(cat, [(title, fmt(val))], "var(--s1)")
        parts.append(col_v(cx - col_w / 2, y, col_w, ch, "var(--s1)", f'class="mark" {tip}'))
        parts.append(f'<text x="{cx:.1f}" y="{y - 7:.1f}" class="val" text-anchor="middle" pointer-events="none">{fmt(val)}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{h - 12}" class="cat" text-anchor="middle">{esc(cat)}</text>')
    if ref is not None:
        ry = mt + fr["ph"] * (1 - ref / vmax)
        parts.append(f'<line x1="{ml}" x2="{w - mr}" y1="{ry:.1f}" y2="{ry:.1f}" class="refline"/>')
        parts.append(f'<text x="{w - mr}" y="{ry - 5:.1f}" class="anno" text-anchor="end">{pt_int(ref)}%</text>')
    parts.append(f'<line x1="{ml}" x2="{w - mr}" y1="{mt + fr["ph"]}" y2="{mt + fr["ph"]}" class="axis"/>')
    svg = f'<div class="chart"><svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(aria or title)}">{"".join(parts)}</svg></div>'
    table = details_table("Ver dados em tabela", table_headers, table_rows) if table_headers else ""
    return card(6, title, note, svg, table)


def chart_elo_bands(elo: pd.DataFrame) -> str:
    w, h, ml, mr, mt, mb = 545, 260, 52, 16, 18, 48
    fr = {"x0": ml, "y0": mt, "pw": w - ml - mr, "ph": h - mt - mb}
    vmax = 100.0
    band = fr["pw"] / len(elo)
    col_w = 24.0
    parts = []
    for pct in (0, 25, 50, 75, 100):
        y = mt + fr["ph"] * (1 - pct / vmax)
        parts.append(f'<line x1="{ml}" x2="{w - mr}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{ml - 8}" y="{y + 4:.1f}" class="tick" text-anchor="end">{pct}%</text>')
    for i, row in elo.reset_index(drop=True).iterrows():
        rate = 100 * row["taxa_vitoria_maior_elo"]
        cx = ml + band * i + band / 2
        ch = fr["ph"] * rate / vmax
        y = mt + fr["ph"] - ch
        tip = tip_attr(f"Diferença de ELO {row['faixa_elo']}", [
            ("Vitória do maior ELO", pt_pct(rate)), ("Partidas", pt_int(row["partidas"])),
            ("Diferença média", pt_num(row["diferenca_elo_media"])),
        ], f"var(--r{i + 1})")
        parts.append(col_v(cx - col_w / 2, y, col_w, ch, f"var(--r{i + 1})", f'class="mark" {tip}'))
        parts.append(f'<text x="{cx:.1f}" y="{y - 7:.1f}" class="val" text-anchor="middle" pointer-events="none">{pt_pct(rate)}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{h - 26}" class="cat" text-anchor="middle">{esc(row["faixa_elo"])}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{h - 11}" class="tick" text-anchor="middle">n={pt_int(row["partidas"])}</text>')
    ry = mt + fr["ph"] * 0.5
    parts.append(f'<line x1="{ml}" x2="{w - mr}" y1="{ry:.1f}" y2="{ry:.1f}" class="refline"/>')
    parts.append(f'<text x="{w - mr}" y="{ry - 5:.1f}" class="anno" text-anchor="end">50%</text>')
    parts.append(f'<line x1="{ml}" x2="{w - mr}" y1="{mt + fr["ph"]}" y2="{mt + fr["ph"]}" class="axis"/>')
    svg = f'<div class="chart"><svg viewBox="0 0 {w} {h}" role="img" aria-label="Taxa de vitória da equipe de maior ELO por faixa de diferença">{"".join(parts)}</svg></div>'
    table = details_table(
        "Ver dados em tabela", ["Faixa de diferença de ELO", "Partidas", "Vitória do maior ELO"],
        [[r["faixa_elo"], pt_int(r["partidas"]), pt_pct(100 * r["taxa_vitoria_maior_elo"])]
         for _, r in elo.iterrows()],
    )
    return card(6, "Vantagem de ELO e resultado",
                "Rampa ordinal: quanto maior a diferença de ELO pré-jogo, maior a taxa de vitória do favorito.", svg, table)


def chart_split_multiples(split: pd.DataFrame) -> str:
    w, h = 545, 240
    panel_w = (w - 24) / 2
    names = {"train": "Treino", "holdout": "Holdout"}
    cats = [names.get(r, r) for r in split["recorte"]]
    panels = []
    specs = [
        ("Diferença média de ELO", split["diferenca_elo_media"].tolist(), pt_num),
        ("Partidas em LAN (%)", split["partidas_lan_pct"].tolist(), pt_pct),
    ]
    for p, (ptitle, values, fmt) in enumerate(specs):
        ox = p * (panel_w + 24)
        ml, mr, mt, mb = 44, 8, 34, 30
        pw, ph = panel_w - ml - mr, h - mt - mb
        ticks = nice_ticks(max(values) * 1.2, 4)
        vmax = ticks[-1]
        band = pw / len(cats)
        parts = [f'<text x="{ox + ml}" y="18" class="panel-title">{esc(ptitle)}</text>']
        for tick in ticks:
            y = mt + ph * (1 - tick / vmax)
            parts.append(f'<line x1="{ox + ml}" x2="{ox + ml + pw}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
            parts.append(f'<text x="{ox + ml - 6}" y="{y + 4:.1f}" class="tick" text-anchor="end">{pt_int(tick)}</text>')
        for i, (cat, val) in enumerate(zip(cats, values)):
            cx = ox + ml + band * i + band / 2
            ch = ph * val / vmax
            y = mt + ph - ch
            tip = tip_attr(cat, [(ptitle, fmt(val))], "var(--s1)")
            parts.append(col_v(cx - 12, y, 24, ch, "var(--s1)", f'class="mark" {tip}'))
            parts.append(f'<text x="{cx:.1f}" y="{y - 7:.1f}" class="val" text-anchor="middle" pointer-events="none">{fmt(val)}</text>')
            parts.append(f'<text x="{cx:.1f}" y="{h - 10}" class="cat" text-anchor="middle">{esc(cat)}</text>')
        parts.append(f'<line x1="{ox + ml}" x2="{ox + ml + pw}" y1="{mt + ph}" y2="{mt + ph}" class="axis"/>')
        panels.append("".join(parts))
    svg = f'<div class="chart"><svg viewBox="0 0 {w} {h}" role="img" aria-label="Perfil comparado dos recortes de treino e holdout">{"".join(panels)}</svg></div>'
    table = details_table(
        "Ver dados em tabela", ["Recorte", "Partidas", "Diferença média de ELO", "Partidas em LAN"],
        [[names.get(r["recorte"], r["recorte"]), pt_int(r["partidas"]),
          pt_num(r["diferenca_elo_media"]), pt_pct(r["partidas_lan_pct"])]
         for _, r in split.iterrows()],
    )
    return card(6, "Perfil de treino e holdout",
                "Medidas de escalas diferentes ficam em painéis separados (small multiples), nunca no mesmo eixo.", svg, table)


def chart_accuracy_volume(pred_monthly: pd.DataFrame) -> str:
    w, h = 545, 300
    ml, mr = 52, 16
    top = {"mt": 20, "ph": 130}
    bottom = {"mt": 186, "ph": 74}
    pw = w - ml - mr
    labels = pred_monthly["mes"].tolist()
    acc = (100 * pred_monthly["acuracia"]).tolist()
    vol = pred_monthly["partidas"].tolist()
    n = len(labels)
    xs = [ml + pw * (i + 0.5) / n for i in range(n)]
    parts = [f'<text x="{ml}" y="14" class="panel-title">Acurácia no mês (%)</text>']
    for pct in (0, 50, 100):
        y = top["mt"] + top["ph"] * (1 - pct / 100)
        cls = "refline" if pct == 50 else "grid"
        parts.append(f'<line x1="{ml}" x2="{w - mr}" y1="{y:.1f}" y2="{y:.1f}" class="{cls}"/>')
        parts.append(f'<text x="{ml - 8}" y="{y + 4:.1f}" class="tick" text-anchor="end">{pct}%</text>')
    ys = [top["mt"] + top["ph"] * (1 - a / 100) for a in acc]
    line = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(zip(xs, ys)))
    parts.append(f'<path d="{line}" fill="none" stroke="var(--s1)" stroke-width="2" stroke-linejoin="round"/>')
    for x, y, a, m, v in zip(xs, ys, acc, labels, vol):
        tip = tip_attr(m, [("Acurácia", pt_pct(a)), ("Partidas", pt_int(v))], "var(--s1)")
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="var(--s1)" stroke="var(--surface)" stroke-width="2" class="mark" {tip}/>')
        parts.append(f'<text x="{x:.1f}" y="{y - 11:.1f}" class="val" text-anchor="middle" pointer-events="none">{pt_pct(a)}</text>')
    parts.append(f'<text x="{ml}" y="{bottom["mt"] - 8}" class="panel-title">Partidas avaliadas</text>')
    vmax = nice_ticks(max(vol), 3)[-1]
    for x, m, v in zip(xs, labels, vol):
        ch = bottom["ph"] * v / vmax
        y = bottom["mt"] + bottom["ph"] - ch
        tip = tip_attr(m, [("Partidas", pt_int(v))], "var(--s1)")
        parts.append(col_v(x - 12, y, 24, ch, "var(--s1)", f'class="mark" {tip}'))
        parts.append(f'<text x="{x:.1f}" y="{y - 6:.1f}" class="val" text-anchor="middle" pointer-events="none">{pt_int(v)}</text>')
        parts.append(f'<text x="{x:.1f}" y="{h - 10}" class="cat" text-anchor="middle">{esc(m)}</text>')
    base_y = bottom["mt"] + bottom["ph"]
    parts.append(f'<line x1="{ml}" x2="{w - mr}" y1="{base_y}" y2="{base_y}" class="axis"/>')
    svg = f'<div class="chart"><svg viewBox="0 0 {w} {h}" role="img" aria-label="Acurácia mensal do modelo e volume de partidas avaliadas, em dois painéis empilhados">{"".join(parts)}</svg></div>'
    table = details_table(
        "Ver dados em tabela", ["Mês", "Partidas", "Acurácia"],
        [[r["mes"], pt_int(r["partidas"]), pt_pct(100 * r["acuracia"])] for _, r in pred_monthly.iterrows()],
    )
    return card(6, "Acurácia e volume por mês",
                "Dois painéis com o mesmo eixo de meses — sem eixo duplo, que induz correlações artificiais.", svg, table)


def chart_confusion(confusion: pd.DataFrame) -> str:
    w, h = 545, 300
    ox, oy, cell, gap = 150, 56, 118, 2
    seq_steps = 5
    total = int(confusion.to_numpy().sum())
    vmax = confusion.to_numpy().max()
    names = ["Derrota", "Vitória"]
    parts = [
        f'<text x="{ox + cell + gap / 2:.1f}" y="{oy - 34}" class="panel-title" text-anchor="middle">Previsto</text>',
        f'<text x="{ox - 96}" y="{oy + cell + gap / 2:.1f}" class="panel-title" transform="rotate(-90 {ox - 96} {oy + cell + gap / 2:.1f})" text-anchor="middle">Real</text>',
    ]
    for j, name in enumerate(names):
        parts.append(f'<text x="{ox + j * (cell + gap) + cell / 2:.1f}" y="{oy - 12}" class="cat" text-anchor="middle">{name}</text>')
        parts.append(f'<text x="{ox - 10}" y="{oy + j * (cell + gap) + cell / 2 + 4:.1f}" class="cat" text-anchor="end">{name}</text>')
    for i in (0, 1):
        for j in (0, 1):
            value = int(confusion.iloc[i, j])
            step = max(1, math.ceil(seq_steps * value / vmax)) if vmax else 1
            x, y = ox + j * (cell + gap), oy + i * (cell + gap)
            tip = tip_attr(f"Real {names[i]} × previsto {names[j]}", [
                ("Partidas", pt_int(value)), ("Fatia do holdout", pt_pct(100 * value / total)),
                ("Tipo", "Acerto" if i == j else "Erro"),
            ], f"var(--q{step})")
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="4" fill="var(--q{step})" class="mark" {tip}/>')
            parts.append(
                f'<text x="{x + cell / 2:.1f}" y="{y + cell / 2 + 2:.1f}" text-anchor="middle" class="cell-value" '
                f'style="fill:var(--on-q{step})" pointer-events="none">{value}</text>'
            )
            parts.append(
                f'<text x="{x + cell / 2:.1f}" y="{y + cell / 2 + 22:.1f}" text-anchor="middle" class="cell-sub" '
                f'style="fill:var(--on-q{step})" pointer-events="none">{pt_pct(100 * value / total)}</text>'
            )
    svg = f'<div class="chart"><svg viewBox="0 0 {w} {h}" role="img" aria-label="Matriz de confusão do modelo no holdout">{"".join(parts)}</svg></div>'
    table = details_table(
        "Ver dados em tabela", ["", "Previsto derrota", "Previsto vitória"],
        [["Real derrota", pt_int(confusion.iloc[0, 0]), pt_int(confusion.iloc[0, 1])],
         ["Real vitória", pt_int(confusion.iloc[1, 0]), pt_int(confusion.iloc[1, 1])]],
    )
    return card(6, "Matriz de confusão",
                "Diagonal principal concentra os acertos; intensidade segue o número de partidas.", svg, table)


def chart_context_accuracy(pred_context: pd.DataFrame, pred_elo: pd.DataFrame) -> str:
    rows = [(r["environment"], 100 * r["acuracia"], int(r["partidas"])) for _, r in pred_context.iterrows()]
    rows += [(f"ELO {r['elo_band']}", 100 * r["acuracia"], int(r["partidas"])) for _, r in pred_elo.iterrows()]
    band, bar_th = 34, 16
    ml, mr, mt, mb = 118, 56, 8, 30
    w = 545
    h = mt + band * len(rows) + mb
    pw = w - ml - mr
    parts = [
        f'<line x1="{ml + pw * p / 100:.1f}" x2="{ml + pw * p / 100:.1f}" y1="{mt}" y2="{mt + band * len(rows)}" class="{"refline" if p == 50 else "grid"}"/>'
        f'<text x="{ml + pw * p / 100:.1f}" y="{h - 10}" class="tick" text-anchor="middle">{p}%</text>'
        for p in (0, 25, 50, 75, 100)
    ]
    for i, (label, rate, n) in enumerate(rows):
        y = mt + i * band + (band - bar_th) / 2
        bw = pw * rate / 100
        tip = tip_attr(label, [("Acurácia", pt_pct(rate)), ("Partidas", pt_int(n))], "var(--s1)")
        parts.append(f'<rect x="0" y="{mt + i * band:.1f}" width="{w}" height="{band}" fill="transparent" class="row-hit" {tip}/>')
        parts.append(bar_h(ml, y, bw, bar_th, "var(--s1)", 'class="mark" pointer-events="none"'))
        parts.append(f'<text x="{ml - 8}" y="{y + bar_th - 4:.1f}" class="cat" text-anchor="end" pointer-events="none">{esc(label)}</text>')
        parts.append(f'<text x="{ml + bw + 6:.1f}" y="{y + bar_th - 4:.1f}" class="val" pointer-events="none">{pt_pct(rate)}</text>')
    svg = f'<div class="chart"><svg viewBox="0 0 {w} {h}" role="img" aria-label="Acurácia do modelo por ambiente e por faixa de ELO">{"".join(parts)}</svg></div>'
    table = details_table(
        "Ver dados em tabela", ["Recorte", "Partidas", "Acurácia"],
        [[label, pt_int(n), pt_pct(rate)] for label, rate, n in rows],
    )
    return card(6, "Acurácia por contexto",
                "Diagnóstico descritivo do holdout: ambiente e faixa de diferença de ELO. Linha de referência em 50%.", svg, table)


def chart_calibration(calibration: pd.DataFrame) -> str:
    w, h, ml, mr, mt, mb = 545, 300, 52, 16, 18, 52
    fr = {"pw": w - ml - mr, "ph": h - mt - mb}
    labels = calibration["probability_band"].tolist()
    n = len(labels)
    xs = [ml + fr["pw"] * (i + 0.5) / n for i in range(n)]

    def sy(pct: float) -> float:
        return mt + fr["ph"] * (1 - pct / 100)

    parts = []
    for pct in (0, 25, 50, 75, 100):
        parts.append(f'<line x1="{ml}" x2="{w - mr}" y1="{sy(pct):.1f}" y2="{sy(pct):.1f}" class="grid"/>')
        parts.append(f'<text x="{ml - 8}" y="{sy(pct) + 4:.1f}" class="tick" text-anchor="end">{pct}%</text>')
    diag = f"M{xs[0]:.1f},{sy(5):.1f} L{xs[-1]:.1f},{sy(85):.1f}"
    parts.append(f'<path d="{diag}" class="refline" fill="none"/>')
    series = [("Probabilidade média prevista", "var(--s1)", (100 * calibration["prevista"]).tolist()),
              ("Vitória observada", "var(--s2)", (100 * calibration["observada"]).tolist())]
    payload_series = []
    for name, color, values in series:
        ys = [sy(v) for v in values]
        line = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(zip(xs, ys)))
        parts.append(f'<path d="{line}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"/>')
        for x, y in zip(xs, ys):
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" stroke="var(--surface)" stroke-width="2"/>')
        payload_series.append({"name": name, "color": color,
                               "values": [pt_pct(v) for v in values], "ys": [round(y, 1) for y in ys]})
    payload_series.append({"name": "Partidas na faixa", "color": "var(--muted)",
                           "values": [pt_int(v) for v in calibration["partidas"]],
                           "ys": [sy(0) for _ in labels]})
    for i, label in enumerate(labels):
        parts.append(f'<text x="{xs[i]:.1f}" y="{h - 30}" class="tick" text-anchor="middle">{esc(label.replace("%", ""))}</text>')
    parts.append(f'<text x="{ml + fr["pw"] / 2:.1f}" y="{h - 10}" class="tick" text-anchor="middle">Faixa de probabilidade prevista (%)</text>')
    parts.append(f'<line x1="{ml}" x2="{w - mr}" y1="{sy(0)}" y2="{sy(0)}" class="axis"/>')
    payload = {"xs": [round(x, 1) for x in xs], "labels": labels, "series": payload_series}
    svg = f"""<div class="chart" data-xh-chart>
<svg viewBox="0 0 {w} {h}" role="img" aria-label="Curva de calibração: probabilidade prevista versus vitória observada">
{''.join(parts)}
<line class="xh-line" x1="0" x2="0" y1="{mt}" y2="{sy(0)}"/>
<g class="xh-dots"><circle class="xh-dot" r="4.5" fill="var(--s1)" stroke="var(--surface)" stroke-width="2"/>
<circle class="xh-dot" r="4.5" fill="var(--s2)" stroke="var(--surface)" stroke-width="2"/>
<circle class="xh-dot" r="0" fill="none"/></g>
</svg>
{json_script(payload, 'xh')}
</div>"""
    chips = legend([("Probabilidade média prevista", "var(--s1)"), ("Vitória observada", "var(--s2)")], swatch="line")
    table = details_table(
        "Ver dados em tabela", ["Faixa prevista", "Partidas", "Probabilidade média", "Vitória observada"],
        [[r["probability_band"], pt_int(r["partidas"]), pt_pct(100 * r["prevista"]), pt_pct(100 * r["observada"])]
         for _, r in calibration.iterrows()],
    )
    return card(6, "Previsto versus observado",
                "Quanto mais próximas as linhas ficam da diagonal, melhor calibrado o modelo.", chips + svg, table)


def errors_table_card(errors: pd.DataFrame) -> str:
    rows = "\n".join(
        "<tr>"
        f"<td>{esc(r.match_date_iso)}</td><td>{esc(r.team_name)}</td>"
        f"<td>{esc(r.opponent_name)}</td>"
        f"<td>{'Vitória' if r.predicted_class == 1 else 'Derrota'}</td>"
        f'<td class="num">{pt_pct(100 * r.confidence)}</td>'
        "</tr>"
        for r in errors.itertuples()
    )
    body = (
        '<div class="table-scroll"><table class="plain-table">'
        '<thead><tr><th scope="col">Data</th><th scope="col">Equipe analisada</th>'
        '<th scope="col">Adversário</th><th scope="col">Previsão</th>'
        '<th scope="col" class="num">Confiança</th></tr></thead>'
        f"<tbody>\n{rows}\n</tbody></table></div>"
    )
    return card(6, "Erros de alta confiança",
                "Previsões com confiança de pelo menos 80% que não coincidiram com o resultado.", body)


# ------------------------------------------------------------------ pagina

def theme_css() -> str:
    def block(p: dict) -> str:
        def on(hex_color: str) -> str:
            return "#0b0b0b" if luminance(hex_color) > 0.4 else "#ffffff"

        lines = [
            f"--s1:{p['s1']};--s2:{p['s2']};--s3:{p['s3']};--s4:{p['s4']};",
            f"--surface:{p['surface']};--page:{p['page']};--ink:{p['ink']};",
            f"--ink2:{p['ink2']};--muted:{p['muted']};--grid:{p['grid']};",
            f"--baseline:{p['baseline']};--border:{p['border']};",
        ]
        for i, hex_color in enumerate(p["ramp"], start=1):
            lines.append(f"--r{i}:{hex_color};--on-r{i}:{on(hex_color)};")
        for i, hex_color in enumerate(p["seq"], start=1):
            lines.append(f"--q{i}:{hex_color};--on-q{i}:{on(hex_color)};")
        for slot in ("s1", "s2", "s3", "s4"):
            lines.append(f"--on-{slot}:{on(p[slot])};")
        return "".join(lines)

    return (
        f":root{{{block(PALETTE_DARK)}color-scheme:dark;}}\n"
        f':root[data-theme="light"]{{{block(PALETTE_LIGHT)}color-scheme:light;}}\n'
    )


PAGE_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth;scroll-padding-top:76px}
body{background:var(--page);color:var(--ink);font-family:system-ui,-apple-system,'Segoe UI',sans-serif;
  font-size:15px;line-height:1.5;-webkit-font-smoothing:antialiased}
a{color:var(--s1);text-decoration:none}
a:hover{text-decoration:underline}
header.top{position:sticky;top:0;z-index:40;display:flex;align-items:center;gap:20px;
  padding:0 clamp(16px,4vw,40px);height:56px;background:color-mix(in srgb,var(--page) 88%,transparent);
  backdrop-filter:blur(10px);border-bottom:1px solid var(--border)}
.brand{font-weight:700;font-size:15px;letter-spacing:.04em;white-space:nowrap}
nav.sections{display:flex;gap:4px;overflow-x:auto;scrollbar-width:none;flex:1}
nav.sections a{color:var(--ink2);font-size:13.5px;padding:6px 12px;border-radius:8px;white-space:nowrap}
nav.sections a:hover{color:var(--ink);text-decoration:none;background:color-mix(in srgb,var(--ink) 6%,transparent)}
nav.sections a.active{color:var(--ink);background:color-mix(in srgb,var(--s1) 14%,transparent)}
.top-actions{display:flex;align-items:center;gap:8px}
#theme-toggle{border:1px solid var(--border);background:var(--surface);color:var(--ink);
  border-radius:8px;padding:6px 12px;font:inherit;font-size:13px;cursor:pointer}
#theme-toggle:hover{border-color:var(--muted)}
.gh-link{font-size:13px;color:var(--ink2)}
.hero{padding:clamp(28px,5vw,52px) clamp(16px,4vw,40px) 8px;max-width:1240px;margin:0 auto}
.hero h1{font-size:clamp(26px,4vw,36px);font-weight:700;letter-spacing:-.01em}
.hero p.sub{color:var(--ink2);max-width:820px;margin-top:8px}
.hero .period{display:inline-block;margin-top:14px;font-size:12.5px;color:var(--ink2);
  border:1px solid var(--border);border-radius:999px;padding:4px 12px}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;max-width:1240px;
  margin:20px auto 0;padding:0 clamp(16px,4vw,40px)}
.kpi{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:18px 20px;
  display:flex;flex-direction:column;gap:2px}
.kpi-value{font-size:30px;font-weight:600;letter-spacing:-.01em}
.kpi-label{font-size:13px;color:var(--ink2)}
.kpi-note{font-size:12px;color:var(--muted)}
main{max-width:1240px;margin:0 auto;padding:8px clamp(16px,4vw,40px) 48px}
section.block{padding-top:34px}
section.block>h2{font-size:20px;font-weight:700;display:flex;align-items:center;gap:10px}
section.block>h2::before{content:"";width:5px;height:20px;border-radius:3px;background:var(--s1)}
section.block>p.lead{color:var(--ink2);font-size:13.5px;margin:6px 0 16px;max-width:860px}
.cards{display:grid;grid-template-columns:repeat(12,1fr);gap:16px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;
  padding:18px 20px 14px;min-width:0}
.card:hover{border-color:color-mix(in srgb,var(--muted) 55%,var(--border))}
.span6{grid-column:span 6}.span12{grid-column:span 12}
.card h3{font-size:15px;font-weight:600}
.card p.note{font-size:12.5px;color:var(--ink2);margin:3px 0 12px}
.chart{position:relative}
.chart svg{display:block;width:100%;height:auto}
svg text{font-family:system-ui,-apple-system,'Segoe UI',sans-serif}
svg .grid{stroke:var(--grid);stroke-width:1;fill:none}
.axis{stroke:var(--baseline);stroke-width:1}
.refline{stroke:var(--muted);stroke-width:1;stroke-dasharray:5 4;fill:none}
.tick{font-size:11.5px;fill:var(--muted);font-variant-numeric:tabular-nums}
.cat{font-size:12px;fill:var(--ink2)}
.val{font-size:11.5px;font-weight:600;fill:var(--ink2);font-variant-numeric:tabular-nums}
.anno{font-size:11.5px;fill:var(--ink2)}
.panel-title{font-size:12.5px;font-weight:600;fill:var(--ink2)}
.seg-label{font-size:12px;font-weight:600}
.cell-value{font-size:30px;font-weight:600}
.cell-sub{font-size:12.5px;opacity:.85}
.mark{transition:filter .1s}
.mark:hover,.row-hit:hover+path.mark,.mark:focus-visible{filter:brightness(1.18)}
.mark:focus-visible,.row-hit:focus-visible{outline:1.5px solid var(--s1);outline-offset:2px}
.dot{pointer-events:none}
.xh-line{stroke:var(--muted);stroke-width:1;opacity:0}
.xh-dot{opacity:0}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin:0 0 10px}
.key{display:inline-flex;align-items:center;gap:7px;font-size:12.5px;color:var(--ink2)}
.key-rect{width:12px;height:12px;border-radius:3px;display:inline-block}
.key-line{width:16px;height:3px;border-radius:2px;display:inline-block}
#tooltip{position:fixed;z-index:60;pointer-events:none;background:var(--surface);
  border:1px solid var(--border);border-radius:10px;padding:10px 12px;font-size:12.5px;
  box-shadow:0 6px 24px rgba(0,0,0,.28);opacity:0;transition:opacity .08s;max-width:280px}
#tooltip .tt-title{font-weight:600;color:var(--ink);margin-bottom:4px}
#tooltip .tt-row{display:flex;align-items:center;gap:7px;color:var(--ink2);margin-top:2px}
#tooltip .tt-key{width:12px;height:3px;border-radius:2px;flex:none}
#tooltip .tt-val{font-weight:600;color:var(--ink);font-variant-numeric:tabular-nums}
.table-view{margin-top:10px;border-top:1px solid var(--border);padding-top:8px}
.table-view summary{font-size:12.5px;color:var(--ink2);cursor:pointer;user-select:none}
.table-view summary:hover{color:var(--ink)}
.table-scroll{overflow-x:auto;max-height:340px;overflow-y:auto;margin-top:8px}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th{text-align:left;color:var(--ink2);font-weight:600;padding:6px 10px;border-bottom:1px solid var(--baseline);
  position:sticky;top:0;background:var(--surface)}
td{padding:5px 10px;border-bottom:1px solid var(--grid);color:var(--ink2)}
td:first-child{color:var(--ink)}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.plain-table td,.plain-table th{font-size:13px}
footer{border-top:1px solid var(--border);margin-top:40px;padding:28px clamp(16px,4vw,40px);
  color:var(--muted);font-size:12.5px}
footer .foot-inner{max-width:1240px;margin:0 auto;display:grid;gap:6px}
@media (max-width:900px){
  .kpis{grid-template-columns:repeat(2,1fr)}
  .span6{grid-column:span 12}
  .brand{font-size:13px}
}
"""

PAGE_JS = """
(function(){
  var doc=document, root=doc.documentElement;
  var toggle=doc.getElementById('theme-toggle');
  function paintToggle(){toggle.textContent=root.dataset.theme==='light'?'\\u263E Escuro':'\\u2600 Claro';}
  toggle.addEventListener('click',function(){
    root.dataset.theme=root.dataset.theme==='light'?'dark':'light';
    try{localStorage.setItem('cs2-theme',root.dataset.theme);}catch(e){}
    paintToggle();
  });
  paintToggle();

  var tip=doc.getElementById('tooltip');
  function tipRow(value,label,color){
    var row=doc.createElement('div');row.className='tt-row';
    if(color){var k=doc.createElement('span');k.className='tt-key';k.style.background=color;row.appendChild(k);}
    var v=doc.createElement('span');v.className='tt-val';v.textContent=value;row.appendChild(v);
    var l=doc.createElement('span');l.textContent=label;row.appendChild(l);
    return row;
  }
  function fillTip(title,rows){
    tip.textContent='';
    var t=doc.createElement('div');t.className='tt-title';t.textContent=title;tip.appendChild(t);
    rows.forEach(function(r){tip.appendChild(tipRow(r[0],r[1],r[2]));});
  }
  function placeTip(x,y){
    var pad=14,w=tip.offsetWidth,h=tip.offsetHeight;
    var left=x+pad, top=y+pad;
    if(left+w>innerWidth-8)left=x-w-pad;
    if(top+h>innerHeight-8)top=y-h-pad;
    tip.style.left=left+'px';tip.style.top=top+'px';
  }
  function showTip(){tip.style.opacity='1';}
  function hideTip(){tip.style.opacity='0';}

  function markTip(el,x,y){
    var rows=(el.getAttribute('data-tip-rows')||'').split(';;').map(function(pair){
      var i=pair.indexOf('::');return [pair.slice(0,i),pair.slice(i+2),null];
    });
    var key=el.getAttribute('data-tip-key');
    if(key&&rows.length)rows[0][2]=key;
    fillTip(el.getAttribute('data-tip-title')||'',rows);
    placeTip(x,y);showTip();
  }
  doc.addEventListener('pointermove',function(e){
    var el=e.target.closest&&e.target.closest('[data-tip-title]');
    if(el){markTip(el,e.clientX,e.clientY);}
  });
  doc.addEventListener('pointerover',function(e){
    if(!e.target.closest)return;
    if(e.target.closest('[data-xh-chart],[data-nearest-chart],#tooltip'))return;
    if(!e.target.closest('[data-tip-title]'))hideTip();
  });
  doc.addEventListener('focusin',function(e){
    var el=e.target.closest&&e.target.closest('[data-tip-title]');
    if(el){var r=el.getBoundingClientRect();markTip(el,r.left+r.width/2,r.top);}
  });
  doc.addEventListener('focusout',hideTip);

  function svgPoint(svg,e){
    var pt=new DOMPoint(e.clientX,e.clientY);
    return pt.matrixTransform(svg.getScreenCTM().inverse());
  }
  doc.querySelectorAll('[data-xh-chart]').forEach(function(wrap){
    var svg=wrap.querySelector('svg');
    var data=JSON.parse(wrap.querySelector('script[data-xh]').textContent);
    var line=svg.querySelector('.xh-line');
    var dots=svg.querySelectorAll('.xh-dot');
    svg.addEventListener('pointermove',function(e){
      var p=svgPoint(svg,e), best=0, dist=Infinity;
      data.xs.forEach(function(x,i){var d=Math.abs(x-p.x);if(d<dist){dist=d;best=i;}});
      line.setAttribute('x1',data.xs[best]);line.setAttribute('x2',data.xs[best]);
      line.style.opacity='1';
      data.series.forEach(function(s,si){
        if(dots[si]&&s.ys){dots[si].setAttribute('cx',data.xs[best]);dots[si].setAttribute('cy',s.ys[best]);
          dots[si].style.opacity=dots[si].getAttribute('r')==='0'?'0':'1';}
      });
      fillTip(data.labels[best],data.series.map(function(s){
        var c=s.color&&s.color.indexOf('var(')===0?getComputedStyle(root).getPropertyValue(s.color.slice(4,-1)).trim():s.color;
        return [s.values[best],s.name,c];
      }));
      placeTip(e.clientX,e.clientY);showTip();
    });
    svg.addEventListener('pointerleave',function(){
      line.style.opacity='0';dots.forEach(function(d){d.style.opacity='0';});hideTip();
    });
  });
  doc.querySelectorAll('[data-nearest-chart]').forEach(function(wrap){
    var svg=wrap.querySelector('svg');
    var data=JSON.parse(wrap.querySelector('script[data-nearest]').textContent);
    var halo=svg.querySelector('.nearest-halo');
    svg.addEventListener('pointermove',function(e){
      var p=svgPoint(svg,e), best=null, dist=Infinity;
      data.points.forEach(function(pt){
        var d=(pt.x-p.x)*(pt.x-p.x)+(pt.y-p.y)*(pt.y-p.y);
        if(d<dist){dist=d;best=pt;}
      });
      if(best&&dist<data.radius*data.radius){
        halo.setAttribute('cx',best.x);halo.setAttribute('cy',best.y);halo.style.opacity='1';
        fillTip(best.title,best.rows.map(function(r){return [r[0],r[1],null];}));
        placeTip(e.clientX,e.clientY);showTip();
      }else{halo.style.opacity='0';hideTip();}
    });
    svg.addEventListener('pointerleave',function(){halo.style.opacity='0';hideTip();});
  });

  var links=[].slice.call(doc.querySelectorAll('nav.sections a'));
  var map={};
  links.forEach(function(a){map[a.getAttribute('href').slice(1)]=a;});
  var spy=new IntersectionObserver(function(entries){
    entries.forEach(function(en){
      if(en.isIntersecting){
        links.forEach(function(a){a.classList.remove('active');});
        var a=map[en.target.id];if(a)a.classList.add('active');
      }
    });
  },{rootMargin:'-30% 0px -60% 0px'});
  doc.querySelectorAll('section.block').forEach(function(s){spy.observe(s);});
})();
"""


def render_index_html(data: dict, png_names: list[str]) -> str:
    overview = data["overview"]
    kpis = "".join([
        kpi_tile("Partidas únicas", pt_int(overview.partidas_unicas), "uma linha por partida"),
        kpi_tile("Perspectivas de equipe", pt_int(overview.perspectivas_de_equipe), "duas por partida na fonte"),
        kpi_tile("Recorte de modelagem", pt_int(overview.partidas_modelagem), "38 atributos pré-jogo"),
        kpi_tile("Equipes representadas", pt_int(overview.equipes_representadas),
                 f"mín. de 50 partidas: {len(data['teams'])}"),
    ])
    model_kpis = "".join([
        kpi_tile("Partidas no holdout", pt_int(len(data["pred"])), "modelo congelado"),
        kpi_tile("ROC-AUC", pt_num(data["auc"], 4), "calculado do holdout"),
        kpi_tile("Acurácia", pt_pct(100 * data["accuracy"], 2), "classe prevista × real"),
        kpi_tile("Erros com confiança ≥ 80%", pt_int(len(data["errors"])),
                 "de " + pt_int(len(data["pred"])) + " previsões"),
    ])
    ambiente = chart_stacked_share(
        "Ambiente das partidas", "LAN e online descrevem o ambiente da série; não identificam mapas.",
        [("LAN", int(overview.partidas_lan), "var(--s1)"), ("Online", int(overview.partidas_online), "var(--s2)")],
        "Partidas únicas", int(overview.partidas_unicas),
    )
    formatos = chart_stacked_share(
        "Formato das séries", "Distribuição de melhor-de-1, melhor-de-3 e melhor-de-5 no período.",
        [(r["formato"], int(r["partidas_unicas"]), f"var(--s{i + 1})")
         for i, (_, r) in enumerate(data["formats"].iterrows())],
        "Partidas únicas", int(data["formats"]["partidas_unicas"].sum()),
    )
    context = data["context"]
    elo_diff = chart_two_cols(
        "Diferença média absoluta de ELO", "Distância média de força entre as equipes, por ambiente.",
        context["ambiente"].tolist(), context["diferenca_elo_media"].tolist(), pt_num,
        aria="Diferença média absoluta de ELO por ambiente",
        table_headers=["Ambiente", "Partidas", "Diferença média de ELO"],
        table_rows=[[r["ambiente"], pt_int(r["partidas"]), pt_num(r["diferenca_elo_media"])]
                    for _, r in context.iterrows()],
    )
    favorite = chart_two_cols(
        "Vitória da equipe com maior ELO", "Frequência com que o favorito pré-jogo confirma, por ambiente.",
        context["ambiente"].tolist(), (100 * context["taxa_vitoria_maior_elo"]).tolist(), pt_pct, ref=50,
        aria="Taxa de vitória da equipe com maior ELO por ambiente",
        table_headers=["Ambiente", "Partidas", "Vitória do maior ELO"],
        table_rows=[[r["ambiente"], pt_int(r["partidas"]), pt_pct(100 * r["taxa_vitoria_maior_elo"])]
                    for _, r in context.iterrows()],
    )
    png_links = " · ".join(
        f'<a href="{esc(name)}">{esc(name)}</a>' for name in png_names
    )

    document = f"""<!doctype html>
<html lang="pt-BR" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CS2 Analytics</title>
<script>try{{var t=localStorage.getItem('cs2-theme');if(t)document.documentElement.dataset.theme=t;}}catch(e){{}}</script>
<style>{theme_css()}{PAGE_CSS}</style>
</head>
<body>
<header class="top">
  <span class="brand">CS2 ANALYTICS</span>
  <nav class="sections">
    <a href="#visao-geral">Visão geral</a>
    <a href="#equipes">Equipes</a>
    <a href="#tempo-contexto">Tempo e contexto</a>
    <a href="#elo">ELO</a>
    <a href="#modelo">Modelo</a>
  </nav>
  <div class="top-actions">
    <a class="gh-link" href="https://github.com/ceegu1N/cs2-analytics">GitHub</a>
    <button id="theme-toggle" type="button">Tema</button>
  </div>
</header>

<div class="hero">
  <h1>CS2 Analytics</h1>
  <p class="sub">Pipeline reproduzível de análise de partidas profissionais de Counter-Strike 2:
  bases consolidadas viram um banco SQLite, nove consultas SQL respondem as perguntas e este painel
  apresenta os resultados. O modelo preditivo permanece congelado — nada aqui o retreina.</p>
  <span class="period">Período operacional: {esc(overview.primeira_data)} a {esc(overview.ultima_data)}</span>
</div>
<div class="kpis">{kpis}</div>

<main>
<section class="block" id="visao-geral">
  <h2>Visão geral</h2>
  <p class="lead">Contagens globais usam uma linha por partida; perspectivas de equipe são analisadas separadamente.</p>
  <div class="cards">
    {chart_monthly_line(data["monthly"])}
    {ambiente}
    {formatos}
  </div>
</section>

<section class="block" id="equipes">
  <h2>Equipes</h2>
  <p class="lead">A taxa considera apenas as partidas em que a equipe aparece como lado representado na fonte operacional.</p>
  <div class="cards">
    {chart_teams_bar(data["teams"])}
    {chart_teams_scatter(data["teams"])}
  </div>
</section>

<section class="block" id="tempo-contexto">
  <h2>Tempo e contexto</h2>
  <p class="lead">Evolução mensal por ambiente e o efeito do contexto sobre o equilíbrio das partidas.</p>
  <div class="cards">
    {chart_stacked_area(data["monthly"])}
    {elo_diff}
    {favorite}
  </div>
</section>

<section class="block" id="elo">
  <h2>ELO e recortes</h2>
  <p class="lead">Análise descritiva da base de modelagem; associação não implica causalidade.</p>
  <div class="cards">
    {chart_elo_bands(data["elo"])}
    {chart_split_multiples(data["split"])}
  </div>
</section>

<section class="block" id="modelo">
  <h2>Desempenho do modelo</h2>
  <p class="lead">Resultado principal no holdout de {pt_int(len(data["pred"]))} partidas; recortes de contexto servem para diagnóstico descritivo.</p>
  <div class="kpis" style="padding:0;margin:0 0 16px">{model_kpis}</div>
  <div class="cards">
    {chart_accuracy_volume(data["pred_monthly"])}
    {chart_confusion(data["confusion"])}
    {chart_context_accuracy(data["pred_context"], data["pred_elo"])}
    {chart_calibration(data["calibration"])}
    {errors_table_card(data["errors"])}
  </div>
</section>
</main>

<footer>
  <div class="foot-inner">
    <span>Dados derivados de páginas públicas da HLTV.org, preparados em contexto acadêmico (TCC).
    Este painel não tem vínculo com a HLTV.org e não deve ser interpretado como sistema de apostas
    ou garantia de resultado.</span>
    <span>Código sob licença MIT · dados sujeitos às condições de atribuição descritas no repositório.</span>
    <span>Painéis estáticos (PNG): {png_links}</span>
    <span>Gabriel Herdy Rocha · Engenharia de Computação, UFSCar ·
    <a href="https://github.com/ceegu1N/cs2-analytics">GitHub</a> ·
    <a href="https://www.linkedin.com/in/gabriel-herdy/">LinkedIn</a></span>
  </div>
</footer>

<div id="tooltip" role="status" aria-live="polite"></div>
<script>{PAGE_JS}</script>
</body>
</html>
"""
    return "\n".join(line.rstrip() for line in document.splitlines()) + "\n"
