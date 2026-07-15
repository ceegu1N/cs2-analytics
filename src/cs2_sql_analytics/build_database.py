from __future__ import annotations

import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .build_predictions import prepare_holdout_predictions
from .config import (
    DATABASE_PATH,
    MODEL_ARTIFACT_PATH,
    MODEL_MATCHES_PATH,
    RAW_MATCHES_PATH,
    REPO_ROOT,
    SCHEMA_PATH,
)


RAW_REQUIRED_COLUMNS = {
    "Match_Date",
    "Season_Label",
    "Season_Usage",
    "Team_ID",
    "Team_Name",
    "Opponent",
    "Team_Score",
    "Opponent_Score",
    "Win",
    "Best_Of",
    "LAN_Online",
    "Match_ID",
    "Match_URL",
}

MODEL_ID_COLUMNS = {
    "match_date",
    "actual_match_id",
    "season_label",
    "season_usage",
    "team_name",
    "opponent_name",
    "win_target",
}


def normalize_team_name(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def validate_columns(frame: pd.DataFrame, required: set[str], source_name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source_name} não contém as colunas obrigatórias: {missing}")


def _display_name(names: Counter[str]) -> str:
    return sorted(names.items(), key=lambda item: (-item[1], item[0].casefold(), item[0]))[0][0]


def build_team_dimension(raw: pd.DataFrame, model: pd.DataFrame) -> pd.DataFrame:
    names_by_key: dict[str, Counter[str]] = defaultdict(Counter)
    hltv_ids: dict[str, set[int]] = defaultdict(set)

    sources = (
        raw["Team_Name"],
        raw["Opponent"],
        model["team_name"],
        model["opponent_name"],
    )
    for series in sources:
        for name in series.astype(str):
            names_by_key[normalize_team_name(name)][name.strip()] += 1

    for row in raw[["Team_Name", "Team_ID"]].itertuples(index=False):
        hltv_ids[normalize_team_name(row.Team_Name)].add(int(row.Team_ID))

    conflicting_ids = {key: ids for key, ids in hltv_ids.items() if len(ids) > 1}
    if conflicting_ids:
        raise ValueError(f"Uma equipe foi associada a múltiplos IDs da HLTV: {conflicting_ids}")

    records = []
    for team_id, key in enumerate(sorted(names_by_key), start=1):
        ids = hltv_ids.get(key, set())
        records.append(
            {
                "team_id": team_id,
                "team_key": key,
                "team_name": _display_name(names_by_key[key]),
                "hltv_team_id": next(iter(ids)) if ids else None,
            }
        )
    return pd.DataFrame.from_records(records)


def build_date_dimension(raw: pd.DataFrame, model: pd.DataFrame) -> pd.DataFrame:
    dates = pd.concat(
        [pd.to_datetime(raw["Match_Date"]), pd.to_datetime(model["match_date"])],
        ignore_index=True,
    ).drop_duplicates().sort_values()
    return pd.DataFrame(
        {
            "date_key": dates.dt.strftime("%Y%m%d").astype(int),
            "date_iso": dates.dt.strftime("%Y-%m-%d"),
            "year": dates.dt.year.astype(int),
            "month": dates.dt.month.astype(int),
            "quarter": dates.dt.quarter.astype(int),
            "semester": np.where(dates.dt.month <= 6, 1, 2).astype(int),
            "year_month": dates.dt.strftime("%Y-%m"),
        }
    )


def _lookup_maps(teams: pd.DataFrame) -> tuple[dict[str, int], dict[str, str]]:
    id_by_key = dict(zip(teams["team_key"], teams["team_id"], strict=True))
    name_by_key = dict(zip(teams["team_key"], teams["team_name"], strict=True))
    return id_by_key, name_by_key


def prepare_team_matches(raw: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    id_by_key, _ = _lookup_maps(teams)
    frame = raw.copy()
    frame["team_key"] = frame["Team_Name"].map(normalize_team_name)
    frame["opponent_key"] = frame["Opponent"].map(normalize_team_name)
    result = pd.DataFrame(
        {
            "perspective_id": np.arange(1, len(frame) + 1, dtype=int),
            "match_id": frame["Match_ID"].astype(int),
            "match_date_key": pd.to_datetime(frame["Match_Date"]).dt.strftime("%Y%m%d").astype(int),
            "season_label": frame["Season_Label"].astype(str),
            "season_usage": frame["Season_Usage"].astype(str).str.lower(),
            "team_id": frame["team_key"].map(id_by_key).astype(int),
            "opponent_team_id": frame["opponent_key"].map(id_by_key).astype(int),
            "team_score": frame["Team_Score"].astype(int),
            "opponent_score": frame["Opponent_Score"].astype(int),
            "win": frame["Win"].astype(int),
            "best_of": frame["Best_Of"].astype(str).str.lower(),
            "lan_online": frame["LAN_Online"].astype(str).str.upper(),
            "match_url": frame["Match_URL"].astype(str),
        }
    )
    if result.duplicated(["match_id", "team_id"]).any():
        raise ValueError("A fonte operacional contém perspectivas duplicadas para a mesma equipe e partida.")
    return result


def prepare_canonical_matches(raw: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    id_by_key, _ = _lookup_maps(teams)
    frame = raw.copy()
    frame["team_key"] = frame["Team_Name"].map(normalize_team_name)
    frame["opponent_key"] = frame["Opponent"].map(normalize_team_name)
    frame["team_a_key"] = frame[["team_key", "opponent_key"]].min(axis=1)
    frame["team_b_key"] = frame[["team_key", "opponent_key"]].max(axis=1)
    team_is_a = frame["team_key"] == frame["team_a_key"]
    frame["team_a_score"] = np.where(team_is_a, frame["Team_Score"], frame["Opponent_Score"])
    frame["team_b_score"] = np.where(team_is_a, frame["Opponent_Score"], frame["Team_Score"])
    frame["team_a_win"] = np.where(team_is_a, frame["Win"], 1 - frame["Win"])

    consistency_columns = [
        "Match_Date",
        "Season_Label",
        "Season_Usage",
        "team_a_key",
        "team_b_key",
        "team_a_score",
        "team_b_score",
        "team_a_win",
        "Best_Of",
        "LAN_Online",
        "Match_URL",
    ]
    conflicts = {
        column: int((frame.groupby("Match_ID")[column].nunique(dropna=False) > 1).sum())
        for column in consistency_columns
    }
    conflicts = {column: count for column, count in conflicts.items() if count}
    if conflicts:
        raise ValueError(f"As duas perspectivas da partida não são consistentes: {conflicts}")

    canonical = frame.sort_values(["Match_ID", "team_a_key"]).drop_duplicates("Match_ID")
    return pd.DataFrame(
        {
            "match_id": canonical["Match_ID"].astype(int),
            "match_date_key": pd.to_datetime(canonical["Match_Date"]).dt.strftime("%Y%m%d").astype(int),
            "season_label": canonical["Season_Label"].astype(str),
            "season_usage": canonical["Season_Usage"].astype(str).str.lower(),
            "team_a_id": canonical["team_a_key"].map(id_by_key).astype(int),
            "team_b_id": canonical["team_b_key"].map(id_by_key).astype(int),
            "team_a_score": canonical["team_a_score"].astype(int),
            "team_b_score": canonical["team_b_score"].astype(int),
            "team_a_win": canonical["team_a_win"].astype(int),
            "best_of": canonical["Best_Of"].astype(str).str.lower(),
            "lan_online": canonical["LAN_Online"].astype(str).str.upper(),
            "match_url": canonical["Match_URL"].astype(str),
        }
    )


def prepare_model_matches(model: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    id_by_key, _ = _lookup_maps(teams)
    feature_columns = [column for column in model.columns if column not in MODEL_ID_COLUMNS]
    if len(feature_columns) != 38:
        raise ValueError(
            f"A base de modelagem deveria ter 38 atributos, mas foram encontrados {len(feature_columns)}."
        )

    result = pd.DataFrame(
        {
            "model_match_id": model["actual_match_id"].astype(int),
            "match_date_key": pd.to_datetime(model["match_date"]).dt.strftime("%Y%m%d").astype(int),
            "season_label": model["season_label"].astype(str),
            "season_usage": model["season_usage"].astype(str).str.lower(),
            "team_id": model["team_name"].map(normalize_team_name).map(id_by_key).astype(int),
            "opponent_team_id": model["opponent_name"].map(normalize_team_name).map(id_by_key).astype(int),
            "win_target": model["win_target"].astype(int),
        }
    )
    for column in feature_columns:
        result[column] = pd.to_numeric(model[column], errors="raise")
    if result["model_match_id"].duplicated().any():
        raise ValueError("A base de modelagem contém IDs de partida duplicados.")
    return result


def build_database(
    repo_root: Path | str = REPO_ROOT,
    database_path: Path | str = DATABASE_PATH,
) -> Path:
    repo_root = Path(repo_root).resolve()
    database_path = Path(database_path).resolve()
    raw_path = repo_root / RAW_MATCHES_PATH.relative_to(REPO_ROOT)
    model_path = repo_root / MODEL_MATCHES_PATH.relative_to(REPO_ROOT)
    schema_path = repo_root / SCHEMA_PATH.relative_to(REPO_ROOT)
    model_artifact_path = repo_root / MODEL_ARTIFACT_PATH.relative_to(REPO_ROOT)

    raw = pd.read_csv(raw_path)
    model = pd.read_csv(model_path)
    validate_columns(raw, RAW_REQUIRED_COLUMNS, raw_path.name)
    validate_columns(model, MODEL_ID_COLUMNS, model_path.name)

    teams = build_team_dimension(raw, model)
    dates = build_date_dimension(raw, model)
    team_matches = prepare_team_matches(raw, teams)
    matches = prepare_canonical_matches(raw, teams)
    model_matches = prepare_model_matches(model, teams)
    holdout_predictions = prepare_holdout_predictions(
        source_model_matches=model,
        prepared_model_matches=model_matches,
        teams=teams,
        model_path=model_artifact_path,
    )

    database_path.parent.mkdir(parents=True, exist_ok=True)
    database_path.unlink(missing_ok=True)
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(schema_path.read_text(encoding="utf-8"))
        teams.to_sql("dim_team", connection, if_exists="append", index=False)
        dates.to_sql("dim_date", connection, if_exists="append", index=False)
        team_matches.to_sql("fact_team_match", connection, if_exists="append", index=False)
        matches.to_sql("fact_match", connection, if_exists="append", index=False)
        model_matches.to_sql("fact_model_match", connection, if_exists="append", index=False)
        holdout_predictions.to_sql(
            "fact_prediction_holdout", connection, if_exists="append", index=False
        )
        metadata = [
            ("raw_source", str(raw_path.relative_to(repo_root))),
            ("model_source", str(model_path.relative_to(repo_root))),
            ("team_perspectives", str(len(team_matches))),
            ("unique_matches", str(len(matches))),
            ("model_matches", str(len(model_matches))),
            ("model_attributes", "38"),
            ("holdout_predictions", str(len(holdout_predictions))),
        ]
        connection.executemany(
            "INSERT INTO build_metadata(metadata_key, metadata_value) VALUES (?, ?)",
            metadata,
        )
        foreign_key_errors = list(connection.execute("PRAGMA foreign_key_check"))
        if foreign_key_errors:
            raise ValueError(f"Falha de integridade referencial: {foreign_key_errors[:5]}")
        connection.commit()
    except Exception:
        connection.rollback()
        connection.close()
        database_path.unlink(missing_ok=True)
        raise
    else:
        connection.close()
    return database_path


if __name__ == "__main__":
    path = build_database()
    print(f"Banco criado em: {path}")
