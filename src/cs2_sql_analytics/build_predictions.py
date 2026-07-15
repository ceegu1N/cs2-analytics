from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


OFFICIAL_THRESHOLD = 0.5
HIGH_CONFIDENCE_THRESHOLD = 0.8


def _elo_bands(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    categories = pd.cut(
        values,
        bins=[0, 50, 100, 200, np.inf],
        labels=["0-49", "50-99", "100-199", "200+"],
        right=False,
        include_lowest=True,
    )
    return categories.astype(str), categories.cat.codes.add(1).astype(int)


def _probability_bands(values: np.ndarray) -> tuple[pd.Series, pd.Series]:
    indexes = np.minimum((values * 10).astype(int), 9)
    labels = pd.Series(
        [f"{index * 10}-{(index + 1) * 10}%" for index in indexes],
        dtype="string",
    )
    return labels, pd.Series(indexes + 1, dtype=int)


def _confusion_labels(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    conditions = [
        (y_true == 0) & (y_pred == 0),
        (y_true == 0) & (y_pred == 1),
        (y_true == 1) & (y_pred == 0),
        (y_true == 1) & (y_pred == 1),
    ]
    labels = ["Verdadeiro negativo", "Falso positivo", "Falso negativo", "Verdadeiro positivo"]
    return np.select(conditions, labels, default="Invalido")


def prepare_holdout_predictions(
    source_model_matches: pd.DataFrame,
    prepared_model_matches: pd.DataFrame,
    teams: pd.DataFrame,
    model_path: Path | str,
) -> pd.DataFrame:
    model_path = Path(model_path)
    estimator = joblib.load(model_path)
    feature_names = list(estimator.feature_names_in_)
    missing_features = sorted(set(feature_names) - set(source_model_matches.columns))
    if missing_features:
        raise ValueError(f"O artefato exige atributos ausentes na base: {missing_features}")

    source = source_model_matches.loc[
        source_model_matches["season_usage"].astype(str).str.lower().eq("holdout")
    ].copy()
    prepared = prepared_model_matches.loc[
        prepared_model_matches["season_usage"].eq("holdout")
    ].copy()
    prepared_by_id = prepared.set_index("model_match_id")
    source_ids = source["actual_match_id"].astype(int)
    if set(source_ids) != set(prepared_by_id.index):
        raise ValueError("As linhas de holdout da fonte e da tabela preparada nao coincidem.")
    prepared = prepared_by_id.loc[source_ids].reset_index()

    probability = estimator.predict_proba(source[feature_names])[:, 1]
    predicted_class = (probability >= OFFICIAL_THRESHOLD).astype(int)
    y_true = source["win_target"].astype(int).to_numpy()
    correct = (predicted_class == y_true).astype(int)
    confidence = np.maximum(probability, 1.0 - probability)
    high_confidence = (confidence >= HIGH_CONFIDENCE_THRESHOLD).astype(int)
    high_confidence_error = ((high_confidence == 1) & (correct == 0)).astype(int)
    abs_elo = source["diff_elo_pre_match"].abs().astype(float).reset_index(drop=True)
    elo_band, elo_band_sort = _elo_bands(abs_elo)
    probability_band, probability_band_sort = _probability_bands(probability)

    team_name_by_id = dict(zip(teams["team_id"], teams["team_name"], strict=True))
    clipped_probability = np.clip(probability, 1e-15, 1 - 1e-15)
    log_loss_component = -(
        y_true * np.log(clipped_probability)
        + (1 - y_true) * np.log(1 - clipped_probability)
    )

    result = pd.DataFrame(
        {
            "model_match_id": prepared["model_match_id"].astype(int),
            "match_date_key": prepared["match_date_key"].astype(int),
            "match_date_iso": pd.to_datetime(source["match_date"]).dt.strftime("%Y-%m-%d").to_numpy(),
            "season_label": source["season_label"].astype(str).to_numpy(),
            "team_id": prepared["team_id"].astype(int),
            "team_name": prepared["team_id"].map(team_name_by_id),
            "opponent_team_id": prepared["opponent_team_id"].astype(int),
            "opponent_name": prepared["opponent_team_id"].map(team_name_by_id),
            "environment": np.where(source["context_is_lan"].astype(float).to_numpy() == 1, "LAN", "ONLINE"),
            "abs_elo_difference": abs_elo,
            "elo_band": elo_band,
            "elo_band_sort": elo_band_sort,
            "y_true": y_true,
            "actual_result": np.where(y_true == 1, "Vitoria", "Derrota"),
            "probability": probability,
            "probability_band": probability_band,
            "probability_band_sort": probability_band_sort,
            "predicted_class": predicted_class,
            "predicted_result": np.where(predicted_class == 1, "Vitoria", "Derrota"),
            "correct": correct,
            "confidence": confidence,
            "high_confidence": high_confidence,
            "high_confidence_error": high_confidence_error,
            "confusion_cell": _confusion_labels(y_true, predicted_class),
            "brier_component": np.square(probability - y_true),
            "log_loss_component": log_loss_component,
        }
    )
    if result.isna().any().any():
        null_columns = result.columns[result.isna().any()].tolist()
        raise ValueError(f"A tabela de previsoes possui valores ausentes: {null_columns}")
    return result
