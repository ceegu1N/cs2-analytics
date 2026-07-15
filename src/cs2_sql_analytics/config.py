from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
ANALYTICS_ROOT = PACKAGE_ROOT.parents[1]
REPO_ROOT = ANALYTICS_ROOT

RAW_MATCHES_PATH = REPO_ROOT / "data" / "raw" / "core" / "matches_top50_dated.csv"
MODEL_MATCHES_PATH = (
    REPO_ROOT / "data" / "processed" / "match_feature_differences.csv"
)
MODEL_ARTIFACT_PATH = REPO_ROOT / "models" / "logistic_regression.joblib"
DATABASE_PATH = ANALYTICS_ROOT / "output" / "cs2_analytics.sqlite"
SCHEMA_PATH = ANALYTICS_ROOT / "sql" / "schema.sql"
QUERY_DIR = ANALYTICS_ROOT / "sql" / "queries"
QUERY_OUTPUT_DIR = ANALYTICS_ROOT / "output" / "query_results"
DASHBOARD_DIR = ANALYTICS_ROOT / "output" / "dashboard"
