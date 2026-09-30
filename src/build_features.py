import logging
import os
import sqlite3
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

load_dotenv()

DB_PATH = Path(os.getenv("DB_PATH", "data/fpl.db"))


def build_features() -> None:
    """Loads raw player gameweek data from SQLite, computes rolling normalized features,

    and writes the enriched feature set to the fact_player_features table.
    """
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"Database file not found at {DB_PATH}! Please run"
            " ingest_fpl_data.py first."
        )

    with sqlite3.connect(DB_PATH) as conn:
        logging.info(
            "Loading and enriching player gameweek facts from SQLite"
            " database..."
        )

        query = """
        SELECT 
            f.player_id,
            p.web_name AS player_name,
            p.position_id,
            CASE p.position_id 
                WHEN 1 THEN 'GKP'
                WHEN 2 THEN 'DEF'
                WHEN 3 THEN 'MID'
                WHEN 4 THEN 'FWD'
            END AS position,
            t.short_name AS team_name,
            f.round AS gw,
            f.fixture_id,
            f.opponent_team AS opponent_team_id,
            opt.short_name AS opponent_team_name,
            f.was_home,
            f.minutes,
            f.starts,
            f.total_points,
            f.goals_scored,
            f.assists,
            f.expected_goals AS xG,
            f.expected_assists AS xA,
            f.expected_goal_involvements AS xGI,
            f.expected_goals_conceded AS xGC,
            f.bps,
            COALESCE(f.saves, 0) AS saves,
            COALESCE(f.defensive_contribution, 0) AS defcon
        FROM fact_player_gameweek f
        JOIN dim_players p ON f.player_id = p.player_id
        JOIN fact_fixtures fix ON f.fixture_id = fix.fixture_id
        JOIN dim_teams t ON t.team_id = CASE WHEN f.was_home = 1 THEN fix.home_team_id ELSE fix.away_team_id END
        JOIN dim_teams opt ON f.opponent_team = opt.team_id
        ORDER BY f.player_id, f.round;
        """

        df = pd.read_sql(query, conn)

    logging.info("Computing rolling window features (3-GW and 5-GW windows)...")

    # Ensure strictly sorted sequence to guarantee proper rolling calculations
    df = df.sort_values(["player_id", "gw"]).reset_index(drop=True)
    grouped = df.groupby("player_id")

    # 1. Aggregate features (Minutes, Starts, Points) shifted by 1 GW to prevent data leakage
    for window in [3, 5]:
        df[f"rolling_min_{window}"] = (
            grouped["minutes"]
            .shift(1)
            .rolling(window=window, min_periods=1)
            .mean()
        )
        df[f"rolling_starts_{window}"] = (
            grouped["starts"]
            .shift(1)
            .rolling(window=window, min_periods=1)
            .mean()
        )
        df[f"rolling_pts_{window}"] = (
            grouped["total_points"]
            .shift(1)
            .rolling(window=window, min_periods=1)
            .mean()
        )

        # 2. Per-90 normalized features (xG, xA, xGC, defcon, saves)
        roll_sum_min = (
            grouped["minutes"]
            .shift(1)
            .rolling(window=window, min_periods=1)
            .sum()
        )

        for col, new_name in [
            ("xG", f"rolling_xG_90_{window}"),
            ("xA", f"rolling_xA_90_{window}"),
            ("xGC", f"rolling_xGC_90_{window}"),
            ("defcon", f"rolling_defcon_90_{window}"),
            ("saves", f"rolling_saves_90_{window}"),
        ]:
            roll_sum_stat = (
                grouped[col]
                .shift(1)
                .rolling(window=window, min_periods=1)
                .sum()
            )

            # Per-90 normalization with zero-division guard
            df[new_name] = np.where(
                roll_sum_min > 0, (roll_sum_stat / roll_sum_min) * 90.0, 0.0
            )

    # Fill NaN values for debutants or unpopulated historical windows
    rolling_cols = [c for c in df.columns if c.startswith("rolling_")]
    df[rolling_cols] = df[rolling_cols].fillna(0.0)

    # 3. Final feature schema selection for the Feature Mart
    feature_cols: List[str] = [
        "player_id",
        "player_name",
        "position_id",
        "position",
        "team_name",
        "gw",
        "fixture_id",
        "opponent_team_id",
        "opponent_team_name",
        "was_home",
        "minutes",
        "starts",
        "total_points",
        # 3-Gameweek rolling features
        "rolling_min_3",
        "rolling_starts_3",
        "rolling_xG_90_3",
        "rolling_xA_90_3",
        "rolling_xGC_90_3",
        "rolling_pts_3",
        "rolling_defcon_90_3",
        "rolling_saves_90_3",
        # 5-Gameweek rolling features (broader form context)
        "rolling_min_5",
        "rolling_starts_5",
        "rolling_xG_90_5",
        "rolling_xA_90_5",
        "rolling_xGC_90_5",
        "rolling_pts_5",
        "rolling_defcon_90_5",
        "rolling_saves_90_5",
    ]

    df_features = df[feature_cols].copy()

    logging.info(
        "Saving enriched 'fact_player_features' table to SQLite database..."
    )
    with sqlite3.connect(DB_PATH) as conn:
        df_features.to_sql(
            "fact_player_features", conn, if_exists="replace", index=False
        )

        cursor = conn.cursor()
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_features_player_gw ON"
            " fact_player_features(player_id, gw);"
        )
        conn.commit()

    logging.info(
        f"Successfully saved {len(df_features)} feature rows to SQLite"
        " database!"
    )


if __name__ == "__main__":
    build_features()