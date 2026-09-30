import logging
import sqlite3
from typing import List

import numpy as np
import pandas as pd
import scipy.stats as stats
import xgboost as xgb

# Feature matrix specification for Machine Learning models
FEATURE_COLS_ML: List[str] = [
    "pred_xmin",  # Added expected minutes as a core ML feature
    "rolling_min_3",
    "rolling_starts_3",
    "rolling_xG_90_3",
    "rolling_xA_90_3",
    "rolling_xGC_90_3",
    "rolling_pts_3",
    "rolling_defcon_90_3",
    "rolling_saves_90_3",
    "rolling_min_5",
    "rolling_starts_5",
    "rolling_xG_90_5",
    "rolling_xA_90_5",
    "rolling_xGC_90_5",
    "rolling_pts_5",
    "rolling_defcon_90_5",
    "rolling_saves_90_5",
    "was_home",
    "pos_GKP",
    "pos_DEF",
    "pos_MID",
    "pos_FWD",
]

def predict_xmin(row: pd.Series) -> float:
    """Estimates expected player playing time (minutes) based on historical starting appearance rates."""
    rolling_min = row.get("rolling_min_3", 0.0)
    starts_rate = row.get("rolling_starts_3", 0.0)

    if starts_rate >= 0.67:
        return max(rolling_min, 60.0)
    elif starts_rate == 0.0:
        return min(rolling_min, 35.0)
    else:
        return rolling_min


def calculate_poisson_xp(row: pd.Series) -> float:
    """Core expected points (xP) Poisson calculation engine for FPL assets."""
    xmin = row["pred_xmin"]
    pos = row["position"]
    xg = row["pred_player_xG"]
    xa = row["pred_player_xA"]
    p_cs = row["prob_clean_sheet"]
    opp_xg = row["pred_opp_xG"]
    exp_defcon = row["exp_defcon"]
    exp_saves = row["exp_saves"]

    if xmin <= 0:
        return 0.0

    # 1. Appearance points
    pts_min = 1.0 if xmin < 60 else 2.0

    # 2. Offensive returns (Goals + Assists)
    goal_pts_map = {"FWD": 4, "MID": 5, "DEF": 6, "GKP": 6, "GK": 6}
    pts_offense = (xg * goal_pts_map.get(pos, 4)) + (xa * 3.0)

    # 3. Clean sheet returns (scaled by likelihood of playing >= 60 mins)
    prob_60_min = min(1.0, xmin / 90.0) if xmin >= 60 else 0.0
    cs_pts_map = {"GKP": 4, "GK": 4, "DEF": 4, "MID": 1, "FWD": 0}
    pts_cs = p_cs * cs_pts_map.get(pos, 0) * prob_60_min

    # 4. Goals conceded penalty (GK / DEF)
    pts_conceded_penalty = 0.0
    if pos in ["GKP", "GK", "DEF"]:
        exp_xgc_on_pitch = opp_xg * (xmin / 90.0)
        pts_conceded_penalty = -(exp_xgc_on_pitch / 2.0)

    # 5. Defensive Contribution (DefCon using Poisson CDF)
    pts_defcon = 0.0
    prob_defcon = 0.0
    if exp_defcon > 0:
        if pos == "DEF":
            prob_defcon = 1.0 - stats.poisson.cdf(9, exp_defcon)
            pts_defcon = prob_defcon * 2.0
        elif pos in ["MID", "FWD"]:
            prob_defcon = 1.0 - stats.poisson.cdf(11, exp_defcon)
            pts_defcon = prob_defcon * 2.0

    # 6. Goalkeeper Saves (using Poisson CDF)
    pts_saves = 0.0
    p_3_plus = 0.0
    if pos in ["GKP", "GK"] and exp_saves > 0:
        p_3_5 = stats.poisson.cdf(5, exp_saves) - stats.poisson.cdf(2, exp_saves)
        p_6_8 = stats.poisson.cdf(8, exp_saves) - stats.poisson.cdf(5, exp_saves)
        p_9_plus = 1.0 - stats.poisson.cdf(8, exp_saves)

        pts_saves = (1.0 * p_3_5) + (2.0 * p_6_8) + (3.0 * p_9_plus)
        p_3_plus = 1.0 - stats.poisson.cdf(2, exp_saves)

    # 7. Estimated Bonus Points System (BPS) contributions
    pts_bonus = (xg * 1.2) + (xa * 1.0)
    if pos == "DEF":
        pts_bonus += p_cs * prob_defcon * 1.5
    elif pos in ["GKP", "GK"]:
        pts_bonus += p_cs * p_3_plus * 1.5

    return (
        pts_min
        + pts_offense
        + pts_cs
        + pts_conceded_penalty
        + pts_defcon
        + pts_saves
        + pts_bonus
    )


def prepare_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encodes positional One-Hot features required by the Machine Learning feature matrix."""
    df_encoded = df.copy()

    for pos_code in ["GKP", "DEF", "MID", "FWD"]:
        df_encoded[f"pos_{pos_code}"] = (
            df_encoded["position"] == pos_code
        ).astype(int)

    return df_encoded


def run_poisson_engine(
    conn: sqlite3.Connection, target_gw: int
) -> pd.DataFrame:
    """Generates expected points predictions using the Poisson distribution baseline engine."""
    team_xg_query = f"""
    SELECT 
        t.short_name AS team_name,
        f.fixture_id,
        f.was_home,
        f.round AS gw,
        SUM(f.expected_goals) AS team_xG
    FROM fact_player_gameweek f
    JOIN fact_fixtures fix ON f.fixture_id = fix.fixture_id
    JOIN dim_teams t ON t.team_id = CASE WHEN f.was_home = 1 THEN fix.home_team_id ELSE fix.away_team_id END
    WHERE f.round < {target_gw}
    GROUP BY t.short_name, f.fixture_id, f.was_home, f.round;
    """
    team_stats = pd.read_sql(team_xg_query, conn)

    if len(team_stats) > 0:
        fixture_teams = team_stats.merge(
            team_stats, on="fixture_id", suffixes=("_team", "_opponent")
        )
        fixture_teams = fixture_teams[
            fixture_teams["team_name_team"] != fixture_teams["team_name_opponent"]
        ]

        team_perf = (
            fixture_teams.groupby(["team_name_team", "was_home_team"])["team_xG_team"]
            .mean()
            .reset_index()
            .rename(
                columns={
                    "team_name_team": "team_name",
                    "was_home_team": "was_home",
                    "team_xG_team": "avg_xG_scored",
                }
            )
        )

        team_def = (
            fixture_teams.groupby(["team_name_team", "was_home_team"])["team_xG_opponent"]
            .mean()
            .reset_index()
            .rename(
                columns={
                    "team_name_team": "team_name",
                    "was_home_team": "was_home",
                    "team_xG_opponent": "avg_xGC_conceded",
                }
            )
        )
        team_ratings = team_perf.merge(team_def, on=["team_name", "was_home"])
    else:
        team_ratings = pd.DataFrame()

    latest_gw_query = "SELECT MAX(gw) FROM fact_player_features;"
    last_played_gw = pd.read_sql(latest_gw_query, conn).iloc[0, 0]

    player_features = pd.read_sql(
        f"SELECT * FROM fact_player_features WHERE gw = {last_played_gw};", conn
    )
    player_features = player_features.drop(
        columns=["fixture_id", "was_home"], errors="ignore"
    )

    fixtures_query = f"""
    SELECT 
        f.fixture_id,
        f.round AS gw,
        t_home.short_name AS home_team,
        t_away.short_name AS away_team
    FROM fact_fixtures f
    JOIN dim_teams t_home ON f.home_team_id = t_home.team_id
    JOIN dim_teams t_away ON f.away_team_id = t_away.team_id
    WHERE f.round = {target_gw};
    """
    fixtures_df = pd.read_sql(fixtures_query, conn)

    if fixtures_df.empty:
        logging.warning(f"No scheduled fixtures found for GW{target_gw}!")
        return pd.DataFrame()

    home_fixtures = fixtures_df[["fixture_id", "home_team", "away_team"]].rename(
        columns={"home_team": "team_name", "away_team": "opposition_name"}
    )
    home_fixtures["was_home"] = 1
    home_fixtures["fixture_display"] = home_fixtures["opposition_name"] + " (H)"

    away_fixtures = fixtures_df[["fixture_id", "away_team", "home_team"]].rename(
        columns={"away_team": "team_name", "home_team": "opposition_name"}
    )
    away_fixtures["was_home"] = 0
    away_fixtures["fixture_display"] = away_fixtures["opposition_name"] + " (A)"

    gw_schedule = pd.concat([home_fixtures, away_fixtures], ignore_index=True)

    val_model = player_features.merge(gw_schedule, on="team_name", how="inner")
    val_model["gw"] = target_gw

    val_model["pred_xmin"] = val_model.apply(predict_xmin, axis=1)
    val_model["pred_xmin"] = np.clip(val_model["pred_xmin"], 0, 90)

    if len(team_ratings) > 0:
        team_avg = (
            team_ratings.groupby("team_name")[["avg_xG_scored", "avg_xGC_conceded"]]
            .mean()
            .reset_index()
        )
        val_model = val_model.merge(team_avg, on="team_name", how="left")
        opp_avg = team_avg.rename(
            columns={
                "team_name": "opposition_name",
                "avg_xG_scored": "opp_avg_xG_scored",
                "avg_xGC_conceded": "opp_avg_xGC_conceded",
            }
        )
        val_model = val_model.merge(opp_avg, on="opposition_name", how="left")
    else:
        val_model["avg_xG_scored"] = 1.2
        val_model["avg_xGC_conceded"] = 1.2
        val_model["opp_avg_xG_scored"] = 1.2
        val_model["opp_avg_xGC_conceded"] = 1.2

    val_model["pred_team_xG"] = (
        val_model["avg_xG_scored"].fillna(1.2) + val_model["opp_avg_xGC_conceded"].fillna(1.2)
    ) / 2.0
    val_model["pred_opp_xG"] = (
        val_model["opp_avg_xG_scored"].fillna(1.2) + val_model["avg_xGC_conceded"].fillna(1.2)
    ) / 2.0

    team_xG_sum = val_model.groupby("team_name")["rolling_xG_90_3"].transform("sum")
    val_model["xG_share"] = np.where(
        team_xG_sum > 0, val_model["rolling_xG_90_3"] / team_xG_sum, 0.0
    )

    val_model["pred_player_xG"] = (
        val_model["pred_team_xG"] * val_model["xG_share"] * (val_model["pred_xmin"] / 90.0)
    )
    val_model["pred_player_xA"] = val_model["rolling_xA_90_3"] * (val_model["pred_xmin"] / 90.0)

    val_model["prob_clean_sheet"] = np.exp(-val_model["pred_opp_xG"])
    val_model["exp_defcon"] = val_model["rolling_defcon_90_3"] * (val_model["pred_xmin"] / 90.0)
    val_model["exp_saves"] = np.where(
        val_model["position"].isin(["GKP", "GK"]),
        val_model["rolling_saves_90_3"] * (val_model["pred_xmin"] / 90.0),
        0.0,
    )

    val_model["pred_xP"] = val_model.apply(calculate_poisson_xp, axis=1)
    val_model["model_engine"] = "Poisson"
    return val_model


def run_xgboost_engine(
    conn: sqlite3.Connection, target_gw: int
) -> pd.DataFrame:
    """Trains XGBoost Regressor on available historical SQLite data and predicts expected points."""
    # 1. Check the latest completed gameweek available in the database
    latest_gw_query = "SELECT MAX(gw) FROM fact_player_features;"
    last_played_gw = pd.read_sql(latest_gw_query, conn).iloc[0, 0]

    logging.info(f"Training XGBoost model on actual historical data (GW1 to GW{last_played_gw})...")

    # 2. Fetch historical observations up to last completed gameweek
    train_query = f"""
    SELECT * FROM fact_player_features 
    WHERE gw <= {last_played_gw};
    """
    train_df = pd.read_sql(train_query, conn)

    if len(train_df) < 50:
        logging.warning("Insufficient training data for XGBoost. Falling back to Poisson engine.")
        return pd.DataFrame()

    # Calculate pred_xmin on historical data for training
    train_df["pred_xmin"] = train_df.apply(predict_xmin, axis=1)
    train_df["pred_xmin"] = np.clip(train_df["pred_xmin"], 0, 90)

    train_df = prepare_ml_features(train_df)
    X_train = train_df[FEATURE_COLS_ML]
    y_train = train_df["total_points"]

    # 3. Train XGBoost Regressor
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=13,
    )
    xgb_model.fit(X_train, y_train)

    # 4. Map upcoming fixture schedule for target gameweek
    player_features = pd.read_sql(
        f"SELECT * FROM fact_player_features WHERE gw = {last_played_gw};", conn
    )

    fixtures_query = f"""
    SELECT 
        f.fixture_id,
        f.round AS gw,
        t_home.short_name AS home_team,
        t_away.short_name AS away_team
    FROM fact_fixtures f
    JOIN dim_teams t_home ON f.home_team_id = t_home.team_id
    JOIN dim_teams t_away ON f.away_team_id = t_away.team_id
    WHERE f.round = {target_gw};
    """
    fixtures_df = pd.read_sql(fixtures_query, conn)

    if fixtures_df.empty:
        logging.warning(f"No scheduled fixtures found for GW{target_gw}!")
        return pd.DataFrame()

    home_fixtures = fixtures_df[["fixture_id", "home_team", "away_team"]].rename(
        columns={"home_team": "team_name", "away_team": "opposition_name"}
    )
    home_fixtures["was_home"] = 1
    home_fixtures["fixture_display"] = home_fixtures["opposition_name"] + " (H)"

    away_fixtures = fixtures_df[["fixture_id", "away_team", "home_team"]].rename(
        columns={"away_team": "team_name", "home_team": "opposition_name"}
    )
    away_fixtures["was_home"] = 0
    away_fixtures["fixture_display"] = away_fixtures["opposition_name"] + " (A)"

    gw_schedule = pd.concat([home_fixtures, away_fixtures], ignore_index=True)

    val_model = player_features.drop(columns=["fixture_id", "was_home"], errors="ignore")
    val_model = val_model.merge(gw_schedule, on="team_name", how="inner")
    val_model["gw"] = target_gw

    # 5. Estimate expected minutes (xMin)
    val_model["pred_xmin"] = val_model.apply(predict_xmin, axis=1)
    val_model["pred_xmin"] = np.clip(val_model["pred_xmin"], 0, 90)

    val_model = prepare_ml_features(val_model)
    X_test = val_model[FEATURE_COLS_ML]

    # 6. Generate points prediction from XGBoost model
    raw_xp = xgb_model.predict(X_test)
    raw_xp = np.maximum(0.0, raw_xp)

    # 7. Apply Domain Safety Guardrail: Hard zero for 0 expected minutes
    val_model["pred_xP"] = np.where(val_model["pred_xmin"] == 0.0, 0.0, raw_xp)
    val_model["model_engine"] = "XGBoost"

    return val_model