import argparse
import logging
import os
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

from src.models import run_poisson_engine, run_xgboost_engine

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

load_dotenv()

DB_PATH = Path(os.getenv("DB_PATH", "data/fpl.db"))


def run_xp_pipeline_for_gw(
    conn: sqlite3.Connection, target_gw: int, engine: str = "auto"
) -> pd.DataFrame:
    """Orchestrates prediction engine: auto-selects XGBoost for GW >= 3 based on backtest conclusions."""
    use_xgboost = (engine == "xgboost") or (engine == "auto" and target_gw >= 3)

    if use_xgboost:
        logging.info(f"Applying XGBoost Predictive Engine for GW{target_gw}...")
        df_xgb = run_xgboost_engine(conn, target_gw)
        if not df_xgb.empty:
            return df_xgb
        logging.warning("XGBoost prediction failed. Falling back to Poisson baseline engine.")

    logging.info(f"Applying Poisson Baseline Engine for GW{target_gw}...")
    return run_poisson_engine(conn, target_gw)


def run_pipeline(
    start_gw: int,
    end_gw: Optional[int] = None,
    engine: str = "auto",
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    """Generates expected points predictions for a single gameweek or range of gameweeks."""
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found at {db_path}!")

    target_end_gw = end_gw if end_gw is not None else start_gw

    all_predictions = []
    with sqlite3.connect(db_path) as conn:
        for gw in range(start_gw, target_end_gw + 1):
            logging.info(f"Generating xP predictions for GW{gw} [Engine: {engine}]...")
            df_gw = run_xp_pipeline_for_gw(conn, target_gw=gw, engine=engine)
            if not df_gw.empty:
                all_predictions.append(df_gw)

    if not all_predictions:
        raise ValueError("Failed to generate xP predictions for the specified gameweeks.")

    return pd.concat(all_predictions, ignore_index=True)


def cli_interface() -> None:
    """Interactive CLI Interface for Top 10 by position and Player Spotlight search."""
    parser = argparse.ArgumentParser(description="FPL Points Predictor — Analytics Interface")
    parser.add_argument("start_gw", type=int, nargs="?", default=6)
    parser.add_argument("end_gw", type=int, nargs="?", default=None)
    parser.add_argument(
        "-e",
        "--engine",
        type=str,
        choices=["auto", "xgboost", "poisson"],
        default="auto",
        help="Model engine selection: 'auto' (Poisson <GW3, XGBoost >=GW3), 'xgboost', or 'poisson'",
    )
    parser.add_argument(
        "-p",
        "--player",
        type=str,
        default=None,
        help="Specify player name for detailed spotlight view",
    )

    args = parser.parse_args()
    start_gw = args.start_gw
    end_gw = args.end_gw
    target_player = args.player
    selected_engine = args.engine

    df_results = run_pipeline(start_gw=start_gw, end_gw=end_gw, engine=selected_engine)

    positions = ["GKP", "DEF", "MID", "FWD"]

    if end_gw is None or end_gw == start_gw:
        print(
            f"\n================ TOP 10 BY POSITION (GW{start_gw}) ================"
        )
        show_cols = [
            "player_name",
            "team_name",
            "fixture_display",
            "pred_xmin",
            "pred_xP",
            "model_engine",
        ]

        for pos in positions:
            print(f"\n--- POSITION: {pos} ---")
            top_pos = (
                df_results[
                    (df_results["position"] == pos)
                    & (df_results["pred_xmin"] > 30)
                ][show_cols]
                .sort_values(by="pred_xP", ascending=False)
                .head(10)
                .rename(columns={"fixture_display": "Opponent", "model_engine": "Engine"})
            )
            print(top_pos.to_string(index=False))

    else:
        print(
            f"\n================ TOP 10 BY POSITION (GW{start_gw} - GW{end_gw}) ================"
        )

        summary_df = (
            df_results.groupby(["player_name", "team_name", "position"])[
                "pred_xP"
            ]
            .agg(Total_xP="sum", Avg_xP="mean")
            .reset_index()
        )
        summary_df["Total_xP"] = summary_df["Total_xP"].round(2)
        summary_df["Avg_xP"] = summary_df["Avg_xP"].round(2)

        for pos in positions:
            print(f"\n--- POSITION: {pos} ---")
            top_pos = (
                summary_df[summary_df["position"] == pos]
                .sort_values(by="Total_xP", ascending=False)
                .head(10)
            )
            print(
                top_pos[
                    ["player_name", "team_name", "Total_xP", "Avg_xP"]
                ].to_string(index=False)
            )

    # --- DYNAMIC SMART SPOTLIGHT FOR TARGET PLAYER ---
    if target_player:
        print(
            f"\n================ SPOTLIGHT: {target_player.upper()} ================"
        )
        search_term = target_player.strip().lower()

        exact_mask = df_results["player_name"].str.lower() == search_term
        player_df = df_results[exact_mask]

        if player_df.empty:
            word_mask = df_results["player_name"].str.contains(
                rf"\b{search_term}\b", case=False, na=False
            )
            player_df = df_results[word_mask]

        if player_df.empty:
            player_df = df_results[
                df_results["player_name"].str.contains(
                    search_term, case=False, na=False
                )
            ]

        if not player_df.empty:
            spotlight_cols = [
                "gw",
                "player_name",
                "team_name",
                "fixture_display",
                "pred_xmin",
                "pred_xP",
                "model_engine",
            ]
            show_p = (
                player_df[spotlight_cols]
                .rename(columns={"fixture_display": "Opponent", "model_engine": "Engine"})
                .copy()
            )
            show_p["pred_xmin"] = show_p["pred_xmin"].round(1)
            show_p["pred_xP"] = show_p["pred_xP"].round(2)
            print(show_p.to_string(index=False))
        else:
            print(
                f"No player found matching search query: '{target_player}'"
            )


if __name__ == "__main__":
    cli_interface()