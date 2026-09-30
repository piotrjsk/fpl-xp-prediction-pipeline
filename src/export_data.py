import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

load_dotenv()

EXPORTS_DIR = Path(os.getenv("EXPORTS_DIR", "data/exports"))


def export_to_csv(
    df: pd.DataFrame,
    start_gw: int,
    end_gw: Optional[int] = None,
    data_dir: Path = EXPORTS_DIR,
) -> Path:
    """Handles dynamic column selection, numerical formatting, and CSV export for Power BI reporting."""
    preferred_cols = [
        "gw",
        "player_id",
        "player_name",
        "position_id",
        "position",
        "team_name",
        "fixture_id",
        "opposition_name",
        "was_home",
        "model_engine",
        "pred_xmin",
        "pred_player_xG",
        "pred_player_xA",
        "prob_clean_sheet",
        "pred_xP",
    ]

    # Dynamically select only columns that exist in the generated DataFrame
    available_cols = [col for col in preferred_cols if col in df.columns]
    df_export = df[available_cols].copy()

    # Numerical formatting applied conditionally to available metrics
    round_precision_map = {
        "pred_xmin": 1,
        "pred_player_xG": 3,
        "pred_player_xA": 3,
        "prob_clean_sheet": 3,
        "pred_xP": 2,
    }

    for col, precision in round_precision_map.items():
        if col in df_export.columns:
            df_export[col] = df_export[col].round(precision)

    # Dynamic filename resolution
    if end_gw is None or end_gw == start_gw:
        file_name = f"xp_gw{start_gw}.csv"
    else:
        file_name = f"xp_gw{start_gw}_to_gw{end_gw}.csv"

    output_path = data_dir / file_name

    os.makedirs(data_dir, exist_ok=True)
    df_export.to_csv(output_path, index=False, encoding="utf-8-sig")
    logging.info(f"Successfully exported CSV file: {output_path}")

    return output_path