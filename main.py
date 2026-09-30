import argparse
import logging
from typing import Optional

from src.build_features import build_features
from src.export_data import export_to_csv
from src.ingest_fpl_data import main as run_ingest
from src.predict_xp import run_pipeline

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def run_full_pipeline(
    start_gw: int,
    end_gw: Optional[int] = None,
    engine: str = "auto",
    skip_ingest: bool = False,
) -> None:
    """Executes the end-to-end FPL predictive pipeline."""
    logging.info("=== STARTING FPL PIPELINE ===")

    # 1. Ingestion (API -> SQLite)
    if not skip_ingest:
        logging.info("--- STEP 1/4: Data Ingestion (API -> SQLite) ---")
        run_ingest()
    else:
        logging.info("--- STEP 1/4: Skipping Data Ingestion ---")

    # 2. Build Features
    logging.info("--- STEP 2/4: Feature Engineering ---")
    build_features()

    # 3. Predict xP (Hybrid Engine: Poisson < GW5 / XGBoost >= GW5)
    logging.info(f"--- STEP 3/4: xP Prediction Engine (Selected Engine: {engine}) ---")
    df_results = run_pipeline(start_gw=start_gw, end_gw=end_gw, engine=engine)

    # 4. Export
    logging.info("--- STEP 4/4: Exporting Data to CSV ---")
    csv_path = export_to_csv(df_results, start_gw=start_gw, end_gw=end_gw)

    logging.info(
        f"=== PIPELINE COMPLETED SUCCESSFULLY! Results saved to: {csv_path} ==="
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FPL Points Predictor — End-to-End Execution Pipeline"
    )
    parser.add_argument(
        "start_gw",
        type=int,
        nargs="?",
        default=6,
        help="Starting Gameweek for prediction (default: 6)",
    )

    parser.add_argument(
        "end_gw",
        type=int,
        nargs="?",
        default=None,
        help="Ending Gameweek for prediction range (optional)",
    )

    parser.add_argument(
        "-e",
        "--engine",
        type=str,
        choices=["auto", "xgboost", "poisson"],
        default="auto",
        help="Model engine selection: 'auto' (Poisson <GW5, XGBoost >=GW5), 'xgboost', or 'poisson'",
    )

    parser.add_argument(
        "-f",
        "--fast",
        action="store_true",
        help="Skip API ingestion (uses existing local SQLite database)",
    )

    args = parser.parse_args()

    run_full_pipeline(
        start_gw=args.start_gw,
        end_gw=args.end_gw,
        engine=args.engine,
        skip_ingest=args.fast,
    )


if __name__ == "__main__":
    main()