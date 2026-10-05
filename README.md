# FPL Live Predictive Engine: End-to-End ML Pipeline

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-150458.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient_Boosting-green.svg)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57.svg)

## Project Overview

The **FPL Live Predictive Engine** is a production-ready Data Science pipeline designed to forecast player performance (Expected Points - xP) for upcoming Fantasy Premier League (FPL) gameweeks.

While my [Backtesting Framework](https://github.com/piotrjsk/fpl-backtest-engine) proved *which* models work best at different stages of the season, this repository is the **live application**. It autonomously ingests real-time API data, processes features, dynamically trains models, and provides actionable insights for FPL managers via an interactive CLI and exportable CSV reports.

## Domain Context: How FPL Works
If you are unfamiliar with Fantasy Premier League, it is a game where managers build virtual teams of real-life Premier League footballers. 
The objective of this machine learning model is to predict **Expected Points (xP)** for the upcoming matches. Players score real points based on their on-pitch performances:
* **Minutes Played:** Points for appearances (starting or coming off the bench).
* **Attacking Returns:** High points for scoring goals and providing assists.
* **Defensive Returns:** Points awarded to Goalkeepers and Defenders for keeping a "Clean Sheet" (not conceding any goals).
* **Other Actions:** Saves, bonus points, and defensive contributions.

Accurately forecasting these points requires balancing historical player stats (xG, xA) with opponent strength and expected playing time (xMin).

## Architecture & Pipeline Flow

This project focuses on building a robust, automated pipeline from data collection to prediction:

1. **Data Ingestion (`ingest_fpl_data.py`)**
   * Connects to the official REST FPL API to fetch the latest gameweek data.
   * Transforms raw JSONs into a structured relational format stored locally in a SQLite database.

2. **Feature Engineering (`build_features.py`)**
   * Reads from SQLite and calculates rolling metrics (3-GW and 5-GW windows) for minutes, xG, xA, and defensive contributions.
   * **Strict Data Leakage Protection:** All features are explicitly shifted by 1 Gameweek to ensure models only predict based on *past* information.

3. **Hybrid Inference Engine (`predict_xp.py` & `models.py`)**
   * Implements a hybrid approach directly based on my backtest findings:
     * **Poisson Baseline:** Used very early in the season (< GW3) when historical data is scarce.
     * **XGBoost Regressor:** Takes over automatically (>= GW3) once sufficient rolling features are accumulated. Models are trained dynamically on the most recent SQLite data.

4. **Export & Reporting (`export_data.py`)**
   * Generates clean, formatted CSV files for easy review and downstream analysis.

## Interactive CLI Features

The application features a Command Line Interface allowing users to run the pipeline, view Top 10 predictions by position, and spotlight specific players.

### Usage Examples

**1. Run the full pipeline for the upcoming Gameweek (e.g., GW6):**
```bash
python main.py 6
```

**2. Fast Execution (Skip API ingestion, use existing DB) & Spotlight a Player:**
```bash
# Example: Predict GW8, skip API sync, and search for "Haaland"
python main.py 8 -f -p Haaland
```

**3. Predict a range of Gameweeks (e.g., GW6 to GW10) forcing the XGBoost engine:**
```bash
python main.py 6 10 -e xgboost
```

## Repository Structure
```text
.
├── main.py                         # Application entry point
├── requirements.txt                # Dependency requirements
├── .env.example                    # Environment variables (DB paths, API URLs)
├── .gitignore                      # Git exclusion rules
├── README.md                       # Project documentation
│
├── data/
│   ├── fpl.db                      # SQLite Relational Database (Generated)
│   └── exports/                    # Exported CSV predictions
│
└── src/
    ├── build_features.py           # Rolling metrics & SQLite feature generation
    ├── export_data.py              # Formatting and CSV export logic
    ├── ingest_fpl_data.py          # API wrapper & data extraction
    ├── models.py                   # XGBoost and Poisson statistical engines
    └── predict_xp.py               # Inference orchestration & CLI logic

```

## Key Technical Skills Demonstrated
* **Data Processing/ETL:** REST API data extraction, SQLite database management, automated data cleaning.
* **Data Science / ML:** XGBoost, Poisson distributions, dynamic model retraining, feature engineering, handling data leakage.
* **Python Programming:** Modular design, command-line interfaces, custom logging, and robust error handling.

## How to Run
```bash
git clone https://github.com/piotrjsk/fpl-xp-prediction-pipeline.git
cd fpl-xp-prediction-pipeline

# Install dependencies
pip install -r requirements.txt

# Run your first prediction
python main.py 6
```

---
*Developed by Piotr Jasiak | [LinkedIn Profile](https://www.linkedin.com/in/piotrjasiak) | See the [Backtest Experiment Here](https://github.com/piotrjsk/fpl-backtest-engine)*