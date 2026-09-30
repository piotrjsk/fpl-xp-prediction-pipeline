import logging
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

load_dotenv()

BASE_URL = os.getenv("FPL_BASE_URL", "https://fantasy.premierleague.com/api")
DB_PATH = Path(os.getenv("DB_PATH", "data/fpl.db"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "10"))


def safe_float(val: Any, default: float = 0.0) -> float:
    """Safely converts value to float with default fallback."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def create_session() -> requests.Session:
    """Creates an optimized HTTP session with automatic retries and thread-pool connection pooling."""
    session = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=MAX_WORKERS,
        pool_maxsize=MAX_WORKERS,
        max_retries=Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        ),
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
    })
    return session


def init_db(conn: sqlite3.Connection) -> None:
    """Initializes relational schema in SQLite database and enables WAL mode optimizations."""
    cursor = conn.cursor()

    # Enable WAL mode for concurrent write performance
    cursor.execute("PRAGMA journal_mode = WAL;")
    cursor.execute("PRAGMA synchronous = NORMAL;")

    # Reset schema for data consistency
    cursor.executescript("""
    DROP TABLE IF EXISTS fact_player_gameweek;
    DROP TABLE IF EXISTS fact_fixtures;
    DROP TABLE IF EXISTS dim_players;
    DROP TABLE IF EXISTS dim_teams;

    CREATE TABLE dim_teams (
        team_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        short_name TEXT NOT NULL,
        strength INTEGER,
        strength_overall_home INTEGER,
        strength_overall_away INTEGER,
        strength_attack_home INTEGER,
        strength_attack_away INTEGER,
        strength_defence_home INTEGER,
        strength_defence_away INTEGER
    );

    CREATE TABLE dim_players (
        player_id INTEGER PRIMARY KEY,
        first_name TEXT,
        second_name TEXT,
        web_name TEXT NOT NULL,
        team_id INTEGER,
        position_id INTEGER,
        now_cost INTEGER,
        status TEXT,
        news TEXT,
        FOREIGN KEY (team_id) REFERENCES dim_teams (team_id)
    );

    CREATE TABLE fact_fixtures (
        fixture_id INTEGER PRIMARY KEY,
        round INTEGER,
        home_team_id INTEGER,
        away_team_id INTEGER,
        home_score INTEGER,
        away_score INTEGER,
        finished INTEGER,
        kickoff_time TEXT,
        home_difficulty INTEGER,
        away_difficulty INTEGER,
        FOREIGN KEY (home_team_id) REFERENCES dim_teams (team_id),
        FOREIGN KEY (away_team_id) REFERENCES dim_teams (team_id)
    );

    CREATE TABLE fact_player_gameweek (
        player_id INTEGER,
        round INTEGER,
        fixture_id INTEGER,
        opponent_team INTEGER,
        was_home INTEGER,
        total_points INTEGER,
        minutes INTEGER,
        starts INTEGER,
        goals_scored INTEGER,
        assists INTEGER,
        clean_sheets INTEGER,
        goals_conceded INTEGER,
        own_goals INTEGER,
        penalties_saved INTEGER,
        penalties_missed INTEGER,
        yellow_cards INTEGER,
        red_cards INTEGER,
        saves INTEGER,
        clearances_blocks_interceptions INTEGER,
        recoveries INTEGER,
        tackles INTEGER,
        defensive_contribution INTEGER,
        bonus INTEGER,
        bps INTEGER,
        influence REAL,
        creativity REAL,
        threat REAL,
        ict_index REAL,
        expected_goals REAL,
        expected_assists REAL,
        expected_goal_involvements REAL,
        expected_goals_conceded REAL,
        value INTEGER,
        transfers_balance INTEGER,
        selected INTEGER,
        PRIMARY KEY (player_id, round, fixture_id),
        FOREIGN KEY (player_id) REFERENCES dim_players (player_id),
        FOREIGN KEY (opponent_team) REFERENCES dim_teams (team_id),
        FOREIGN KEY (fixture_id) REFERENCES fact_fixtures (fixture_id)
    );

    CREATE INDEX idx_fact_player_round ON fact_player_gameweek(player_id, round);
    CREATE INDEX idx_fact_round ON fact_player_gameweek(round);
    CREATE INDEX idx_fixtures_round ON fact_fixtures(round);
    """)
    conn.commit()


def fetch_bootstrap_data(session: requests.Session) -> Dict[str, Any]:
    url = f"{BASE_URL}/bootstrap-static/"
    resp = session.get(url, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_fixtures_data(session: requests.Session) -> List[Dict[str, Any]]:
    url = f"{BASE_URL}/fixtures/"
    resp = session.get(url, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_player_history(
    session: requests.Session, player_id: int
) -> Tuple[int, List[Dict[str, Any]]]:
    url = f"{BASE_URL}/element-summary/{player_id}/"
    try:
        resp = session.get(url, timeout=10)
        if resp.status_code == 200:
            return player_id, resp.json().get("history", [])
    except requests.RequestException as e:
        logging.warning(
            f"Error fetching player history for player_id {player_id}: {e}"
        )
    return player_id, []


def save_dimensions(
    conn: sqlite3.Connection, bootstrap_data: Dict[str, Any]
) -> None:
    cursor = conn.cursor()

    teams_data = [
        (
            t["id"],
            t["name"],
            t["short_name"],
            t["strength"],
            t["strength_overall_home"],
            t["strength_overall_away"],
            t["strength_attack_home"],
            t["strength_attack_away"],
            t["strength_defence_home"],
            t["strength_defence_away"],
        )
        for t in bootstrap_data["teams"]
    ]
    cursor.executemany(
        "INSERT OR REPLACE INTO dim_teams VALUES (?,?,?,?,?,?,?,?,?,?)",
        teams_data,
    )

    players_data = [
        (
            p["id"],
            p["first_name"],
            p["second_name"],
            p["web_name"],
            p["team"],
            p["element_type"],
            p["now_cost"],
            p["status"],
            p["news"],
        )
        for p in bootstrap_data["elements"]
    ]
    cursor.executemany(
        "INSERT OR REPLACE INTO dim_players VALUES (?,?,?,?,?,?,?,?,?)",
        players_data,
    )

    conn.commit()


def save_fixtures(
    conn: sqlite3.Connection, fixtures_data: List[Dict[str, Any]]
) -> None:
    cursor = conn.cursor()
    rows = [
        (
            f["id"],
            f.get("event"),
            f["team_h"],
            f["team_a"],
            f.get("team_h_score"),
            f.get("team_a_score"),
            1 if f.get("finished") else 0,
            f.get("kickoff_time"),
            f.get("team_h_difficulty"),
            f.get("team_a_difficulty"),
        )
        for f in fixtures_data
    ]
    cursor.executemany(
        "INSERT OR REPLACE INTO fact_fixtures VALUES (?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()


def save_facts(
    conn: sqlite3.Connection,
    all_histories: List[Tuple[int, List[Dict[str, Any]]]],
) -> None:
    cursor = conn.cursor()
    fact_rows = []

    for player_id, history in all_histories:
        for gw in history:
            fact_rows.append((
                player_id,
                gw.get("round"),
                gw.get("fixture"),
                gw.get("opponent_team"),
                1 if gw.get("was_home") else 0,
                gw.get("total_points"),
                gw.get("minutes"),
                gw.get("starts", 0),
                gw.get("goals_scored"),
                gw.get("assists"),
                gw.get("clean_sheets"),
                gw.get("goals_conceded"),
                gw.get("own_goals"),
                gw.get("penalties_saved"),
                gw.get("penalties_missed"),
                gw.get("yellow_cards"),
                gw.get("red_cards"),
                gw.get("saves"),
                gw.get("clearances_blocks_interceptions", 0),
                gw.get("recoveries", 0),
                gw.get("tackles", 0),
                gw.get("defensive_contribution", 0),
                gw.get("bonus"),
                gw.get("bps"),
                safe_float(gw.get("influence")),
                safe_float(gw.get("creativity")),
                safe_float(gw.get("threat")),
                safe_float(gw.get("ict_index")),
                safe_float(gw.get("expected_goals")),
                safe_float(gw.get("expected_assists")),
                safe_float(gw.get("expected_goal_involvements")),
                safe_float(gw.get("expected_goals_conceded")),
                gw.get("value"),
                gw.get("transfers_balance"),
                gw.get("selected"),
            ))

    cursor.executemany(
        """
        INSERT OR REPLACE INTO fact_player_gameweek VALUES (
            ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
        )
        """,
        fact_rows,
    )
    conn.commit()


def main() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    logging.info("Initializing / updating SQLite database schema...")
    init_db(conn)

    session = create_session()
    logging.info("Fetching general FPL bootstrap-static data...")
    bootstrap_data = fetch_bootstrap_data(session)

    logging.info("Fetching fixture list and match details...")
    fixtures_data = fetch_fixtures_data(session)

    logging.info(
        "Saving dimensions (dim_teams, dim_players) and fixtures"
        " (fact_fixtures)..."
    )
    save_dimensions(conn, bootstrap_data)
    save_fixtures(conn, fixtures_data)

    player_ids = [p["id"] for p in bootstrap_data["elements"]]
    logging.info(
        f"Fetching detailed performance history for {len(player_ids)} players..."
    )

    all_histories = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(fetch_player_history, session, pid): pid
            for pid in player_ids
        }
        completed = 0
        for future in as_completed(futures):
            pid, history = future.result()
            all_histories.append((pid, history))
            completed += 1
            if completed % 100 == 0 or completed == len(player_ids):
                logging.info(
                    f"Fetched history for {completed}/{len(player_ids)} players"
                )

    logging.info("Saving fact table (fact_player_gameweek)...")
    save_facts(conn, all_histories)

    conn.close()
    logging.info(f"Database ingestion successfully completed: {DB_PATH}")


if __name__ == "__main__":
    main()