import sqlite3
import pandas as pd
from datetime import datetime

DB_NAME = "valuations.db"

def init_db():
    """Creates the valuation tracking table if it does not exist."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS valuation_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            ticker TEXT,
            current_price REAL,
            median_val REAL,
            p10_val REAL,
            p90_val REAL,
            scenario TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_run(ticker, current_price, median_val, p10_val, p90_val, scenario):
    """Logs a single valuation simulation run."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO valuation_runs (timestamp, ticker, current_price, median_val, p10_val, p90_val, scenario)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (datetime.now().strftime("%Y-%m-%d %H:%M"), ticker.upper(), current_price, median_val, p10_val, p90_val, scenario))
    conn.commit()
    conn.close()

def get_history(ticker=None):
    """Retrieves past valuation runs from the database."""
    conn = sqlite3.connect(DB_NAME)
    if ticker:
        df = pd.read_sql_query("SELECT * FROM valuation_runs WHERE ticker = ? ORDER BY timestamp DESC", conn, params=(ticker.upper(),))
    else:
        df = pd.read_sql_query("SELECT * FROM valuation_runs ORDER BY timestamp DESC", conn)
    conn.close()
    return df
