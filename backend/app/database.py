import sqlite3
from contextlib import contextmanager
from datetime import datetime

from .config import DB_PATH


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        existing = conn.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'model_info'").fetchone()
        if existing and "CHECK (id = 1)" in existing["sql"]:
            conn.execute("DROP TABLE model_info")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS battery_dataset (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                battery_type TEXT NOT NULL,
                theoretical_capacity REAL NOT NULL,
                rated_capacity REAL NOT NULL,
                c_rate REAL NOT NULL,
                cycle_life INTEGER NOT NULL,
                current_soh REAL NOT NULL,
                capacity_curve TEXT NOT NULL,
                source TEXT,
                note TEXT,
                created_at TEXT NOT NULL,
                label_status TEXT NOT NULL DEFAULT '未评估',
                training_eligible INTEGER NOT NULL DEFAULT 1,
                quality_flags TEXT NOT NULL DEFAULT '[]',
                capacity_baseline REAL,
                additional_features TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('teacher', 'student')),
                display_name TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                predict_time TEXT NOT NULL,
                battery_type TEXT NOT NULL,
                rated_capacity REAL NOT NULL,
                predicted_remaining_life INTEGER NOT NULL,
                soh_at_prediction REAL NOT NULL,
                matched_dataset_id INTEGER,
                correlation_score REAL NOT NULL,
                input_summary TEXT NOT NULL,
                input_curve TEXT NOT NULL,
                predicted_curve TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS model_info (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_key TEXT NOT NULL UNIQUE DEFAULT 'xgboost',
                model_type TEXT NOT NULL,
                training_data_size INTEGER NOT NULL,
                metrics TEXT NOT NULL,
                feature_list TEXT NOT NULL,
                hyperparameters TEXT NOT NULL,
                trained_at TEXT NOT NULL,
                source_path TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS training_jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                progress INTEGER NOT NULL DEFAULT 0,
                message TEXT NOT NULL DEFAULT '',
                result TEXT NOT NULL DEFAULT '{}',
                error TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        ensure_column(conn, "model_info", "model_key", "TEXT NOT NULL DEFAULT 'xgboost'")
        ensure_column(conn, "model_info", "base_model_key", "TEXT NOT NULL DEFAULT 'xgboost'")
        ensure_column(conn, "model_info", "version", "INTEGER NOT NULL DEFAULT 1")
        ensure_column(conn, "model_info", "status", "TEXT NOT NULL DEFAULT 'published'")
        ensure_column(conn, "model_info", "dataset_ids", "TEXT NOT NULL DEFAULT '[]'")
        ensure_column(conn, "model_info", "chemistry", "TEXT NOT NULL DEFAULT '未标注化学成分'")
        ensure_column(conn, "model_info", "visibility", "TEXT NOT NULL DEFAULT 'teacher'")
        ensure_column(conn, "battery_dataset", "chemistry", "TEXT NOT NULL DEFAULT '未标注化学成分'")
        ensure_column(conn, "battery_dataset", "dataset_name", "TEXT NOT NULL DEFAULT '演示数据集'")
        ensure_column(conn, "battery_dataset", "cell_name", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "battery_dataset", "label_status", "TEXT NOT NULL DEFAULT '未评估'")
        ensure_column(conn, "battery_dataset", "training_eligible", "INTEGER NOT NULL DEFAULT 1")
        ensure_column(conn, "battery_dataset", "quality_flags", "TEXT NOT NULL DEFAULT '[]'")
        ensure_column(conn, "battery_dataset", "capacity_baseline", "REAL")
        ensure_column(conn, "battery_dataset", "additional_features", "TEXT NOT NULL DEFAULT '{}'")
        ensure_column(conn, "prediction_history", "username", "TEXT NOT NULL DEFAULT 'student_demo'")
        ensure_column(conn, "prediction_history", "model_key", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "prediction_history", "model_name", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "prediction_history", "warnings", "TEXT NOT NULL DEFAULT '[]'")
        conn.execute(
            """
            UPDATE model_info
            SET base_model_key = model_key
            WHERE model_key IN ('xgboost', 'lstm', 'tcn', 'cnn', 'gpr')
              AND base_model_key = 'xgboost'
            """
        )
        conn.execute("UPDATE battery_dataset SET chemistry = '实验组锂离子电池' WHERE chemistry = '未标注化学成分'")
        conn.execute("UPDATE model_info SET chemistry = '实验组锂离子电池' WHERE chemistry = '未标注化学成分'")


def ensure_column(conn, table: str, column: str, definition: str):
    columns = [row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
