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
                capacity_baseline REAL
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
            """
        )
        columns = [row["name"] for row in conn.execute("PRAGMA table_info(model_info)").fetchall()]
        if "model_key" not in columns:
            conn.execute("ALTER TABLE model_info ADD COLUMN model_key TEXT NOT NULL DEFAULT 'xgboost'")
        dataset_columns = [row["name"] for row in conn.execute("PRAGMA table_info(battery_dataset)").fetchall()]
        if "chemistry" not in dataset_columns:
            conn.execute("ALTER TABLE battery_dataset ADD COLUMN chemistry TEXT NOT NULL DEFAULT '未标注化学成分'")
        if "dataset_name" not in dataset_columns:
            conn.execute("ALTER TABLE battery_dataset ADD COLUMN dataset_name TEXT NOT NULL DEFAULT '演示数据集'")
        if "cell_name" not in dataset_columns:
            conn.execute("ALTER TABLE battery_dataset ADD COLUMN cell_name TEXT NOT NULL DEFAULT ''")
        if "label_status" not in dataset_columns:
            conn.execute("ALTER TABLE battery_dataset ADD COLUMN label_status TEXT NOT NULL DEFAULT '未评估'")
        if "training_eligible" not in dataset_columns:
            conn.execute("ALTER TABLE battery_dataset ADD COLUMN training_eligible INTEGER NOT NULL DEFAULT 1")
        if "quality_flags" not in dataset_columns:
            conn.execute("ALTER TABLE battery_dataset ADD COLUMN quality_flags TEXT NOT NULL DEFAULT '[]'")
        if "capacity_baseline" not in dataset_columns:
            conn.execute("ALTER TABLE battery_dataset ADD COLUMN capacity_baseline REAL")


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
