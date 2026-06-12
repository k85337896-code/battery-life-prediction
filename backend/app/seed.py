import json
import random
from pathlib import Path

import numpy as np

from .config import DB_PATH
from .auth import hash_password
from .database import get_db, init_db, now_iso
from .services.modeling import train_all_models


TYPE_SETTINGS = {
    "LCO": {"base": 168, "life": (520, 850), "fade": 0.24},
    "LFP": {"base": 155, "life": (900, 1500), "fade": 0.17},
    "LS": {"base": 850, "life": (220, 520), "fade": 0.34},
}


def make_curve(battery_type: str, life: int, rated_capacity: float):
    points = []
    step = max(life // 90, 5)
    for cycle in range(0, life + step, step):
        progress = min(cycle / life, 1)
        # 合成曲线采用指数衰减叠加轻微噪声，便于演示但不声称代表真实实验数据。
        soh = 100 - 20 * (progress ** (1 + TYPE_SETTINGS[battery_type]["fade"])) - random.uniform(0, 1.2)
        soh = max(78.5, soh)
        specific_capacity = rated_capacity * soh / 100
        points.append({"cycle": cycle, "specific_capacity": round(specific_capacity, 4), "soh": round(soh, 4)})
    return points


def seed(force=False):
    init_db()
    ensure_demo_users()
    if force and DB_PATH.exists():
        with get_db() as conn:
            conn.execute("DELETE FROM prediction_history")
            conn.execute("DELETE FROM battery_dataset")
            conn.execute("DELETE FROM model_info")

    with get_db() as conn:
        count = conn.execute("SELECT COUNT(*) AS c FROM battery_dataset").fetchone()["c"]
        if count:
            return {"message": "种子数据已存在", "count": count}

    random.seed(42)
    np.random.seed(42)
    with get_db() as conn:
        for battery_type, cfg in TYPE_SETTINGS.items():
            for index in range(26):
                life = random.randint(*cfg["life"])
                rated = cfg["base"] * random.uniform(0.92, 1.08)
                theoretical = rated * random.uniform(1.04, 1.18)
                c_rate = random.choice([0.3, 0.5, 0.8, 1.0, 1.5])
                curve = make_curve(battery_type, life, rated)
                conn.execute(
                    """
                    INSERT INTO battery_dataset (
                        battery_type, theoretical_capacity, rated_capacity, c_rate,
                        cycle_life, current_soh, capacity_curve, source, note, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        battery_type,
                        round(theoretical, 3),
                        round(rated, 3),
                        c_rate,
                        life,
                        curve[-1]["soh"],
                        json.dumps(curve, ensure_ascii=False),
                        "合成演示数据",
                        f"{battery_type}-{index + 1:02d}，可替换为真实实验记录",
                        now_iso(),
                    ),
                )

    sample_dir = Path(__file__).resolve().parents[2] / "sample_data"
    sample_dir.mkdir(exist_ok=True)
    for battery_type in ["LCO", "LFP", "LS"]:
        cfg = TYPE_SETTINGS[battery_type]
        life = random.randint(*cfg["life"])
        rated = cfg["base"] * random.uniform(0.95, 1.05)
        curve = make_curve(battery_type, life, rated)[:32]
        with (sample_dir / f"{battery_type}_sample.csv").open("w", encoding="utf-8") as f:
            f.write("cycle,specific_capacity\n")
            for point in curve:
                f.write(f"{point['cycle']},{point['specific_capacity']}\n")

    try:
        train_all_models()
    except Exception as exc:
        return {"message": f"种子数据已生成，模型训练暂未完成：{exc}", "count": 78}
    return {"message": "种子数据与模型已生成", "count": 78}


def ensure_demo_users():
    with get_db() as conn:
        for username, password, role, display_name in (
            ("teacher_demo", "123456", "teacher", "教师演示账号"),
            ("student_demo", "123456", "student", "学生演示账号"),
            ("teacher", "123456", "teacher", "教师演示账号"),
            ("student", "123456", "student", "学生演示账号"),
        ):
            conn.execute(
                """
                INSERT OR REPLACE INTO users (username, password_hash, role, display_name, created_at)
                VALUES (?, ?, ?, ?, COALESCE((SELECT created_at FROM users WHERE username = ?), ?))
                """,
                (username, hash_password(password), role, display_name, username, now_iso()),
            )


if __name__ == "__main__":
    print(seed(force=True))
