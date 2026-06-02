import json
from math import sqrt

import numpy as np

from ..config import EOL_SOH, MODEL_OPTIONS, model_path
from ..database import get_db, now_iso
from ..utils import row_to_dict, rows_to_dicts


def _normalized_soh(curve, grid_size=120):
    cycles = np.array([p["cycle"] for p in curve], dtype=float)
    soh = np.array([p["soh"] for p in curve], dtype=float)
    if cycles.max() == cycles.min():
        x = np.zeros_like(cycles)
    else:
        x = (cycles - cycles.min()) / (cycles.max() - cycles.min())
    grid = np.linspace(0, 1, grid_size)
    return np.interp(grid, x, soh)


def _corr(a, b):
    if np.std(a) < 1e-6 or np.std(b) < 1e-6:
        distance = np.linalg.norm(a - b)
        return float(1 / (1 + distance / sqrt(len(a))))
    return float(np.corrcoef(a, b)[0, 1])


def extract_features(battery_type, theoretical_capacity, rated_capacity, c_rate, curve):
    life_hint = max(point["cycle"] for point in curve)
    first = curve[0]["specific_capacity"]
    last = curve[-1]["specific_capacity"]
    retention = last / first if first else 0
    slope = (last - first) / max(life_hint, 1)
    base_features = {
        "type_LCO": 1 if battery_type == "LCO" else 0,
        "type_LFP": 1 if battery_type == "LFP" else 0,
        "type_LS": 1 if battery_type == "LS" else 0,
        "type_G1": 1 if battery_type == "G1" else 0,
        "type_G2": 1 if battery_type == "G2" else 0,
        "type_G3": 1 if battery_type == "G3" else 0,
        "type_G4": 1 if battery_type == "G4" else 0,
        "theoretical_capacity": theoretical_capacity,
        "rated_capacity": rated_capacity,
        "c_rate": c_rate,
        "observed_cycles": life_hint,
        "initial_capacity": first,
        "latest_capacity": last,
        "retention": retention,
        "early_slope": slope,
    }
    sampled_soh = _normalized_soh(curve, grid_size=16)
    for index, value in enumerate(sampled_soh):
        base_features[f"seq_soh_{index:02d}"] = float(value)
    return base_features


def predict_from_curve(battery_type, theoretical_capacity, rated_capacity, c_rate, curve, model_key="xgboost"):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM battery_dataset WHERE battery_type = ? ORDER BY id DESC",
            (battery_type,),
        ).fetchall()

    if not rows:
        raise ValueError("数据库中没有同类型电池条目，请先导入或生成种子数据。")

    input_norm = _normalized_soh(curve)
    ranked = []
    for row in rows:
        item = row_to_dict(row)
        score = _corr(input_norm, _normalized_soh(item["capacity_curve"]))
        ranked.append((score, item))
    ranked.sort(key=lambda pair: pair[0], reverse=True)

    best_score, best = ranked[0]
    top_matches = [
        {
            "id": item["id"],
            "battery_type": item["battery_type"],
            "cycle_life": item["cycle_life"],
            "rated_capacity": item["rated_capacity"],
            "correlation_score": round(score, 4),
            "capacity_curve": item["capacity_curve"],
        }
        for score, item in ranked[:3]
    ]

    current_cycle = max(point["cycle"] for point in curve)
    current_soh = curve[-1]["soh"]
    remaining_life = max(int(best["cycle_life"] - current_cycle), 0)

    model_prediction = None
    try:
        import joblib

        path = model_path(model_key)
        if path.exists():
            model_payload = joblib.load(path)
            feature_names = model_payload["feature_names"]
            features = extract_features(battery_type, theoretical_capacity, rated_capacity, c_rate, curve)
            model_prediction = round(float(model_payload["model"].predict([[features[name] for name in feature_names]])[0]), 0)
    except Exception:
        model_prediction = None

    result = {
        "predicted_cycle_life": int(best["cycle_life"]),
        "predicted_remaining_life": remaining_life,
        "soh_at_prediction": round(float(current_soh), 2),
        "correlation_score": round(best_score, 4),
        "matched_dataset": best,
        "top_matches": top_matches,
        "input_curve": curve,
        "predicted_curve": best["capacity_curve"],
        "selected_model_key": model_key,
        "selected_model_name": MODEL_OPTIONS.get(model_key, model_key),
        "model_predicted_life": model_prediction,
        "xgb_predicted_life": model_prediction if model_key == "xgboost" else None,
    }

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO prediction_history (
                predict_time, battery_type, rated_capacity, predicted_remaining_life,
                soh_at_prediction, matched_dataset_id, correlation_score,
                input_summary, input_curve, predicted_curve
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now_iso(),
                battery_type,
                rated_capacity,
                remaining_life,
                current_soh,
                best["id"],
                best_score,
                json.dumps(
                    {
                        "theoretical_capacity": theoretical_capacity,
                        "rated_capacity": rated_capacity,
                        "c_rate": c_rate,
                        "说明": "预测剩余寿命=匹配条目寿命-输入曲线最大循环次数。",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(curve, ensure_ascii=False),
                json.dumps(best["capacity_curve"], ensure_ascii=False),
            ),
        )

    return result
