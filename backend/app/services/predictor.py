import json
from math import sqrt

import numpy as np

from ..config import EOL_SOH, MODEL_OPTIONS, model_path
from ..database import get_db, now_iso
from ..utils import row_to_dict, rows_to_dicts
from .features import extract_features


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


def _interp_soh_at_cycles(curve, target_cycles):
    cycles = np.array([p["cycle"] for p in curve], dtype=float)
    soh = np.array([p["soh"] for p in curve], dtype=float)
    return np.interp(target_cycles, cycles, soh)


def _prefix_match_score(input_curve, candidate_curve):
    input_cycles = np.array([p["cycle"] for p in input_curve], dtype=float)
    input_soh = np.array([p["soh"] for p in input_curve], dtype=float)
    if len(input_cycles) < 2:
        return 0.0

    max_cycle = min(float(input_cycles.max()), float(candidate_curve[-1]["cycle"]))
    mask = input_cycles <= max_cycle
    if mask.sum() < 2:
        return 0.0

    aligned_input = input_soh[mask]
    aligned_cycles = input_cycles[mask]
    candidate_soh = _interp_soh_at_cycles(candidate_curve, aligned_cycles)
    mae = float(np.mean(np.abs(aligned_input - candidate_soh)))
    shape_score = _corr(aligned_input, candidate_soh)
    distance_score = 1 / (1 + mae / 5)
    if not np.isfinite(shape_score):
        shape_score = distance_score
    return float(0.65 * distance_score + 0.35 * max(shape_score, 0))


def _rescale_future_curve(input_curve, matched_curve, matched_life, predicted_life, rated_capacity):
    current_cycle = max(point["cycle"] for point in input_curve)
    predicted_life = max(float(predicted_life), float(current_cycle))
    scale = predicted_life / max(float(matched_life), 1.0)
    current_soh = float(input_curve[-1]["soh"])
    matched_soh_at_current = float(_interp_soh_at_cycles(matched_curve, np.array([min(current_cycle / scale, matched_curve[-1]["cycle"])]))[0])
    offset = current_soh - matched_soh_at_current
    result = [
        {
            "cycle": int(point["cycle"]),
            "specific_capacity": round(float(point["specific_capacity"]), 4),
            "soh": round(float(point["soh"]), 4),
        }
        for point in input_curve
    ]
    existing_cycles = {point["cycle"] for point in result}
    for point in matched_curve:
        new_cycle = int(round(float(point["cycle"]) * scale))
        if new_cycle <= current_cycle or new_cycle in existing_cycles:
            continue
        progress_after_current = (new_cycle - current_cycle) / max(predicted_life - current_cycle, 1)
        adjusted_soh = float(point["soh"]) + offset * max(1 - progress_after_current, 0)
        adjusted_soh = max(min(adjusted_soh, 120), 0)
        result.append(
            {
                "cycle": new_cycle,
                "specific_capacity": round(rated_capacity * adjusted_soh / 100, 4),
                "soh": round(adjusted_soh, 4),
            }
        )
        existing_cycles.add(new_cycle)
    return sorted(result, key=lambda point: point["cycle"])


def predict_from_curve(battery_type, theoretical_capacity, rated_capacity, c_rate, curve, model_key="xgboost", username="student_demo", extra_features=None, chemistry=None):
    warnings = []
    if len(curve) < 50:
        warnings.append("上传循环数偏少，预测不确定性较高。")
    with get_db() as conn:
        if chemistry:
            rows = conn.execute(
                "SELECT * FROM battery_dataset WHERE chemistry = ? ORDER BY id DESC",
                (chemistry,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM battery_dataset WHERE battery_type = ? ORDER BY id DESC",
                (battery_type,),
            ).fetchall()
        model_row = conn.execute(
            "SELECT * FROM model_info WHERE model_key = ? AND status = 'published'",
            (model_key,),
        ).fetchone()

    if not rows:
        raise ValueError("数据库中没有匹配化学体系/类型的电池条目，请先导入训练数据。")
    if not model_row:
        raise ValueError("所选模型不存在或尚未发布，请选择教师已发布的模型。")

    ranked = []
    for row in rows:
        item = row_to_dict(row)
        score = _prefix_match_score(curve, item["capacity_curve"])
        ranked.append((score, item))
    ranked.sort(key=lambda pair: pair[0], reverse=True)

    best_score, best = ranked[0]
    top_matches = [
        {
            "cell_id": item["id"],
            "id": item["id"],
            "battery_type": item["battery_type"],
            "cell_name": item.get("cell_name"),
            "cycle_life": item["cycle_life"],
            "rated_capacity": item["rated_capacity"],
            "similarity": round(score, 4),
            "correlation_score": round(score, 4),
            "curve": item["capacity_curve"],
            "capacity_curve": item["capacity_curve"],
        }
        for score, item in ranked[:3]
    ]

    current_cycle = max(point["cycle"] for point in curve)
    current_soh = curve[-1]["soh"]
    model_prediction = None
    prediction_uncertainty = None
    try:
        import joblib

        path = model_path(model_key)
        if path.exists():
            model_payload = joblib.load(path)
            feature_names = model_payload["feature_names"]
            features = extract_features(battery_type, theoretical_capacity, rated_capacity, c_rate, curve, extra_features)
            model_prediction = round(float(model_payload["model"].predict([[features[name] for name in feature_names]])[0]), 0)
            prediction_uncertainty = model_payload.get("uncertainty_cycles")
    except Exception:
        model_prediction = None

    predicted_cycle_life = int(max(model_prediction or best["cycle_life"], current_cycle))
    remaining_life = max(int(predicted_cycle_life - current_cycle), 0)
    if prediction_uncertainty is None:
        prediction_uncertainty = max(30, int(abs(best["cycle_life"] - predicted_cycle_life) * 0.5))
    prediction_uncertainty = int(round(float(prediction_uncertainty)))
    predicted_curve = _rescale_future_curve(curve, best["capacity_curve"], best["cycle_life"], predicted_cycle_life, rated_capacity)
    if best_score < 0.55:
        warnings.append("上传数据与训练分布差异较大，请谨慎参考预测结果。")
    lower_life = max(predicted_cycle_life - prediction_uncertainty, current_cycle)
    upper_life = predicted_cycle_life + prediction_uncertainty
    soh_curve = []
    for point in predicted_curve:
        cycle = point["cycle"]
        width = max(2, prediction_uncertainty / max(predicted_cycle_life, 1) * 20)
        if cycle <= current_cycle:
            lower = upper = point["soh"]
        else:
            progress = (cycle - current_cycle) / max(predicted_cycle_life - current_cycle, 1)
            lower = max(point["soh"] - width * progress, 0)
            upper = min(point["soh"] + width * progress, 120)
        soh_curve.append({"cycle": cycle, "soh": round(point["soh"], 4), "lower": round(lower, 4), "upper": round(upper, 4)})
    model_name = model_row["model_type"]

    result = {
        "predicted_eol_cycle": predicted_cycle_life,
        "predicted_cycle_life": predicted_cycle_life,
        "remaining_cycles": remaining_life,
        "predicted_remaining_life": remaining_life,
        "prediction_uncertainty_cycles": prediction_uncertainty,
        "predicted_life_lower": lower_life,
        "predicted_life_upper": upper_life,
        "soh_at_prediction": round(float(current_soh), 2),
        "correlation_score": round(best_score, 4),
        "matched_dataset": best,
        "top_matches": top_matches,
        "input_curve": curve,
        "predicted_curve": predicted_curve,
        "soh_curve": soh_curve,
        "selected_model_key": model_key,
        "selected_model_name": model_name,
        "model_name": model_name,
        "model_predicted_life": model_prediction,
        "xgb_predicted_life": model_prediction if model_key == "xgboost" else None,
        "warnings": warnings,
    }

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO prediction_history (
                predict_time, battery_type, rated_capacity, predicted_remaining_life,
                soh_at_prediction, matched_dataset_id, correlation_score,
                input_summary, input_curve, predicted_curve, username, model_key, model_name, warnings
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        "说明": "使用输入早期循环预测完整寿命，并按最相似完整曲线外推后续 SOH 曲线。",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(curve, ensure_ascii=False),
                json.dumps(predicted_curve, ensure_ascii=False),
                username,
                model_key,
                model_name,
                json.dumps(warnings, ensure_ascii=False),
            ),
        )

    return result
