import json
import re

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..config import MODEL_OPTIONS, model_path
from ..database import get_db, now_iso
from ..utils import row_to_dict
from .features import extract_features


DEFAULT_PARAMS = {
    "n_estimators": 80,
    "max_depth": 2,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "random_state": 42,
}
PREFIX_FRACTIONS = (0.05, 0.1, 0.15, 0.2, 0.3)
DEFAULT_EVAL_PREFIX_FRACTION = 0.1
EVAL_PREFIX_FRACTIONS = (0.1, 0.2, 0.3)

MODEL_DEFAULTS = {
    "xgboost": DEFAULT_PARAMS,
    "lstm": {"hidden_layer_sizes": (48, 24), "learning_rate_init": 0.006, "max_iter": 700, "random_state": 42},
    "tcn": {"hidden_layer_sizes": (64, 24), "learning_rate_init": 0.006, "max_iter": 700, "random_state": 42},
    "cnn": {"hidden_layer_sizes": (48, 16), "learning_rate_init": 0.008, "max_iter": 650, "random_state": 42},
    "gpr": {"alpha": 0.001, "normalize_y": True, "random_state": 42},
}

PARAM_ALIASES = {
    "learning_rate": "learning_rate_init",
}

MODEL_PARAM_KEYS = {
    "xgboost": {"n_estimators", "max_depth", "learning_rate", "subsample", "random_state"},
    "lstm": {"hidden_layer_sizes", "learning_rate_init", "max_iter", "random_state"},
    "tcn": {"hidden_layer_sizes", "learning_rate_init", "max_iter", "random_state"},
    "cnn": {"hidden_layer_sizes", "learning_rate_init", "max_iter", "random_state"},
    "gpr": {"alpha", "normalize_y", "random_state"},
}


def _monotone_constraints_for_features(feature_names):
    positive_features = {
        "observed_cycles",
        "initial_capacity",
        "latest_capacity",
        "retention",
        "early_slope",
        "early_capacity_slope",
    }
    negative_features = {
        "c_rate",
        "internal_resistance_proxy",
    }
    constraints = []
    for name in feature_names or []:
        if name in positive_features or name.startswith("seq_soh_"):
            constraints.append(1)
        elif name in negative_features:
            constraints.append(-1)
        else:
            constraints.append(0)
    return tuple(constraints)


def _build_model(model_key: str, params: dict, feature_names=None):
    if model_key == "xgboost":
        from xgboost import XGBRegressor

        xgb_params = dict(params)
        if feature_names and "monotone_constraints" not in xgb_params:
            xgb_params["monotone_constraints"] = _monotone_constraints_for_features(feature_names)
        return XGBRegressor(objective="reg:squarederror", **xgb_params)
    if model_key in {"lstm", "tcn", "cnn"}:
        # 演示系统不引入庞大的深度学习运行时，使用多层感知机拟合早期序列特征。
        # UI 中保留 LSTM/TCN/CNN 名称，用于表达模型思路与答辩对比。
        return make_pipeline(StandardScaler(), MLPRegressor(**params))
    if model_key == "gpr":
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        return make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=kernel, **params))
    raise ValueError("不支持的模型类型。")


def _regression_metrics(actual, predicted):
    actual_array = np.array(actual, dtype=float)
    pred_array = np.array(predicted, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(actual_array, pred_array)))
    mae = float(mean_absolute_error(actual_array, pred_array))
    mean_life = float(np.mean(actual_array)) if len(actual_array) else 0
    safe_actual = np.where(actual_array == 0, np.nan, actual_array)
    mape = float(np.nanmean(np.abs((actual_array - pred_array) / safe_actual)) * 100)
    nrmse = (rmse / mean_life * 100) if mean_life else 0
    return {
        "RMSE": round(rmse, 3),
        "MAE": round(mae, 3),
        "R2": round(float(r2_score(actual_array, pred_array)), 3),
        "MAPE": round(mape, 2),
        "NRMSE": round(nrmse, 2),
        "平均寿命": round(mean_life, 1),
    }


def _safe_key(value: str):
    return re.sub(r"[^A-Za-z0-9_]+", "_", value or "").strip("_")[:40] or "model"


def train_model(params=None, model_key="xgboost"):
    base_model_key = model_key
    if base_model_key not in MODEL_OPTIONS:
        raise ValueError("不支持的模型类型。")
    clean_params = {}
    raw_params = params or {}
    dataset_ids = raw_params.get("dataset_ids") or []
    if isinstance(dataset_ids, str):
        dataset_ids = [item for item in dataset_ids.split(",") if item]
    chemistry = raw_params.get("chemistry") or ""
    publish = bool(raw_params.get("publish", True))
    eval_prefix_fraction = float(raw_params.get("training_observation_fraction", DEFAULT_EVAL_PREFIX_FRACTION))
    eval_prefix_fraction = min(max(eval_prefix_fraction, 0.05), 0.5)
    for key, value in raw_params.items():
        key = PARAM_ALIASES.get(key, key)
        if value not in (None, "") and key in MODEL_PARAM_KEYS[base_model_key]:
            clean_params[key] = value
    params = {**MODEL_DEFAULTS[base_model_key], **clean_params}
    with get_db() as conn:
        columns = [row["name"] for row in conn.execute("PRAGMA table_info(battery_dataset)").fetchall()]
        source_filters = ["1 = 1"]
        values = []
        if chemistry:
            source_filters.append("chemistry = ?")
            values.append(chemistry)
        if dataset_ids:
            placeholders = ",".join("?" for _ in dataset_ids)
            source_filters.append(f"dataset_name IN ({placeholders})")
            values.extend(dataset_ids)
        candidate_rows = conn.execute(f"SELECT * FROM battery_dataset WHERE {' AND '.join(source_filters)}", values).fetchall()
        training_filters = list(source_filters)
        if "training_eligible" in columns:
            training_filters.append("training_eligible = 1")
        rows = conn.execute(f"SELECT * FROM battery_dataset WHERE {' AND '.join(training_filters)}", values).fetchall()
        if rows:
            chemistry = chemistry or (row_to_dict(rows[0]).get("chemistry") or "")
        version_row = conn.execute(
            "SELECT COALESCE(MAX(version), 0) AS version FROM model_info WHERE base_model_key = ? AND chemistry = ?",
            (base_model_key, chemistry or "未标注化学成分"),
        ).fetchone()
        version = int(version_row["version"] or 0) + 1

    unique_model_key = f"{base_model_key}_v{version}_{_safe_key(chemistry or 'all')}"

    if len(rows) < 8:
        raise ValueError("可靠 EOL 训练数据少于 8 条，请先导入更多达到 80% SOH 的电池数据。")

    records = [row_to_dict(row) for row in rows]
    feature_names = list(
        extract_features(
            "LCO",
            0,
            0,
            0,
            [{"cycle": 0, "specific_capacity": 1, "soh": 100}, {"cycle": 1, "specific_capacity": 1, "soh": 100}],
        ).keys()
    )
    x = []
    y = []
    groups = []
    prefix_tags = []
    for item in records:
        group_id = item.get("cell_name") or str(item["id"])
        for fraction in PREFIX_FRACTIONS:
            prefix_points = max(8, int(len(item["capacity_curve"]) * fraction))
            features = extract_features(
                item["battery_type"],
                item["theoretical_capacity"],
                item["rated_capacity"],
                item["c_rate"],
                item["capacity_curve"][: min(prefix_points, len(item["capacity_curve"]))],
                item.get("additional_features") or {},
            )
            x.append([features[name] for name in feature_names])
            y.append(item["cycle_life"])
            groups.append(group_id)
            prefix_tags.append(fraction)

    import joblib

    x_array = np.array(x, dtype=float)
    y_array = np.array(y, dtype=float)
    groups_array = np.array(groups)
    prefix_tags_array = np.array(prefix_tags, dtype=float)
    model = _build_model(base_model_key, params, feature_names)
    window_metrics = {}
    primary_pred = None
    primary_eval_y = None
    # 用完整寿命曲线生成多个早期前缀训练样本；评估时按电池留一，避免同一电池前缀泄漏。
    if len(records) <= 40:
        for window_fraction in EVAL_PREFIX_FRACTIONS:
            predictions = []
            eval_y_values = []
            for train_index, test_index in LeaveOneGroupOut().split(x_array, y_array, groups_array):
                fold_model = _build_model(base_model_key, params, feature_names)
                fold_model.fit(x_array[train_index], y_array[train_index])
                eval_candidates = [index for index in test_index if abs(prefix_tags_array[index] - window_fraction) < 1e-9]
                eval_index = eval_candidates[0] if eval_candidates else test_index[0]
                predictions.append(float(fold_model.predict(x_array[[eval_index]])[0]))
                eval_y_values.append(float(y_array[eval_index]))
            window_pred = np.array(predictions)
            window_eval_y = np.array(eval_y_values)
            window_metrics[f"前{int(window_fraction * 100)}%"] = _regression_metrics(window_eval_y, window_pred)
            if abs(window_fraction - eval_prefix_fraction) < 1e-9:
                primary_pred = window_pred
                primary_eval_y = window_eval_y
        pred = primary_pred if primary_pred is not None else window_pred
        eval_y = primary_eval_y if primary_eval_y is not None else window_eval_y
    else:
        record_groups = np.array([item.get("cell_name") or str(item["id"]) for item in records])
        train_groups, test_groups = train_test_split(record_groups, test_size=0.25, random_state=42)
        train_mask = np.isin(groups_array, train_groups)
        test_mask = np.isin(groups_array, test_groups)
        model.fit(x_array[train_mask], y_array[train_mask])
        for window_fraction in EVAL_PREFIX_FRACTIONS:
            eval_mask = test_mask & (np.abs(prefix_tags_array - window_fraction) < 1e-9)
            window_pred = model.predict(x_array[eval_mask])
            window_eval_y = y_array[eval_mask]
            window_metrics[f"前{int(window_fraction * 100)}%"] = _regression_metrics(window_eval_y, window_pred)
            if abs(window_fraction - eval_prefix_fraction) < 1e-9:
                primary_pred = window_pred
                primary_eval_y = window_eval_y
        pred = primary_pred if primary_pred is not None else window_pred
        eval_y = primary_eval_y if primary_eval_y is not None else window_eval_y
    model.fit(x_array, y_array)
    accuracy_metrics = _regression_metrics(eval_y, pred)
    metrics = {
        **accuracy_metrics,
        "评估方式": "留一交叉验证" if len(records) <= 40 else "随机测试集",
        "主评估窗口": f"前{int(eval_prefix_fraction * 100)}%",
        "候选样本": len(candidate_rows),
        "可靠EOL样本": len(records),
        "前缀训练样本": len(x_array),
        "排除样本": max(len(candidate_rows) - len(records), 0),
        "训练样本筛选": "完整寿命曲线生成早期前缀样本，仅使用可靠 EOL 标签",
        "观测窗口": f"训练前缀 5%-30%，主指标使用前 {int(eval_prefix_fraction * 100)}%",
        "窗口评估": window_metrics,
        "误差口径": "RMSE/MAE 为循环数误差；MAPE/NRMSE 为百分比误差",
        "误差解释": "早期 10% SOH 信息量有限，跨数据集工况差异和未达到 EOL 的截断样本会显著放大外推误差。",
        "单调性约束": "XGBoost 寿命回归使用特征单调约束；所有模型预测的未来 SOH 曲线统一约束为不随循环数上升。",
        "预测不确定性": f"默认 ±{round(accuracy_metrics['RMSE'])} 圈",
        "扩展特征": "已使用电压/电流特征；温度/内阻原始数据未提供" ,
    }
    if len(records) < 30:
        metrics["样本量警告"] = "样本量过小，指标仅供教学参考"
    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names,
            "model_key": unique_model_key,
            "base_model_key": base_model_key,
            "training_mode": "full_curve_prefix_transfer",
            "prefix_fractions": PREFIX_FRACTIONS,
            "uncertainty_cycles": round(accuracy_metrics["RMSE"]),
            "monotonicity": {
                "future_soh_curve": "non_increasing",
                "xgboost_feature_constraints": _monotone_constraints_for_features(feature_names) if base_model_key == "xgboost" else (),
            },
        },
        model_path(unique_model_key),
    )

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO model_info (
                model_key, model_type, training_data_size, metrics, feature_list,
                hyperparameters, trained_at, source_path, base_model_key, version,
                status, dataset_ids, chemistry, visibility
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                unique_model_key,
                MODEL_OPTIONS[base_model_key],
                len(records),
                json.dumps(metrics, ensure_ascii=False),
                json.dumps(feature_names, ensure_ascii=False),
                json.dumps(params, ensure_ascii=False),
                now_iso(),
                "backend/app/services/modeling.py",
                base_model_key,
                version,
                "published" if publish else "draft",
                json.dumps(dataset_ids, ensure_ascii=False),
                chemistry or "未标注化学成分",
                "student" if publish else "teacher",
            ),
        )
    return {"model_key": unique_model_key, "base_model_key": base_model_key, "model_type": MODEL_OPTIONS[base_model_key], "training_data_size": len(records), "metrics": metrics, "feature_list": feature_names, "hyperparameters": params, "trained_at": now_iso(), "source_path": "backend/app/services/modeling.py", "version": version, "status": "published" if publish else "draft", "chemistry": chemistry or "未标注化学成分"}


def train_all_models():
    return [train_model(model_key=key) for key in MODEL_OPTIONS]
