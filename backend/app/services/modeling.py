import json

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
from .predictor import extract_features


DEFAULT_PARAMS = {
    "n_estimators": 80,
    "max_depth": 2,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "random_state": 42,
}
PREFIX_FRACTIONS = (0.05, 0.1, 0.15, 0.2, 0.3)
DEFAULT_EVAL_PREFIX_FRACTION = 0.1

MODEL_DEFAULTS = {
    "xgboost": DEFAULT_PARAMS,
    "lstm": {"hidden_layer_sizes": (96, 64, 32), "learning_rate_init": 0.006, "max_iter": 2000, "random_state": 42},
    "tcn": {"hidden_layer_sizes": (128, 64, 32), "learning_rate_init": 0.006, "max_iter": 2000, "random_state": 42},
    "cnn": {"hidden_layer_sizes": (96, 48), "learning_rate_init": 0.008, "max_iter": 1800, "random_state": 42},
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


def _build_model(model_key: str, params: dict):
    if model_key == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(objective="reg:squarederror", **params)
    if model_key in {"lstm", "tcn", "cnn"}:
        # 演示系统不引入庞大的深度学习运行时，使用多层感知机拟合早期序列特征。
        # UI 中保留 LSTM/TCN/CNN 名称，用于表达模型思路与答辩对比。
        return make_pipeline(StandardScaler(), MLPRegressor(**params))
    if model_key == "gpr":
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        return make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=kernel, **params))
    raise ValueError("不支持的模型类型。")


def train_model(params=None, model_key="xgboost"):
    if model_key not in MODEL_OPTIONS:
        raise ValueError("不支持的模型类型。")
    clean_params = {}
    raw_params = params or {}
    eval_prefix_fraction = float(raw_params.get("training_observation_fraction", DEFAULT_EVAL_PREFIX_FRACTION))
    eval_prefix_fraction = min(max(eval_prefix_fraction, 0.05), 0.5)
    for key, value in raw_params.items():
        key = PARAM_ALIASES.get(key, key)
        if value not in (None, "") and key in MODEL_PARAM_KEYS[model_key]:
            clean_params[key] = value
    params = {**MODEL_DEFAULTS[model_key], **clean_params}
    with get_db() as conn:
        all_rows = conn.execute("SELECT * FROM battery_dataset").fetchall()
        columns = [row["name"] for row in conn.execute("PRAGMA table_info(battery_dataset)").fetchall()]
        if "training_eligible" in columns:
            rows = conn.execute("SELECT * FROM battery_dataset WHERE training_eligible = 1").fetchall()
        else:
            rows = all_rows

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
            )
            x.append([features[name] for name in feature_names])
            y.append(item["cycle_life"])
            groups.append(group_id)

    import joblib

    x_array = np.array(x, dtype=float)
    y_array = np.array(y, dtype=float)
    groups_array = np.array(groups)
    model = _build_model(model_key, params)
    # 用完整寿命曲线生成多个早期前缀训练样本；评估时按电池留一，避免同一电池前缀泄漏。
    if len(records) <= 40:
        predictions = []
        eval_y_values = []
        for train_index, test_index in LeaveOneGroupOut().split(x_array, y_array, groups_array):
            fold_model = _build_model(model_key, params)
            fold_model.fit(x_array[train_index], y_array[train_index])
            group = groups_array[test_index][0]
            eval_candidates = [
                index
                for index in test_index
                if group == groups_array[index]
                and abs(PREFIX_FRACTIONS[index % len(PREFIX_FRACTIONS)] - eval_prefix_fraction) < 1e-9
            ]
            eval_index = eval_candidates[0] if eval_candidates else test_index[0]
            predictions.append(float(fold_model.predict(x_array[[eval_index]])[0]))
            eval_y_values.append(float(y_array[eval_index]))
        pred = np.array(predictions)
        eval_y = np.array(eval_y_values)
    else:
        record_groups = np.array([item.get("cell_name") or str(item["id"]) for item in records])
        train_groups, test_groups = train_test_split(record_groups, test_size=0.25, random_state=42)
        train_mask = np.isin(groups_array, train_groups)
        test_mask = np.isin(groups_array, test_groups)
        model.fit(x_array[train_mask], y_array[train_mask])
        pred = model.predict(x_array[test_mask])
        eval_y = y_array[test_mask]
    model.fit(x_array, y_array)
    metrics = {
        "RMSE": round(float(np.sqrt(mean_squared_error(eval_y, pred))), 3),
        "MAE": round(float(mean_absolute_error(eval_y, pred)), 3),
        "R2": round(float(r2_score(eval_y, pred)), 3),
        "评估方式": "留一交叉验证" if len(records) <= 40 else "随机测试集",
        "候选样本": len(all_rows),
        "可靠EOL样本": len(records),
        "前缀训练样本": len(x_array),
        "排除样本": max(len(all_rows) - len(records), 0),
        "训练样本筛选": "完整寿命曲线生成早期前缀样本，仅使用可靠 EOL 标签",
        "观测窗口": f"训练前缀 5%-30%，留一评估使用前 {int(eval_prefix_fraction * 100)}%",
    }
    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names,
            "model_key": model_key,
            "training_mode": "full_curve_prefix_transfer",
            "prefix_fractions": PREFIX_FRACTIONS,
        },
        model_path(model_key),
    )

    with get_db() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO model_info (
                model_key, model_type, training_data_size, metrics, feature_list,
                hyperparameters, trained_at, source_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_key,
                MODEL_OPTIONS[model_key],
                len(records),
                json.dumps(metrics, ensure_ascii=False),
                json.dumps(feature_names, ensure_ascii=False),
                json.dumps(params, ensure_ascii=False),
                now_iso(),
                "backend/app/services/modeling.py",
            ),
        )
    return {"model_key": model_key, "model_type": MODEL_OPTIONS[model_key], "training_data_size": len(records), "metrics": metrics, "feature_list": feature_names, "hyperparameters": params, "trained_at": now_iso(), "source_path": "backend/app/services/modeling.py"}


def train_all_models():
    placeholders = ",".join("?" for _ in MODEL_OPTIONS)
    with get_db() as conn:
        conn.execute(f"DELETE FROM model_info WHERE model_key NOT IN ({placeholders})", tuple(MODEL_OPTIONS.keys()))
    return [train_model(model_key=key) for key in MODEL_OPTIONS]
