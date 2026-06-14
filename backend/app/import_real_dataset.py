import json
import re
from pathlib import Path

import pandas as pd

from .database import get_db, init_db, now_iso
from .services.modeling import train_all_models


DATASET_DIR = Path(r"D:\Users\cxy\桌面\电压电流")
HUST_STANDARDIZED_DIR = Path(r"D:\Users\cxy\battery-processed\HUST\standardized_csv")
TJU_NCA_DIR = Path(r"D:\Users\cxy\battery-processed\TJU\Dataset_1_NCA_battery")
TJU_NCM_DIR = Path(r"D:\Users\cxy\battery-processed\TJU\Dataset_2_NCM_battery")
REAL_DATASET_NAME = "电压电流真实数据集"
REAL_CHEMISTRY = "实验组锂金属电池"
EOL_SOH = 80
SUSTAINED_EOL_POINTS = 3
HIGH_ERROR_OUTLIER_CELLS = {"G2_Cell3", "G2_Cell1", "G1_Cell3", "G4_Cell1"}
MAX_STORED_CURVE_POINTS = 650


def _file_meta(path: Path):
    match = re.match(r"(G\d+)_Cell(\d+)_Data\.xlsx", path.name, re.IGNORECASE)
    if not match:
        raise ValueError(f"无法从文件名识别组别和电池编号：{path.name}")
    return match.group(1).upper(), int(match.group(2))


def _baseline_capacity(grouped: pd.DataFrame) -> float:
    window_size = min(max(20, int(len(grouped) * 0.05)), 50, len(grouped))
    early = grouped.head(window_size)["capacity"].sort_values(ascending=False)
    top_count = min(5, len(early))
    return float(early.head(top_count).median())


def _find_sustained_eol(curve: list[dict]):
    for index in range(0, len(curve) - SUSTAINED_EOL_POINTS + 1):
        window = curve[index : index + SUSTAINED_EOL_POINTS]
        if all(point["soh"] <= EOL_SOH for point in window):
            return curve[index]
    return None


def _downsample_curve(curve: list[dict], max_points: int = MAX_STORED_CURVE_POINTS):
    if len(curve) <= max_points:
        return curve
    indexes = sorted(set(int(round(i)) for i in pd.Series(range(max_points)).mul((len(curve) - 1) / max(max_points - 1, 1))))
    return [curve[index] for index in indexes]


def _quality_summary(curve: list[dict], eol_point: dict | None, allow_terminal_label: bool = False):
    soh_values = [point["soh"] for point in curve]
    below_points = [point for point in curve if point["soh"] <= EOL_SOH]
    upward_steps = sum(1 for before, after in zip(soh_values, soh_values[1:]) if after - before > 0.5)
    big_jumps = sum(1 for before, after in zip(soh_values, soh_values[1:]) if abs(after - before) > 5)
    flags = []

    if max(soh_values) > 105:
        flags.append("早期容量激活明显，SOH 曾超过 105%")
    if upward_steps > max(20, len(curve) * 0.08):
        flags.append("容量曲线波动偏多")
    if big_jumps:
        flags.append(f"存在 {big_jumps} 次超过 5% 的 SOH 跳变")

    if eol_point:
        if curve[-1]["soh"] > EOL_SOH:
            label_status = "低于80%后回升"
            training_eligible = 0
            flags.append("首次低于 80% 后又回升，寿命标签不稳定")
        else:
            label_status = "可靠EOL"
            training_eligible = 1
    elif below_points:
        label_status = "低于80%但未持续"
        training_eligible = 0
        flags.append("低于 80% 未连续保持，暂不作为可靠寿命标签")
    else:
        label_status = "未达到EOL"
        training_eligible = 1 if allow_terminal_label and len(curve) >= 30 else 0
        flags.append("记录结束时尚未达到 80% SOH，只能视为寿命下限")
        if allow_terminal_label and training_eligible:
            label_status = "寿命下限标签"
            flags.append("公开数据集未达到 EOL，使用末端循环作为教学训练标签")

    return label_status, training_eligible, flags


def _curve_from_excel(path: Path):
    header = pd.read_excel(path, sheet_name="all_data", nrows=0, engine="openpyxl").columns
    capacity_col = "capacity_mAh" if "capacity_mAh" in header else "capacity_Ah" if "capacity_Ah" in header else None
    if not capacity_col:
        raise ValueError(f"{path.name} 未找到 capacity_mAh/capacity_Ah 列。")

    optional_cols = [col for col in ("voltage_V", "current_mA", "temperature_C", "internal_resistance_mOhm") if col in header]
    df = pd.read_excel(path, sheet_name="all_data", usecols=["cycle", "phase", capacity_col, *optional_cols], engine="openpyxl")
    df = df.rename(columns={capacity_col: "capacity"})
    if capacity_col == "capacity_Ah":
        df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce") * 1000

    df = df.dropna(subset=["cycle", "phase", "capacity"])
    df["cycle"] = pd.to_numeric(df["cycle"], errors="coerce")
    df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce")
    for col in optional_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["cycle", "capacity"])

    discharge = df[df["phase"].astype(str).str.lower().str.contains("discharge")]
    if discharge.empty:
        discharge = df

    aggregations = {"capacity": "max"}
    if "voltage_V" in optional_cols:
        aggregations["voltage_V"] = "mean"
    if "current_mA" in optional_cols:
        aggregations["current_mA"] = "mean"
    if "temperature_C" in optional_cols:
        aggregations["temperature_C"] = "mean"
    if "internal_resistance_mOhm" in optional_cols:
        aggregations["internal_resistance_mOhm"] = "mean"
    grouped = discharge.groupby("cycle", as_index=False).agg(aggregations).sort_values("cycle")
    grouped = grouped[grouped["capacity"] > 0]
    if len(grouped) < 8:
        raise ValueError(f"{path.name} 有效循环点不足，无法导入。")

    baseline = _baseline_capacity(grouped)
    curve = []
    for _, row in grouped.iterrows():
        capacity = float(row["capacity"])
        point = {
            "cycle": int(row["cycle"]),
            "specific_capacity": round(capacity, 6),
            "soh": round(capacity / baseline * 100, 4),
        }
        if "voltage_V" in optional_cols and pd.notna(row.get("voltage_V")):
            point["voltage_V"] = round(float(row["voltage_V"]), 6)
        if "current_mA" in optional_cols and pd.notna(row.get("current_mA")):
            point["current_mA"] = round(float(row["current_mA"]), 6)
        if "temperature_C" in optional_cols and pd.notna(row.get("temperature_C")):
            point["temperature_C"] = round(float(row["temperature_C"]), 6)
        if "internal_resistance_mOhm" in optional_cols and pd.notna(row.get("internal_resistance_mOhm")):
            point["internal_resistance_mOhm"] = round(float(row["internal_resistance_mOhm"]), 6)
        curve.append(point)
    return curve, baseline, _extra_features(curve, optional_cols)


def _extra_features(curve: list[dict], optional_cols: list[str]):
    def values(key):
        return [float(point[key]) for point in curve if key in point]

    early_count = max(8, int(len(curve) * 0.1))
    early = curve[:early_count]
    features = {
        "has_voltage": "voltage_V" in optional_cols,
        "has_current": "current_mA" in optional_cols,
        "has_temperature": "temperature_C" in optional_cols,
        "has_internal_resistance": "internal_resistance_mOhm" in optional_cols,
    }
    voltage = values("voltage_V")
    early_voltage = [float(point["voltage_V"]) for point in early if "voltage_V" in point]
    if voltage:
        features.update(
            {
                "voltage_mean": round(sum(voltage) / len(voltage), 6),
                "voltage_min": round(min(voltage), 6),
                "voltage_max": round(max(voltage), 6),
                "early_voltage_plateau": round(sum(early_voltage) / len(early_voltage), 6) if early_voltage else None,
                "early_voltage_slope": round((early_voltage[-1] - early_voltage[0]) / max(early[-1]["cycle"] - early[0]["cycle"], 1), 8) if len(early_voltage) >= 2 else None,
            }
        )
    current = values("current_mA")
    if current:
        features.update({"current_mean_abs": round(sum(abs(value) for value in current) / len(current), 6)})
    return features


def _curve_from_cycle_frame(frame: pd.DataFrame, capacity_col: str, voltage_col: str | None = None, current_col: str | None = None, capacity_scale: float = 1.0):
    frame = frame.rename(columns={capacity_col: "capacity"})
    frame["capacity"] = pd.to_numeric(frame["capacity"], errors="coerce") * capacity_scale
    frame = frame.dropna(subset=["cycle", "capacity"])
    frame = frame[frame["capacity"] > 0].sort_values("cycle")
    if len(frame) < 8:
        raise ValueError("有效循环点不足，无法导入。")
    baseline = _baseline_capacity(frame[["cycle", "capacity"]])
    curve = []
    for _, row in frame.iterrows():
        point = {
            "cycle": int(row["cycle"]),
            "specific_capacity": round(float(row["capacity"]), 6),
            "soh": round(float(row["capacity"]) / baseline * 100, 4),
        }
        if voltage_col and voltage_col in frame.columns and pd.notna(row.get(voltage_col)):
            point["voltage_V"] = round(float(row[voltage_col]), 6)
        if current_col and current_col in frame.columns and pd.notna(row.get(current_col)):
            point["current_A"] = round(float(row[current_col]), 6)
        curve.append(point)
    optional_cols = []
    if voltage_col:
        optional_cols.append("voltage_V")
    if current_col:
        optional_cols.append("current_A")
    return curve, baseline, _extra_features(curve, optional_cols)


def _hust_curve_from_csv(path: Path):
    chunks = []
    usecols = ["cycle_idx", "voltage_V", "current_A", "Q_cycle_Ah", "is_discharge"]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=250_000):
        chunk = chunk.rename(columns={"cycle_idx": "cycle"})
        chunk["cycle"] = pd.to_numeric(chunk["cycle"], errors="coerce")
        chunk["Q_cycle_Ah"] = pd.to_numeric(chunk["Q_cycle_Ah"], errors="coerce")
        chunk["voltage_V"] = pd.to_numeric(chunk["voltage_V"], errors="coerce")
        chunk["current_A"] = pd.to_numeric(chunk["current_A"], errors="coerce")
        discharge = chunk[chunk["is_discharge"].astype(str).str.lower().isin(["true", "1"])]
        if discharge.empty:
            discharge = chunk
        grouped = discharge.dropna(subset=["cycle", "Q_cycle_Ah"]).groupby("cycle", as_index=False).agg(
            capacity=("Q_cycle_Ah", "max"),
            voltage_V=("voltage_V", "mean"),
            current_A=("current_A", "mean"),
        )
        chunks.append(grouped)
    if not chunks:
        raise ValueError(f"{path.name} 没有有效数据。")
    merged = pd.concat(chunks, ignore_index=True).groupby("cycle", as_index=False).agg(
        capacity=("capacity", "max"),
        voltage_V=("voltage_V", "mean"),
        current_A=("current_A", "mean"),
    )
    return _curve_from_cycle_frame(merged, "capacity", "voltage_V", "current_A", capacity_scale=1000)


def _tju_curve_from_csv(path: Path):
    chunks = []
    usecols = ["cycle number", "Ecell/V", "<I>/mA", "Q discharge/mA.h"]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=250_000):
        chunk = chunk.rename(columns={"cycle number": "cycle", "Ecell/V": "voltage_V", "<I>/mA": "current_mA", "Q discharge/mA.h": "capacity"})
        chunk["cycle"] = pd.to_numeric(chunk["cycle"], errors="coerce")
        chunk["capacity"] = pd.to_numeric(chunk["capacity"], errors="coerce")
        chunk["voltage_V"] = pd.to_numeric(chunk["voltage_V"], errors="coerce")
        chunk["current_mA"] = pd.to_numeric(chunk["current_mA"], errors="coerce")
        grouped = chunk.dropna(subset=["cycle", "capacity"]).groupby("cycle", as_index=False).agg(
            capacity=("capacity", "max"),
            voltage_V=("voltage_V", "mean"),
            current_A=("current_mA", lambda s: float(s.mean()) / 1000 if len(s) else 0),
        )
        chunks.append(grouped)
    if not chunks:
        raise ValueError(f"{path.name} 没有有效数据。")
    merged = pd.concat(chunks, ignore_index=True).groupby("cycle", as_index=False).agg(
        capacity=("capacity", "max"),
        voltage_V=("voltage_V", "mean"),
        current_A=("current_A", "mean"),
    )
    return _curve_from_cycle_frame(merged, "capacity", "voltage_V", "current_A")


def _insert_battery(conn, *, battery_type, chemistry, dataset_name, cell_name, source, note, curve, baseline_capacity, extra_features, allow_terminal_label=False, outlier_cells=None):
    eol_point = _find_sustained_eol(curve)
    label_status, training_eligible, flags = _quality_summary(curve, eol_point, allow_terminal_label=allow_terminal_label)
    if outlier_cells and cell_name in outlier_cells:
        label_status = "高误差离群"
        training_eligible = 0
        flags.append("留一评估残差显著偏大，保留入库但默认不参与训练")
    cycle_life = int(eol_point["cycle"] if eol_point else curve[-1]["cycle"])
    stored_curve = _downsample_curve(curve)
    conn.execute(
        """
        INSERT INTO battery_dataset (
            battery_type, theoretical_capacity, rated_capacity, c_rate,
            cycle_life, current_soh, capacity_curve, source, note, created_at,
            chemistry, dataset_name, cell_name, label_status,
            training_eligible, quality_flags, capacity_baseline, additional_features
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            battery_type,
            round(baseline_capacity, 6),
            round(baseline_capacity, 6),
            1.0,
            cycle_life,
            stored_curve[-1]["soh"],
            json.dumps(stored_curve, ensure_ascii=False),
            source,
            note,
            now_iso(),
            chemistry,
            dataset_name,
            cell_name,
            label_status,
            training_eligible,
            json.dumps(flags, ensure_ascii=False),
            round(baseline_capacity, 6),
            json.dumps(extra_features, ensure_ascii=False),
        ),
    )
    return {
        "file": note,
        "battery_type": battery_type,
        "chemistry": chemistry,
        "dataset_name": dataset_name,
        "cell_name": cell_name,
        "cycles": len(stored_curve),
        "raw_cycles": len(curve),
        "cycle_life": cycle_life,
        "label_status": label_status,
        "training_eligible": bool(training_eligible),
    }


def _train_imported_models():
    from .config import MODEL_OPTIONS
    from .services.modeling import train_model

    results = []
    errors = []
    with get_db() as conn:
        rows = conn.execute("SELECT chemistry, dataset_name, COUNT(*) AS n FROM battery_dataset GROUP BY chemistry, dataset_name ORDER BY chemistry, dataset_name").fetchall()
    by_chemistry: dict[str, list[str]] = {}
    for row in rows:
        by_chemistry.setdefault(row["chemistry"], []).append(row["dataset_name"])
    for chemistry, dataset_names in by_chemistry.items():
        for model_key in MODEL_OPTIONS:
            try:
                results.append(train_model({"chemistry": chemistry, "dataset_ids": dataset_names, "publish": True}, model_key=model_key))
            except Exception as exc:
                errors.append({"chemistry": chemistry, "model_key": model_key, "error": str(exc)})
    return {"models": results, "errors": errors}


def import_real_dataset(dataset_dir: Path = DATASET_DIR, train: bool = True):
    init_db()
    files = sorted(dataset_dir.glob("G*_Cell*_Data.xlsx"))
    if not files:
        raise FileNotFoundError(f"未找到 Excel 数据文件：{dataset_dir}")

    imported = []
    with get_db() as conn:
        conn.execute("DELETE FROM prediction_history")
        conn.execute("DELETE FROM battery_dataset")
        conn.execute("DELETE FROM model_info")

    for path in files:
        group, cell_no = _file_meta(path)
        cell_name = f"{group}_Cell{cell_no}"
        curve, baseline_capacity, extra_features = _curve_from_excel(path)
        with get_db() as conn:
            imported.append(
                _insert_battery(
                    conn,
                    battery_type=group,
                    chemistry=REAL_CHEMISTRY,
                    dataset_name=REAL_DATASET_NAME,
                    cell_name=cell_name,
                    source="真实 Excel 数据集",
                    note=f"{group} Cell {cell_no}，源文件：{path.name}",
                    curve=curve,
                    baseline_capacity=baseline_capacity,
                    extra_features=extra_features,
                    outlier_cells=HIGH_ERROR_OUTLIER_CELLS,
                )
            )

    public_specs = [
        (HUST_STANDARDIZED_DIR, "LFP", "LFP", "HUST 标准化 CSV 数据集", _hust_curve_from_csv),
        (TJU_NCA_DIR, "NCA", "NCA", "TJU Dataset 1 NCA 数据集", _tju_curve_from_csv),
        (TJU_NCM_DIR, "NCM", "NCM", "TJU Dataset 2 NCM 数据集", _tju_curve_from_csv),
    ]
    for folder, battery_type, chemistry, dataset_name, parser in public_specs:
        if not folder.exists():
            imported.append({"dataset_name": dataset_name, "error": f"路径不存在：{folder}"})
            continue
        for path in sorted(folder.glob("*.csv")):
            if chemistry == "LFP" and not re.match(r"HUST_\d+-\d+\.csv$", path.name, re.IGNORECASE):
                continue
            if path.suffix.lower() == ".pkl":
                continue
            try:
                curve, baseline_capacity, extra_features = parser(path)
                with get_db() as conn:
                    imported.append(
                        _insert_battery(
                            conn,
                            battery_type=battery_type,
                            chemistry=chemistry,
                            dataset_name=dataset_name,
                            cell_name=path.stem,
                            source="公开电池数据集 CSV",
                            note=path.name,
                            curve=curve,
                            baseline_capacity=baseline_capacity,
                            extra_features=extra_features,
                            allow_terminal_label=True,
                        )
                    )
            except Exception as exc:
                imported.append({"file": path.name, "chemistry": chemistry, "dataset_name": dataset_name, "error": str(exc)})

    training = _train_imported_models() if train else {"models": [], "errors": []}
    return {"imported_count": len([item for item in imported if "error" not in item]), "items": imported, **training}


if __name__ == "__main__":
    print(json.dumps(import_real_dataset(train=True), ensure_ascii=False, indent=2))
