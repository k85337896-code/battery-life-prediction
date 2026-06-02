import json
import re
from pathlib import Path

import pandas as pd

from .database import get_db, init_db, now_iso
from .services.modeling import train_all_models


DATASET_DIR = Path(r"D:\Users\cxy\桌面\电压电流")
REAL_DATASET_NAME = "电压电流真实数据集"
REAL_CHEMISTRY = "未标注化学成分"
EOL_SOH = 80
SUSTAINED_EOL_POINTS = 3


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


def _quality_summary(curve: list[dict], eol_point: dict | None):
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
        training_eligible = 0
        flags.append("记录结束时尚未达到 80% SOH，只能视为寿命下限")

    return label_status, training_eligible, flags


def _curve_from_excel(path: Path):
    header = pd.read_excel(path, sheet_name="all_data", nrows=0, engine="openpyxl").columns
    capacity_col = "capacity_mAh" if "capacity_mAh" in header else "capacity_Ah" if "capacity_Ah" in header else None
    if not capacity_col:
        raise ValueError(f"{path.name} 未找到 capacity_mAh/capacity_Ah 列。")

    df = pd.read_excel(path, sheet_name="all_data", usecols=["cycle", "phase", capacity_col], engine="openpyxl")
    df = df.rename(columns={capacity_col: "capacity"})
    if capacity_col == "capacity_Ah":
        df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce") * 1000

    df = df.dropna(subset=["cycle", "phase", "capacity"])
    df["cycle"] = pd.to_numeric(df["cycle"], errors="coerce")
    df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce")
    df = df.dropna(subset=["cycle", "capacity"])

    discharge = df[df["phase"].astype(str).str.lower().str.contains("discharge")]
    if discharge.empty:
        discharge = df

    grouped = discharge.groupby("cycle", as_index=False)["capacity"].max().sort_values("cycle")
    grouped = grouped[grouped["capacity"] > 0]
    if len(grouped) < 8:
        raise ValueError(f"{path.name} 有效循环点不足，无法导入。")

    baseline = _baseline_capacity(grouped)
    curve = []
    for _, row in grouped.iterrows():
        capacity = float(row["capacity"])
        curve.append(
            {
                "cycle": int(row["cycle"]),
                "specific_capacity": round(capacity, 6),
                "soh": round(capacity / baseline * 100, 4),
            }
        )
    return curve, baseline


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
        curve, baseline_capacity = _curve_from_excel(path)
        eol_point = _find_sustained_eol(curve)
        label_status, training_eligible, flags = _quality_summary(curve, eol_point)
        cycle_life = int(eol_point["cycle"] if eol_point else curve[-1]["cycle"])
        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO battery_dataset (
                    battery_type, theoretical_capacity, rated_capacity, c_rate,
                    cycle_life, current_soh, capacity_curve, source, note, created_at,
                    chemistry, dataset_name, cell_name, label_status,
                    training_eligible, quality_flags, capacity_baseline
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    group,
                    round(baseline_capacity, 6),
                    round(baseline_capacity, 6),
                    1.0,
                    cycle_life,
                    curve[-1]["soh"],
                    json.dumps(curve, ensure_ascii=False),
                    "真实 Excel 数据集",
                    f"{group} Cell {cell_no}，源文件：{path.name}",
                    now_iso(),
                    REAL_CHEMISTRY,
                    REAL_DATASET_NAME,
                    f"{group}_Cell{cell_no}",
                    label_status,
                    training_eligible,
                    json.dumps(flags, ensure_ascii=False),
                    round(baseline_capacity, 6),
                ),
            )
        imported.append(
            {
                "file": path.name,
                "battery_type": group,
                "cycles": len(curve),
                "cycle_life": cycle_life,
                "label_status": label_status,
                "training_eligible": bool(training_eligible),
            }
        )

    models = train_all_models() if train else []
    return {"imported_count": len(imported), "items": imported, "models": models}


if __name__ == "__main__":
    print(json.dumps(import_real_dataset(train=True), ensure_ascii=False, indent=2))
