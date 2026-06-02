from io import StringIO

import pandas as pd


HEADER_MAP = {
    "循环": "cycle",
    "循环次数": "cycle",
    "cycle": "cycle",
    "cycles": "cycle",
    "容量": "capacity",
    "比容量": "specific_capacity",
    "capacity": "capacity",
    "specific_capacity": "specific_capacity",
    "电压": "voltage_V",
    "电压_v": "voltage_V",
    "voltage": "voltage_V",
    "voltage_v": "voltage_V",
    "电流": "current_mA",
    "电流_ma": "current_mA",
    "current": "current_mA",
    "current_ma": "current_mA",
}


def parse_curve_csv(content: bytes, rated_capacity: float):
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = content.decode("gb18030", errors="ignore")

    df = pd.read_csv(StringIO(text))
    df = df.dropna(axis=1, how="all")
    df.columns = [HEADER_MAP.get(str(col).strip().lower(), str(col).strip()) for col in df.columns]

    if "cycle" not in df.columns:
        raise ValueError("CSV 必须包含 cycle/循环次数 列。")

    value_col = "specific_capacity" if "specific_capacity" in df.columns else "capacity" if "capacity" in df.columns else None
    if not value_col:
        raise ValueError("CSV 必须包含 capacity/容量 或 specific_capacity/比容量 列。")

    optional_cols = [col for col in ("voltage_V", "current_mA") if col in df.columns]
    df = df[["cycle", value_col, *optional_cols]].copy()
    df["cycle"] = pd.to_numeric(df["cycle"], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    for col in optional_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().sort_values("cycle")
    df = df[df["cycle"] >= 0]

    if len(df) < 5:
        raise ValueError("有效曲线点少于 5 个，无法进行可靠匹配。")

    initial_capacity = float(df[value_col].iloc[0]) or rated_capacity
    curve = []
    for _, row in df.iterrows():
        capacity = float(row[value_col])
        soh = capacity / initial_capacity * 100 if initial_capacity else capacity / rated_capacity * 100
        point = {
            "cycle": int(row["cycle"]),
            "specific_capacity": round(capacity, 4),
            "soh": round(soh, 4),
        }
        if "voltage_V" in optional_cols and pd.notna(row.get("voltage_V")):
            point["voltage_V"] = round(float(row["voltage_V"]), 6)
        if "current_mA" in optional_cols and pd.notna(row.get("current_mA")):
            point["current_mA"] = round(float(row["current_mA"]), 6)
        curve.append(point)
    return curve
