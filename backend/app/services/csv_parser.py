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

    df = df[["cycle", value_col]].copy()
    df["cycle"] = pd.to_numeric(df["cycle"], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna().sort_values("cycle")
    df = df[df["cycle"] >= 0]

    if len(df) < 5:
        raise ValueError("有效曲线点少于 5 个，无法进行可靠匹配。")

    initial_capacity = float(df[value_col].iloc[0]) or rated_capacity
    curve = []
    for _, row in df.iterrows():
        capacity = float(row[value_col])
        soh = capacity / initial_capacity * 100 if initial_capacity else capacity / rated_capacity * 100
        curve.append(
            {
                "cycle": int(row["cycle"]),
                "specific_capacity": round(capacity, 4),
                "soh": round(soh, 4),
            }
        )
    return curve
