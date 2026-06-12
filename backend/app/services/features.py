from io import StringIO

import numpy as np
import pandas as pd


REQUIRED_TIMESERIES_COLUMNS = ("cycle", "time_s", "voltage_V", "current_A")
OPTIONAL_TIMESERIES_COLUMNS = ("capacity_Ah", "temperature_C")
MIN_PREDICT_CYCLES = 30


def parse_timeseries_csv(content: bytes, rated_capacity_ah: float, min_cycles: int = MIN_PREDICT_CYCLES):
    if len(content) > 20 * 1024 * 1024:
        raise ValueError("CSV 文件超过 20MB，请压缩或截取早期循环后再上传。")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = content.decode("gb18030", errors="ignore")
    try:
        df = pd.read_csv(StringIO(text))
    except Exception as exc:
        raise ValueError(f"CSV 解析失败：{exc}") from exc

    df = df.dropna(axis=1, how="all")
    df.columns = [str(col).strip() for col in df.columns]
    missing = [col for col in REQUIRED_TIMESERIES_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "CSV 必须使用长表时序格式，并包含列：cycle, time_s, voltage_V, current_A；"
            f"当前缺少：{', '.join(missing)}。"
        )

    used_cols = [*REQUIRED_TIMESERIES_COLUMNS, *[col for col in OPTIONAL_TIMESERIES_COLUMNS if col in df.columns]]
    df = df[used_cols].copy()
    for col in used_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=list(REQUIRED_TIMESERIES_COLUMNS)).sort_values(["cycle", "time_s"])
    if df.empty:
        raise ValueError("CSV 中没有有效的时序数据行。")
    if (df["cycle"] < 0).any():
        raise ValueError("cycle 不能为负数。")
    if not df["voltage_V"].between(0.5, 6.0).all():
        raise ValueError("voltage_V 超出合理范围，请确认单位为 V。")
    if not df["current_A"].between(-500, 500).all():
        raise ValueError("current_A 超出合理范围，请确认单位为 A。")
    if "temperature_C" in df.columns and not df["temperature_C"].dropna().between(-40, 120).all():
        raise ValueError("temperature_C 超出合理范围，请确认单位为摄氏度。")
    cycle_count = int(df["cycle"].nunique())
    if cycle_count < min_cycles:
        raise ValueError(f"有效循环数不足，至少需要 {min_cycles} 圈，当前只有 {cycle_count} 圈。")
    if rated_capacity_ah <= 0 or rated_capacity_ah > 10000:
        raise ValueError("额定容量必须大于 0，单位为 Ah。")

    curve = timeseries_to_curve(df, rated_capacity_ah)
    extra_features = extract_timeseries_features(df, curve)
    return curve, extra_features, df


def timeseries_to_curve(df: pd.DataFrame, rated_capacity_ah: float):
    curve = []
    capacity_by_cycle = {}
    for cycle, group in df.groupby("cycle"):
        group = group.sort_values("time_s")
        if "capacity_Ah" in group.columns and group["capacity_Ah"].notna().any():
            capacity = float(group["capacity_Ah"].dropna().max())
        else:
            time_s = group["time_s"].to_numpy(dtype=float)
            current_a = group["current_A"].to_numpy(dtype=float)
            if len(group) < 2:
                capacity = 0.0
            else:
                capacity = float(np.trapz(np.abs(current_a), time_s) / 3600.0)
        capacity_by_cycle[int(cycle)] = capacity

    positive = [value for value in capacity_by_cycle.values() if value > 0]
    baseline = float(np.median(positive[: min(5, len(positive))])) if positive else rated_capacity_ah
    baseline = baseline or rated_capacity_ah
    for cycle in sorted(capacity_by_cycle):
        capacity = capacity_by_cycle[cycle]
        soh = capacity / baseline * 100 if baseline else 0
        curve.append(
            {
                "cycle": int(cycle),
                "specific_capacity": round(capacity, 6),
                "soh": round(float(np.clip(soh, 0, 130)), 4),
            }
        )
    return curve


def extract_timeseries_features(df: pd.DataFrame, curve: list[dict]):
    cycle_values = sorted(df["cycle"].dropna().unique())
    first_cycle = cycle_values[0]
    later_cycle = cycle_values[min(len(cycle_values) - 1, max(1, int(len(cycle_values) * 0.2)))]
    first_qv = _q_by_voltage(df[df["cycle"] == first_cycle])
    later_qv = _q_by_voltage(df[df["cycle"] == later_cycle])
    delta_q = later_qv - first_qv
    abs_delta = np.abs(delta_q[np.isfinite(delta_q)])
    abs_delta = abs_delta[abs_delta > 1e-12]

    efficiencies = []
    cc_durations = []
    resistance_jumps = []
    for _, group in df.groupby("cycle"):
        group = group.sort_values("time_s")
        time_s = group["time_s"].to_numpy(dtype=float)
        current = group["current_A"].to_numpy(dtype=float)
        voltage = group["voltage_V"].to_numpy(dtype=float)
        if len(group) < 2:
            continue
        positive = np.trapz(np.clip(current, 0, None), time_s) / 3600.0
        negative = np.trapz(np.abs(np.clip(current, None, 0)), time_s) / 3600.0
        if positive > 1e-9:
            efficiencies.append(float(negative / positive))
        abs_current = np.abs(current)
        if abs_current.size:
            target = np.nanmedian(abs_current[abs_current > 1e-9]) if np.any(abs_current > 1e-9) else 0
            if target:
                cc_mask = np.abs(abs_current - target) <= max(target * 0.05, 1e-4)
                cc_durations.append(float(np.sum(np.diff(time_s, prepend=time_s[0]) * cc_mask)))
        dv = np.diff(voltage)
        di = np.diff(current)
        valid = np.abs(di) > 1e-6
        if np.any(valid):
            resistance_jumps.append(float(np.nanmedian(np.abs(dv[valid] / di[valid]))))

    capacities = np.array([point["specific_capacity"] for point in curve], dtype=float)
    cycles = np.array([point["cycle"] for point in curve], dtype=float)
    early_n = min(len(curve), max(5, int(len(curve) * 0.3)))
    slope = 0.0
    if early_n >= 2:
        slope = float(np.polyfit(cycles[:early_n], capacities[:early_n], 1)[0])
    cc_change = 0.0
    if len(cc_durations) >= 2:
        cc_change = float(cc_durations[min(len(cc_durations) - 1, early_n - 1)] - cc_durations[0])

    return {
        "dq_log_var": float(np.log(np.var(abs_delta) + 1e-12)) if abs_delta.size else 0.0,
        "dq_log_min": float(np.log(np.min(abs_delta) + 1e-12)) if abs_delta.size else 0.0,
        "early_capacity_slope": slope,
        "coulombic_efficiency_mean": float(np.mean(efficiencies)) if efficiencies else 0.0,
        "cc_duration_change": cc_change,
        "internal_resistance_proxy": float(np.mean(resistance_jumps)) if resistance_jumps else 0.0,
        "voltage_mean": float(df["voltage_V"].mean()),
        "voltage_min": float(df["voltage_V"].min()),
        "voltage_max": float(df["voltage_V"].max()),
        "current_mean_abs": float(df["current_A"].abs().mean()),
        "temperature_mean": float(df["temperature_C"].mean()) if "temperature_C" in df.columns else 0.0,
        "has_temperature": 1 if "temperature_C" in df.columns else 0,
    }


def _q_by_voltage(cycle_df: pd.DataFrame, bins=120):
    if len(cycle_df) < 2:
        return np.zeros(bins)
    group = cycle_df.sort_values("time_s")
    time_s = group["time_s"].to_numpy(dtype=float)
    current = np.abs(group["current_A"].to_numpy(dtype=float))
    voltage = group["voltage_V"].to_numpy(dtype=float)
    q_step = np.diff(time_s, prepend=time_s[0]) * current / 3600.0
    grid = np.linspace(2.0, 4.5, bins)
    order = np.argsort(voltage)
    voltage_sorted = voltage[order]
    q_cum = np.cumsum(q_step[order])
    if np.max(voltage_sorted) == np.min(voltage_sorted):
        return np.zeros(bins)
    return np.interp(grid, voltage_sorted, q_cum, left=q_cum[0], right=q_cum[-1])


def extract_features(battery_type, theoretical_capacity, rated_capacity, c_rate, curve, extra_features=None):
    extra_features = extra_features or {}
    life_hint = max(point["cycle"] for point in curve)
    first = curve[0]["specific_capacity"]
    last = curve[-1]["specific_capacity"]
    retention = last / first if first else 0
    slope = (last - first) / max(life_hint, 1)
    voltage_values = [float(point["voltage_V"]) for point in curve if "voltage_V" in point]
    current_values = [float(point["current_A"]) for point in curve if "current_A" in point]
    temp_values = [float(point["temperature_C"]) for point in curve if "temperature_C" in point]
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
        "voltage_mean": float(np.mean(voltage_values)) if voltage_values else float(extra_features.get("voltage_mean", 0)),
        "voltage_min": float(np.min(voltage_values)) if voltage_values else float(extra_features.get("voltage_min", 0)),
        "voltage_max": float(np.max(voltage_values)) if voltage_values else float(extra_features.get("voltage_max", 0)),
        "current_mean_abs": float(np.mean(np.abs(current_values))) if current_values else float(extra_features.get("current_mean_abs", 0)),
        "temperature_mean": float(np.mean(temp_values)) if temp_values else float(extra_features.get("temperature_mean", 0)),
        "has_voltage": 1 if voltage_values or extra_features.get("voltage_mean") else 0,
        "has_current": 1 if current_values or extra_features.get("current_mean_abs") else 0,
        "has_temperature": 1 if temp_values or extra_features.get("has_temperature") else 0,
        "dq_log_var": float(extra_features.get("dq_log_var", 0)),
        "dq_log_min": float(extra_features.get("dq_log_min", 0)),
        "early_capacity_slope": float(extra_features.get("early_capacity_slope", slope)),
        "coulombic_efficiency_mean": float(extra_features.get("coulombic_efficiency_mean", 0)),
        "cc_duration_change": float(extra_features.get("cc_duration_change", 0)),
        "internal_resistance_proxy": float(extra_features.get("internal_resistance_proxy", 0)),
    }
    sampled_soh = _normalized_soh(curve, grid_size=16)
    for index, value in enumerate(sampled_soh):
        base_features[f"seq_soh_{index:02d}"] = float(value)
    return base_features


def _normalized_soh(curve, grid_size=120):
    cycles = np.array([p["cycle"] for p in curve], dtype=float)
    soh = np.array([p["soh"] for p in curve], dtype=float)
    if cycles.max() == cycles.min():
        x = np.zeros_like(cycles)
    else:
        x = (cycles - cycles.min()) / (cycles.max() - cycles.min())
    grid = np.linspace(0, 1, grid_size)
    return np.interp(grid, x, soh)
