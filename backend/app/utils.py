import json
from sqlite3 import Row
from typing import Any


def row_to_dict(row: Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    data = dict(row)
    for key in ("capacity_curve", "input_curve", "predicted_curve", "metrics", "feature_list", "hyperparameters", "input_summary"):
        if key in data and isinstance(data[key], str):
            try:
                data[key] = json.loads(data[key])
            except json.JSONDecodeError:
                pass
    return data


def rows_to_dicts(rows):
    return [row_to_dict(row) for row in rows]
