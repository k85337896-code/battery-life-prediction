import json
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from .config import BATTERY_TYPES, MODEL_OPTIONS
from .database import get_db, init_db, now_iso
from .import_real_dataset import import_real_dataset
from .seed import seed
from .services.csv_parser import parse_curve_csv
from .services.modeling import train_all_models, train_model
from .services.predictor import predict_from_curve
from .utils import row_to_dict, rows_to_dicts

app = FastAPI(title="电池寿命预测与健康评估系统")

FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    init_db()
    seed(force=False)


def require_teacher(x_role: str = Header(default="student")):
    if x_role != "teacher":
        raise HTTPException(status_code=403, detail="仅教师账号可执行该操作。")


@app.post("/api/login")
def login(payload: dict):
    users = {
        "student": {"password": "123456", "role": "student", "name": "学生演示账号"},
        "teacher": {"password": "123456", "role": "teacher", "name": "教师演示账号"},
    }
    user = users.get(payload.get("username"))
    if not user or user["password"] != payload.get("password"):
        raise HTTPException(status_code=401, detail="账号或密码错误。")
    return {"token": user["role"], "role": user["role"], "name": user["name"]}


@app.get("/api/meta")
def meta():
    return {"battery_types": BATTERY_TYPES, "model_options": MODEL_OPTIONS}


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    battery_type: str = Form(...),
    theoretical_capacity: float = Form(...),
    rated_capacity: float = Form(...),
    c_rate: float = Form(...),
    model_key: str = Form("xgboost"),
):
    if battery_type not in BATTERY_TYPES:
        raise HTTPException(status_code=400, detail="不支持的电池类型。")
    try:
        curve = parse_curve_csv(await file.read(), rated_capacity)
        return predict_from_curve(battery_type, theoretical_capacity, rated_capacity, c_rate, curve, model_key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/history")
def list_history():
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM prediction_history ORDER BY id DESC").fetchall()
    return rows_to_dicts(rows)


@app.get("/api/history/{history_id}")
def history_detail(history_id: int):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM prediction_history WHERE id = ?", (history_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="历史记录不存在。")
    return row_to_dict(row)


@app.delete("/api/history/{history_id}")
def delete_history(history_id: int):
    with get_db() as conn:
        conn.execute("DELETE FROM prediction_history WHERE id = ?", (history_id,))
    return {"ok": True}


@app.get("/api/datasets")
def list_datasets():
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM battery_dataset ORDER BY id DESC").fetchall()
    return rows_to_dicts(rows)


@app.get("/api/datasets/tree")
def dataset_tree():
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM battery_dataset ORDER BY chemistry, dataset_name, cell_name, id").fetchall()
    tree = {}
    for item in rows_to_dicts(rows):
        chemistry = item.get("chemistry") or "未标注化学成分"
        dataset_name = item.get("dataset_name") or "未命名数据集"
        tree.setdefault(chemistry, {}).setdefault(dataset_name, []).append(item)
    return tree


@app.post("/api/datasets", dependencies=[Depends(require_teacher)])
def create_dataset(payload: dict):
    required = ["battery_type", "theoretical_capacity", "rated_capacity", "c_rate", "cycle_life", "capacity_curve"]
    if any(key not in payload for key in required):
        raise HTTPException(status_code=400, detail="缺少数据库条目必要字段。")
    with get_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO battery_dataset (
                battery_type, theoretical_capacity, rated_capacity, c_rate, cycle_life,
                current_soh, capacity_curve, source, note, created_at,
                chemistry, dataset_name, cell_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["battery_type"],
                payload["theoretical_capacity"],
                payload["rated_capacity"],
                payload["c_rate"],
                payload["cycle_life"],
                payload.get("current_soh", 80),
                json.dumps(payload["capacity_curve"], ensure_ascii=False),
                payload.get("source", "教师录入"),
                payload.get("note", ""),
                now_iso(),
                payload.get("chemistry", "未标注化学成分"),
                payload.get("dataset_name", "手工录入数据集"),
                payload.get("cell_name", ""),
            ),
        )
    return {"id": cursor.lastrowid}


@app.put("/api/datasets/{dataset_id}", dependencies=[Depends(require_teacher)])
def update_dataset(dataset_id: int, payload: dict):
    allowed = ["battery_type", "theoretical_capacity", "rated_capacity", "c_rate", "cycle_life", "current_soh", "source", "note", "chemistry", "dataset_name", "cell_name"]
    sets = [f"{key} = ?" for key in allowed if key in payload]
    values = [payload[key] for key in allowed if key in payload]
    if "capacity_curve" in payload:
        sets.append("capacity_curve = ?")
        values.append(json.dumps(payload["capacity_curve"], ensure_ascii=False))
    if not sets:
        return {"ok": True}
    values.append(dataset_id)
    with get_db() as conn:
        conn.execute(f"UPDATE battery_dataset SET {', '.join(sets)} WHERE id = ?", values)
    return {"ok": True}


@app.delete("/api/datasets/{dataset_id}", dependencies=[Depends(require_teacher)])
def delete_dataset(dataset_id: int):
    with get_db() as conn:
        conn.execute("DELETE FROM battery_dataset WHERE id = ?", (dataset_id,))
    return {"ok": True}


@app.post("/api/datasets/import", dependencies=[Depends(require_teacher)])
async def import_dataset_csv(file: UploadFile = File(...), battery_type: str = Form(...), rated_capacity: float = Form(...), theoretical_capacity: float = Form(...), c_rate: float = Form(1.0)):
    curve = parse_curve_csv(await file.read(), rated_capacity)
    life = max(point["cycle"] for point in curve)
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO battery_dataset (
                battery_type, theoretical_capacity, rated_capacity, c_rate, cycle_life,
                current_soh, capacity_curve, source, note, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (battery_type, theoretical_capacity, rated_capacity, c_rate, life, curve[-1]["soh"], json.dumps(curve, ensure_ascii=False), "教师 CSV 导入", file.filename, now_iso()),
        )
    return {"ok": True}


@app.get("/api/model-info")
def model_info():
    with get_db() as conn:
        placeholders = ",".join("?" for _ in MODEL_OPTIONS)
        rows = conn.execute(f"SELECT * FROM model_info WHERE model_key IN ({placeholders}) ORDER BY model_key", tuple(MODEL_OPTIONS.keys())).fetchall()
    return rows_to_dicts(rows)


@app.get("/api/model-info/{model_key}")
def model_info_detail(model_key: str):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM model_info WHERE model_key = ?", (model_key,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="该模型尚未训练。")
    return row_to_dict(row)


@app.post("/api/model/train", dependencies=[Depends(require_teacher)])
def retrain(payload: dict | None = None):
    try:
        payload = payload or {}
        model_key = payload.pop("model_key", "xgboost")
        if model_key == "all":
            return {"models": train_all_models()}
        return train_model(payload, model_key=model_key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/seed", dependencies=[Depends(require_teacher)])
def reseed():
    return seed(force=True)


@app.post("/api/datasets/import-real", dependencies=[Depends(require_teacher)])
def import_real():
    try:
        return import_real_dataset(train=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"真实数据集导入失败：{exc}") from exc


@app.get("/api/source", response_class=PlainTextResponse)
def source_code():
    path = Path(__file__).resolve().parent / "services" / "modeling.py"
    return path.read_text(encoding="utf-8")


if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")


@app.get("/{full_path:path}", include_in_schema=False)
def serve_spa(full_path: str):
    index_file = FRONTEND_DIST / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="前端构建产物不存在，请先运行 npm run build。")
    return FileResponse(index_file)
