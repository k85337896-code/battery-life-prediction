import json
from pathlib import Path
from uuid import uuid4

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from .config import BATTERY_TYPES, MODEL_OPTIONS
from .database import get_db, init_db, now_iso
from .auth import create_token, current_user, require_teacher, verify_password
from .import_real_dataset import import_real_dataset
from .seed import seed
from .services.features import parse_timeseries_csv
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


def update_job(job_id: str, status: str, progress: int, message: str = "", result=None, error: str = ""):
    with get_db() as conn:
        conn.execute(
            """
            UPDATE training_jobs
            SET status = ?, progress = ?, message = ?, result = ?, error = ?, updated_at = ?
            WHERE id = ?
            """,
            (status, progress, message, json.dumps(result or {}, ensure_ascii=False), error, now_iso(), job_id),
        )


def run_training_job(job_id: str, payload: dict):
    try:
        update_job(job_id, "running", 10, "正在准备训练数据")
        model_key = payload.pop("model_key", "xgboost")
        if model_key == "all":
            results = []
            for index, key in enumerate(MODEL_OPTIONS):
                update_job(job_id, "running", 15 + int(index / max(len(MODEL_OPTIONS), 1) * 75), f"正在训练 {MODEL_OPTIONS[key]}")
                results.append(train_model(dict(payload), model_key=key))
            update_job(job_id, "succeeded", 100, "全部模型训练完成", {"models": results})
        else:
            result = train_model(payload, model_key=model_key)
            update_job(job_id, "succeeded", 100, "模型训练完成", result)
    except Exception as exc:
        update_job(job_id, "failed", 100, "训练失败", error=str(exc))


@app.post("/api/login")
def login(payload: dict):
    with get_db() as conn:
        user = conn.execute("SELECT * FROM users WHERE username = ?", (payload.get("username"),)).fetchone()
    if not user or not verify_password(payload.get("password", ""), user["password_hash"]):
        raise HTTPException(status_code=401, detail="账号或密码错误。")
    return {"token": create_token(user["username"], user["role"]), "role": user["role"], "name": user["display_name"], "username": user["username"]}


@app.get("/api/meta")
def meta(user: dict = Depends(current_user)):
    with get_db() as conn:
        rows = conn.execute("SELECT model_key, model_type, chemistry, version FROM model_info WHERE status = 'published' ORDER BY chemistry, model_type, version DESC").fetchall()
        chem_rows = conn.execute("SELECT DISTINCT chemistry FROM battery_dataset ORDER BY chemistry").fetchall()
    published = [{"value": row["model_key"], "label": f"{row['model_type']} v{row['version']}（{row['chemistry']}）", "chemistry": row["chemistry"]} for row in rows]
    return {"battery_types": BATTERY_TYPES, "model_options": MODEL_OPTIONS, "published_models": published, "chemistries": [row["chemistry"] for row in chem_rows]}


@app.get("/api/models/published")
def published_models(user: dict = Depends(current_user), chemistry: str | None = None):
    with get_db() as conn:
        if chemistry:
            rows = conn.execute("SELECT * FROM model_info WHERE status = 'published' AND chemistry = ? ORDER BY version DESC", (chemistry,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM model_info WHERE status = 'published' ORDER BY chemistry, version DESC").fetchall()
    data = rows_to_dicts(rows)
    if user["role"] != "teacher":
        for item in data:
            item["hyperparameters"] = {}
            item["feature_list"] = []
            item["source_path"] = ""
    return data


@app.post("/api/predict")
async def predict(
    user: dict = Depends(current_user),
    file: UploadFile = File(...),
    chemistry: str = Form(""),
    battery_type: str = Form(""),
    theoretical_capacity: float = Form(0),
    rated_capacity: float = Form(...),
    c_rate: float = Form(1.0),
    model_key: str = Form(...),
):
    try:
        chemistry = chemistry.strip()
        battery_type = battery_type.strip()
        curve, extra_features, _ = parse_timeseries_csv(await file.read(), rated_capacity)
        inferred_type = battery_type or chemistry or "G1"
        if theoretical_capacity <= 0:
            theoretical_capacity = rated_capacity
        return predict_from_curve(
            inferred_type,
            theoretical_capacity,
            rated_capacity,
            c_rate,
            curve,
            model_key,
            username=user["username"],
            extra_features=extra_features,
            chemistry=chemistry or None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/history")
def list_history(user: dict = Depends(current_user), scope: str = "mine"):
    with get_db() as conn:
        if user["role"] == "teacher" and scope == "all":
            rows = conn.execute("SELECT * FROM prediction_history ORDER BY id DESC").fetchall()
        else:
            rows = conn.execute("SELECT * FROM prediction_history WHERE username = ? ORDER BY id DESC", (user["username"],)).fetchall()
    return rows_to_dicts(rows)


@app.get("/api/history/{history_id}")
def history_detail(history_id: int, user: dict = Depends(current_user)):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM prediction_history WHERE id = ?", (history_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="历史记录不存在。")
    if user["role"] != "teacher" and row["username"] != user["username"]:
        raise HTTPException(status_code=403, detail="只能查看自己的预测记录。")
    return row_to_dict(row)


@app.delete("/api/history/{history_id}")
def delete_history(history_id: int, user: dict = Depends(current_user)):
    with get_db() as conn:
        if user["role"] == "teacher":
            conn.execute("DELETE FROM prediction_history WHERE id = ?", (history_id,))
        else:
            conn.execute("DELETE FROM prediction_history WHERE id = ? AND username = ?", (history_id, user["username"]))
    return {"ok": True}


@app.get("/api/datasets")
def list_datasets(user: dict = Depends(current_user)):
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


def infer_dataset_quality(curve: list[dict]):
    below_indexes = [index for index, point in enumerate(curve) if point.get("soh", 100) <= 80]
    eol_index = None
    for index in below_indexes:
        if index + 2 < len(curve) and all(point.get("soh", 100) <= 80 for point in curve[index : index + 3]):
            eol_index = index
            break
    flags = []
    if eol_index is None:
        status = "未达到EOL" if not below_indexes else "低于80%但未持续"
        flags.append("未形成连续 80% SOH 寿命标签，默认不参与训练")
        return status, 0, flags
    if curve[-1].get("soh", 100) > 80:
        flags.append("低于 80% 后又回升，寿命标签不稳定")
        return "低于80%后回升", 0, flags
    return "可靠EOL", 1, flags


@app.post("/api/datasets", dependencies=[Depends(require_teacher)])
def create_dataset(payload: dict):
    required = ["battery_type", "theoretical_capacity", "rated_capacity", "c_rate", "cycle_life", "capacity_curve"]
    if any(key not in payload for key in required):
        raise HTTPException(status_code=400, detail="缺少数据库条目必要字段。")
    if payload.get("chemistry") in (None, "", "未标注化学成分"):
        raise HTTPException(status_code=400, detail="必须标注化学体系后才能写入数据集。")
    with get_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO battery_dataset (
                battery_type, theoretical_capacity, rated_capacity, c_rate, cycle_life,
                current_soh, capacity_curve, source, note, created_at,
                chemistry, dataset_name, cell_name, label_status,
                training_eligible, quality_flags, capacity_baseline, additional_features
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                payload.get("label_status", infer_dataset_quality(payload["capacity_curve"])[0]),
                payload.get("training_eligible", infer_dataset_quality(payload["capacity_curve"])[1]),
                json.dumps(payload.get("quality_flags", infer_dataset_quality(payload["capacity_curve"])[2]), ensure_ascii=False),
                payload.get("capacity_baseline", payload["rated_capacity"]),
                json.dumps(payload.get("additional_features", {}), ensure_ascii=False),
            ),
        )
    return {"id": cursor.lastrowid}


@app.put("/api/datasets/{dataset_id}", dependencies=[Depends(require_teacher)])
def update_dataset(dataset_id: int, payload: dict):
    if payload.get("chemistry") in ("", "未标注化学成分"):
        raise HTTPException(status_code=400, detail="化学体系不能留空或未标注。")
    allowed = [
        "battery_type", "theoretical_capacity", "rated_capacity", "c_rate",
        "cycle_life", "current_soh", "source", "note", "chemistry",
        "dataset_name", "cell_name", "label_status", "training_eligible",
        "capacity_baseline",
    ]
    sets = [f"{key} = ?" for key in allowed if key in payload]
    values = [payload[key] for key in allowed if key in payload]
    if "capacity_curve" in payload:
        sets.append("capacity_curve = ?")
        values.append(json.dumps(payload["capacity_curve"], ensure_ascii=False))
    if "additional_features" in payload:
        sets.append("additional_features = ?")
        values.append(json.dumps(payload["additional_features"], ensure_ascii=False))
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
async def import_dataset_csv(file: UploadFile = File(...), battery_type: str = Form(""), chemistry: str = Form(...), dataset_name: str = Form(...), rated_capacity: float = Form(...), theoretical_capacity: float = Form(0), c_rate: float = Form(1.0)):
    chemistry = chemistry.strip()
    dataset_name = dataset_name.strip()
    battery_type = battery_type.strip()
    if chemistry in ("", "未标注化学成分"):
        raise HTTPException(status_code=400, detail="导入数据集必须选择或新建化学体系。")
    curve, extra_features, _ = parse_timeseries_csv(await file.read(), rated_capacity, min_cycles=1)
    life = max(point["cycle"] for point in curve)
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO battery_dataset (
                battery_type, theoretical_capacity, rated_capacity, c_rate, cycle_life,
                current_soh, capacity_curve, source, note, created_at,
                chemistry, dataset_name, cell_name,
                label_status, training_eligible, quality_flags, capacity_baseline, additional_features
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                battery_type,
                theoretical_capacity or rated_capacity,
                rated_capacity,
                c_rate,
                life,
                curve[-1]["soh"],
                json.dumps(curve, ensure_ascii=False),
                "教师 CSV 导入",
                file.filename,
                now_iso(),
                chemistry,
                dataset_name,
                file.filename,
                infer_dataset_quality(curve)[0],
                infer_dataset_quality(curve)[1],
                json.dumps(infer_dataset_quality(curve)[2], ensure_ascii=False),
                rated_capacity,
                json.dumps(extra_features, ensure_ascii=False),
            ),
        )
    return {"ok": True}


@app.get("/api/model-info")
def model_info(user: dict = Depends(current_user)):
    with get_db() as conn:
        if user["role"] == "teacher":
            rows = conn.execute("SELECT * FROM model_info ORDER BY base_model_key, version DESC, trained_at DESC").fetchall()
        else:
            rows = conn.execute("SELECT * FROM model_info WHERE status = 'published' ORDER BY base_model_key, version DESC").fetchall()
    data = rows_to_dicts(rows)
    if user["role"] != "teacher":
        for item in data:
            item["hyperparameters"] = {}
            item["feature_list"] = []
            item["source_path"] = ""
    return data


@app.get("/api/model-info/{model_key}")
def model_info_detail(model_key: str, user: dict = Depends(current_user)):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM model_info WHERE model_key = ?", (model_key,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="该模型尚未训练。")
    if user["role"] != "teacher" and row["status"] != "published":
        raise HTTPException(status_code=403, detail="学生只能查看已发布模型摘要。")
    return row_to_dict(row)


@app.post("/api/model/train", dependencies=[Depends(require_teacher)])
def retrain(background_tasks: BackgroundTasks, payload: dict | None = None):
    payload = payload or {}
    if not payload.get("chemistry"):
        raise HTTPException(status_code=400, detail="训练前必须选择化学体系。")
    if not payload.get("dataset_ids"):
        raise HTTPException(status_code=400, detail="训练前必须选择一个或多个数据集。")
    job_id = uuid4().hex
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO training_jobs (id, status, progress, message, result, error, created_at, updated_at)
            VALUES (?, 'queued', 0, '已加入训练队列', '{}', '', ?, ?)
            """,
            (job_id, now_iso(), now_iso()),
        )
    background_tasks.add_task(run_training_job, job_id, dict(payload))
    return {"job_id": job_id, "status": "queued", "progress": 0}


@app.get("/api/model/train/{job_id}", dependencies=[Depends(require_teacher)])
def training_job(job_id: str):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM training_jobs WHERE id = ?", (job_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="训练任务不存在。")
    return row_to_dict(row)


@app.post("/api/model-info/{model_key}/publish", dependencies=[Depends(require_teacher)])
def publish_model(model_key: str, payload: dict | None = None):
    status = (payload or {}).get("status", "published")
    if status not in {"published", "draft", "archived"}:
        raise HTTPException(status_code=400, detail="模型状态不合法。")
    with get_db() as conn:
        conn.execute(
            "UPDATE model_info SET status = ?, visibility = ? WHERE model_key = ?",
            (status, "student" if status == "published" else "teacher", model_key),
        )
    return {"ok": True, "status": status}


@app.post("/api/seed", dependencies=[Depends(require_teacher)])
def reseed():
    return seed(force=True)


@app.post("/api/datasets/import-real", dependencies=[Depends(require_teacher)])
def import_real():
    try:
        return import_real_dataset(train=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"真实数据集导入失败：{exc}") from exc


@app.get("/api/source", response_class=PlainTextResponse, dependencies=[Depends(require_teacher)])
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
