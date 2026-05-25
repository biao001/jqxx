from __future__ import annotations

import base64
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .analysis import AnalysisService, decode_image_bytes, encode_upload_name
from .config import get_settings
from . import storage
from . import dataset as dataset_mod
from .training import TrainingManager
from .llm import generate_review, generate_chat


settings = get_settings()
settings.runtime_dir.mkdir(parents=True, exist_ok=True)
service = AnalysisService(settings)

SESSIONS_DB = settings.runtime_dir / "sessions.db"
storage.init_db(SESSIONS_DB)

DATASET_DIR = settings.repo_root / "behavior_algo" / "data" / "reckless_mapped"
UPLOAD_DATASET_DIR = settings.repo_root / "behavior_algo" / "data" / "uploads"
TRAIN_CFG_PATH = settings.runtime_dir / "training_config.json"
TRAIN_CFG_DEFAULT = {"base": "yolov8s.pt", "epochs": 150, "imgsz": 768, "batch": 12, "patience": 30}
trainer = TrainingManager(settings.repo_root, settings.runtime_dir / "train.log")


def _read_train_cfg() -> dict:
    import json
    if TRAIN_CFG_PATH.exists():
        try:
            return {**TRAIN_CFG_DEFAULT, **json.loads(TRAIN_CFG_PATH.read_text())}
        except Exception:
            pass
    return dict(TRAIN_CFG_DEFAULT)

app = FastAPI(title="DMS Local Backend", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "service": "dms-backend",
        "llm_configured": bool(settings.llm.api_key),
        "capabilities": service.capabilities(),
    }


@app.post("/api/videos/analyze")
async def analyze_video(file: UploadFile = File(...)) -> dict:
    filename = encode_upload_name(file.filename)
    upload_dir = settings.runtime_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_path = upload_dir / filename
    try:
        content = await file.read()
        if not content:
            raise ValueError("上传文件为空")
        upload_path.write_bytes(content)
        return service.analyze_video(upload_path, filename)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/reports/{job_id}")
def download_report(job_id: str) -> FileResponse:
    report_path = settings.runtime_dir / "reports" / f"{Path(job_id).name}.md"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="报告不存在")
    return FileResponse(
        report_path,
        media_type="text/markdown; charset=utf-8",
        filename=f"dms-report-{job_id}.md",
    )


@app.post("/api/reports")
def create_report(result: dict) -> dict:
    try:
        return service.write_result_report(result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/sessions")
def save_session(payload: dict) -> dict:
    """保存一次行程会话(汇总 + 事件 + 评分曲线)。"""
    try:
        sid = storage.create_session(SESSIONS_DB, payload)
        return {"id": sid}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/sessions")
def list_sessions(limit: int = 50) -> dict:
    return {"sessions": storage.list_sessions(SESSIONS_DB, limit=limit)}


@app.get("/api/stats")
def stats() -> dict:
    return storage.get_stats(SESSIONS_DB)


@app.get("/api/dataset/info")
def dataset_info() -> dict:
    info = dataset_mod.scan_dataset(DATASET_DIR)
    uploads = []
    if UPLOAD_DATASET_DIR.exists():
        uploads = sorted(p.name for p in UPLOAD_DATASET_DIR.iterdir() if p.is_dir())
    return {"active": info, "uploads": uploads}


@app.post("/api/dataset/upload")
async def dataset_upload(file: UploadFile = File(...)) -> dict:
    name = Path(encode_upload_name(file.filename)).stem[:40] or "dataset"
    try:
        content = await file.read()
        info = dataset_mod.save_uploaded_dataset(content, UPLOAD_DATASET_DIR / name)
        info["scan"] = dataset_mod.scan_dataset(UPLOAD_DATASET_DIR / name)
        return info
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/dataset/images")
def dataset_images(split: str = "train", cls: int | None = None, issue: str | None = None, page: int = 0, page_size: int = 24) -> dict:
    if split not in dataset_mod.VALID_SPLITS:
        raise HTTPException(status_code=400, detail="split 非法")
    return dataset_mod.list_images(DATASET_DIR, split, cls, issue, max(0, page), min(60, max(1, page_size)))


@app.get("/api/dataset/image_file")
def dataset_image_file(split: str, name: str) -> FileResponse:
    if not dataset_mod.safe_split_name(split, name):
        raise HTTPException(status_code=400, detail="参数非法")
    p = dataset_mod.image_path(DATASET_DIR, split, name)
    if p is None:
        raise HTTPException(status_code=404, detail="图片不存在")
    return FileResponse(p)


@app.post("/api/dataset/image_upload")
async def dataset_image_upload(split: str = Form("train"), file: UploadFile = File(...)) -> dict:
    if split not in dataset_mod.VALID_SPLITS:
        raise HTTPException(status_code=400, detail="split 非法")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="文件为空")
    name = dataset_mod.add_single_image(DATASET_DIR, split, file.filename or "img.jpg", content)
    return {"name": name, "split": split}


@app.post("/api/dataset/labels")
def dataset_save_labels(payload: dict) -> dict:
    split, name = str(payload.get("split", "")), str(payload.get("name", ""))
    if not dataset_mod.safe_split_name(split, name):
        raise HTTPException(status_code=400, detail="参数非法")
    return {"saved": dataset_mod.write_labels(DATASET_DIR, split, name, payload.get("boxes", []) or [])}


@app.delete("/api/dataset/image")
def dataset_delete_image(split: str, name: str) -> dict:
    if not dataset_mod.safe_split_name(split, name):
        raise HTTPException(status_code=400, detail="参数非法")
    return {"deleted": dataset_mod.delete_image(DATASET_DIR, split, name)}


@app.get("/api/detector/params")
def get_detector_params() -> dict:
    return service.behavior.get_params()


@app.post("/api/detector/params")
def set_detector_params(payload: dict) -> dict:
    return service.behavior.set_params(payload)


@app.get("/api/training/config")
def get_training_config() -> dict:
    return _read_train_cfg()


@app.post("/api/training/config")
def set_training_config(payload: dict) -> dict:
    import json
    cfg = _read_train_cfg()
    for k in TRAIN_CFG_DEFAULT:
        if k in payload:
            cfg[k] = payload[k]
    TRAIN_CFG_PATH.write_text(json.dumps(cfg, ensure_ascii=False))
    return cfg


@app.post("/api/training/start")
def training_start() -> dict:
    try:
        trainer.start(_read_train_cfg(), DATASET_DIR / "data.yaml")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return trainer.status()


@app.get("/api/training/status")
def training_status() -> dict:
    st = trainer.status()
    # 训练成功完成 → 热部署新模型(train_unified.py 已把 best.pt 拷到 unified.pt)
    if st["finished"] and st["success"] and not trainer.deployed:
        service.behavior.reload()
        trainer.deployed = True
        st["deployed"] = True
    else:
        st["deployed"] = trainer.deployed and st["finished"]
    return st


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> dict:
    data = storage.get_session(SESSIONS_DB, session_id)
    if data is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return data


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str) -> dict:
    ok = storage.delete_session(SESSIONS_DB, session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"deleted": True}


@app.post("/api/llm/review")
def llm_review(payload: dict) -> dict:
    """对一次行程的汇总数据生成 AI 驾驶点评/教练建议。"""
    summary = payload.get("summary", payload)
    return {"review": generate_review(settings.llm, summary)}


@app.post("/api/llm/chat")
def llm_chat(payload: dict) -> dict:
    """就当前/历史驾驶数据进行 AI 问答。"""
    question = str(payload.get("question", "")).strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")
    answer = generate_chat(
        settings.llm,
        question,
        payload.get("context", {}) or {},
        payload.get("history", []) or [],
    )
    return {"answer": answer}


@app.post("/api/driver/register")
def driver_register(payload: dict) -> dict:
    """登记车主：传入姓名 + 若干帧人脸(dataURL)，采集样本并重训。"""
    name = str(payload.get("name", "")).strip()
    if not name:
        raise HTTPException(status_code=400, detail="姓名不能为空")
    frames = []
    for d in payload.get("images", []) or []:
        try:
            frames.append(decode_image_bytes(base64.b64decode(str(d).split(",", 1)[-1])))
        except Exception:
            continue
    try:
        saved = service.driver.register(name, frames)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if saved == 0:
        raise HTTPException(status_code=400, detail="未从画面中检测到清晰人脸，请正对摄像头重试")
    return {"name": name, "saved": saved, "drivers": service.driver.list_drivers()}


@app.get("/api/driver/list")
def driver_list() -> dict:
    return {"drivers": service.driver.list_drivers(), "available": service.driver.available}


@app.delete("/api/driver/{name}")
def driver_delete(name: str) -> dict:
    if not service.driver.delete(name):
        raise HTTPException(status_code=404, detail="车主不存在")
    return {"deleted": True, "drivers": service.driver.list_drivers()}


@app.websocket("/ws/camera")
async def camera_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    frame_id = 0
    try:
        while True:
            payload = await websocket.receive_bytes()
            try:
                frame = decode_image_bytes(payload)
                result = service.analyze_frame(frame, frame_id=frame_id)
                await websocket.send_json(result)
                frame_id += 1
            except Exception as exc:
                await websocket.send_json({"error": str(exc), "frame_id": frame_id})
    except WebSocketDisconnect:
        return
