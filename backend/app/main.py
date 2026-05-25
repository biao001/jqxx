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

UPLOAD_DATASET_DIR = settings.repo_root / "datasets" / "uploads"
TRAIN_CFG_PATH = settings.runtime_dir / "training_config.json"
TRAIN_CFG_DEFAULT = {"base": "yolov8s.pt", "epochs": 150, "imgsz": 768, "batch": 12, "patience": 30}
trainer = TrainingManager(settings.repo_root, settings.runtime_dir / "train.log")

# 多数据集/多模型项目注册表(共用 BehaviorRuntime 内已构造的实例)
registry = service.behavior.registry
DEFAULT_DATASET = "reckless_mapped"


def _dataset_dir(name: str | None) -> Path:
    """按项目名解析数据集目录;未知项目返回 400。"""
    name = name or DEFAULT_DATASET
    proj = registry.get(name)
    if proj is None:
        raise HTTPException(status_code=400, detail=f"未知数据集项目: {name}")
    registry.ensure_dataset_dirs(name)  # 新建项目可能尚无目录
    return registry.dataset_dir(name)


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
    allow_origin_regex=settings.allowed_origin_regex,
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


@app.get("/api/projects")
def list_projects() -> dict:
    """列出所有数据集/模型项目 + 当前生效模型。"""
    return {"projects": registry.list_with_stats(), "active_model": registry.active_model()}


@app.post("/api/projects")
def create_project(payload: dict) -> dict:
    """新建一个空数据集项目(7 类 schema)，供自行上传图片+标注。"""
    try:
        entry = registry.create_project(str(payload.get("name", "")), str(payload.get("label", "")))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"created": entry, "projects": registry.list_with_stats()}


@app.delete("/api/projects/{name}")
def delete_project(name: str) -> dict:
    """删除一个非内置数据集项目(连同其数据集目录与模型文件)。"""
    try:
        res = registry.delete_project(name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if res.get("reset_active"):
        service.behavior.reload()  # 生效模型被删 → 重载回退
    return {**res, "projects": registry.list_with_stats(), "active_model": registry.active_model()}


@app.post("/api/projects/{name}/rename")
def rename_project(name: str, payload: dict) -> dict:
    """重命名项目的显示名(label)和/或标识名(name，仅非内置；会迁移目录+模型)。"""
    try:
        res = registry.rename_project(name, payload.get("new_label"), payload.get("new_name"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if res.get("active_changed"):
        service.behavior.reload()
    return {"project": res["project"], "projects": registry.list_with_stats(), "active_model": registry.active_model()}


@app.get("/api/model/active")
def get_active_model() -> dict:
    return {"active_model": registry.active_model()}


@app.post("/api/model/active")
def set_active_model(payload: dict) -> dict:
    """切换实时检测生效的模型并热重载。"""
    try:
        name = service.behavior.set_active_model(str(payload.get("model", "")))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"active_model": name, "projects": registry.list_with_stats()}


@app.get("/api/dataset/info")
def dataset_info(dataset: str | None = None) -> dict:
    info = dataset_mod.scan_dataset(_dataset_dir(dataset))
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
def dataset_images(split: str = "train", cls: int | None = None, issue: str | None = None, page: int = 0, page_size: int = 24, dataset: str | None = None) -> dict:
    if split != "all" and split not in dataset_mod.BROWSE_SPLITS:
        raise HTTPException(status_code=400, detail="split 非法")
    return dataset_mod.list_images(_dataset_dir(dataset), split, cls, issue, max(0, page), min(60, max(1, page_size)))


@app.get("/api/dataset/image_file")
def dataset_image_file(split: str, name: str, dataset: str | None = None) -> FileResponse:
    if not dataset_mod.safe_split_name(split, name):
        raise HTTPException(status_code=400, detail="参数非法")
    p = dataset_mod.image_path(_dataset_dir(dataset), split, name)
    if p is None:
        raise HTTPException(status_code=404, detail="图片不存在")
    return FileResponse(p)


@app.post("/api/dataset/image_upload")
async def dataset_image_upload(split: str = Form("unassigned"), dataset: str = Form(DEFAULT_DATASET), file: UploadFile = File(...)) -> dict:
    if split not in dataset_mod.BROWSE_SPLITS:
        raise HTTPException(status_code=400, detail="split 非法")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="文件为空")
    name = dataset_mod.add_single_image(_dataset_dir(dataset), split, file.filename or "img.jpg", content)
    return {"name": name, "split": split}


@app.post("/api/dataset/labels")
def dataset_save_labels(payload: dict) -> dict:
    split, name = str(payload.get("split", "")), str(payload.get("name", ""))
    if not dataset_mod.safe_split_name(split, name):
        raise HTTPException(status_code=400, detail="参数非法")
    return {"saved": dataset_mod.write_labels(_dataset_dir(payload.get("dataset")), split, name, payload.get("boxes", []) or [])}


@app.delete("/api/dataset/image")
def dataset_delete_image(split: str, name: str, dataset: str | None = None) -> dict:
    if not dataset_mod.safe_split_name(split, name):
        raise HTTPException(status_code=400, detail="参数非法")
    return {"deleted": dataset_mod.delete_image(_dataset_dir(dataset), split, name)}


@app.post("/api/dataset/delete_batch")
def dataset_delete_batch(payload: dict) -> dict:
    """批量删除某个 split 下的多张图片(连标注)。"""
    split = str(payload.get("split", ""))
    names = payload.get("names", []) or []
    ds_dir = _dataset_dir(payload.get("dataset"))
    splits = list(dataset_mod.BROWSE_SPLITS) if split == "all" else [split]
    deleted = 0
    for nm in names:
        nm = str(nm)
        if not nm or "/" in nm or "\\" in nm or ".." in nm:
            continue
        for sp in splits:  # split='all' 时在各 split 中查找删除
            if dataset_mod.delete_image(ds_dir, sp, nm):
                deleted += 1
                break
    return {"deleted": deleted, "total": len(names)}


@app.post("/api/dataset/move")
def dataset_move_image(payload: dict) -> dict:
    """把图片在 train/valid/test 之间移动。"""
    split, name, to_split = str(payload.get("split", "")), str(payload.get("name", "")), str(payload.get("to_split", ""))
    if not dataset_mod.safe_split_name(split, name):
        raise HTTPException(status_code=400, detail="参数非法")
    ok = dataset_mod.move_image(_dataset_dir(payload.get("dataset")), split, name, to_split)
    if not ok:
        raise HTTPException(status_code=400, detail="移动失败(目标 split 非法或图片不存在)")
    return {"moved": True, "to_split": to_split}


@app.post("/api/dataset/redistribute")
def dataset_redistribute(payload: dict) -> dict:
    """按比例把所有图片(连标注)随机重新划分 train/valid/test。"""
    try:
        res = dataset_mod.redistribute_splits(
            _dataset_dir(payload.get("dataset")),
            float(payload.get("val_frac", 0.1) or 0),
            float(payload.get("test_frac", 0.1) or 0),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return res


@app.get("/api/dataset/export")
def dataset_export(dataset: str | None = None) -> FileResponse:
    """把数据集打包成 zip 下载(图片+标注+data.yaml;符号链接按真实内容打包)。"""
    import tempfile
    import zipfile
    name = dataset or DEFAULT_DATASET
    ds_dir = _dataset_dir(name)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip", dir=settings.runtime_dir)
    tmp.close()
    with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as zf:
        yml = ds_dir / "data.yaml"
        if yml.exists():
            zf.write(yml, "data.yaml")
        for split in dataset_mod.VALID_SPLITS:
            for sub in ("images", "labels"):
                d = ds_dir / split / sub
                if not d.exists():
                    continue
                for p in d.iterdir():
                    if p.is_file():  # 跟随符号链接写入真实内容
                        zf.write(p, f"{split}/{sub}/{p.name}")
    import os as _os

    from starlette.background import BackgroundTask
    return FileResponse(
        tmp.name, media_type="application/zip", filename=f"{name}.zip",
        background=BackgroundTask(_os.remove, tmp.name),  # 下载后删除临时 zip
    )


@app.post("/api/dataset/autolabel")
def dataset_autolabel(payload: dict) -> dict:
    """用当前生效模型对数据集中的某张图做预标注，返回归一化框供前端填充。"""
    split, name = str(payload.get("split", "")), str(payload.get("name", ""))
    if not dataset_mod.safe_split_name(split, name):
        raise HTTPException(status_code=400, detail="参数非法")
    p = dataset_mod.image_path(_dataset_dir(payload.get("dataset")), split, name)
    if p is None:
        raise HTTPException(status_code=404, detail="图片不存在")
    import cv2
    img = cv2.imread(str(p))
    if img is None:
        raise HTTPException(status_code=400, detail="图片无法读取")
    conf = float(payload.get("conf", 0.25) or 0.25)
    return service.behavior.autolabel(img, conf=conf)


@app.post("/api/dataset/autolabel_batch")
def dataset_autolabel_batch(payload: dict) -> dict:
    """对某个 split 下的图片批量预标注。only_empty=True 时只处理尚未标注(空/缺标注)的图。"""
    split = str(payload.get("split", "train"))
    if split != "all" and split not in dataset_mod.BROWSE_SPLITS:
        raise HTTPException(status_code=400, detail="split 非法")
    ds_dir = _dataset_dir(payload.get("dataset"))
    only_empty = bool(payload.get("only_empty", True))
    conf = float(payload.get("conf", 0.25) or 0.25)
    splits = list(dataset_mod.BROWSE_SPLITS) if split == "all" else [split]
    import cv2
    processed = labeled = total_boxes = 0
    note = ""
    for sp in splits:
        img_dir = ds_dir / sp / "images"
        if not img_dir.exists():
            continue
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() not in dataset_mod.IMG_EXT:
                continue
            name = p.stem
            lbl = ds_dir / sp / "labels" / (name + ".txt")
            if only_empty and lbl.exists() and lbl.read_text().strip():
                continue  # 已有标注，跳过
            img = cv2.imread(str(p))
            if img is None:
                continue
            res = service.behavior.autolabel(img, conf=conf)
            if res.get("note"):
                note = res["note"]
                break  # 模型不可用
            processed += 1
            boxes = res.get("boxes", [])
            if boxes:
                dataset_mod.write_labels(ds_dir, sp, name, boxes)
                labeled += 1
                total_boxes += len(boxes)
        if note:
            break
    return {"processed": processed, "labeled": labeled, "boxes": total_boxes, "note": note, "model": service.behavior.active_model()}


@app.post("/api/dataset/video_frames")
async def dataset_video_frames(
    split: str = Form("unassigned"),
    dataset: str = Form(DEFAULT_DATASET),
    every_sec: float = Form(1.0),
    max_frames: int = Form(60),
    file: UploadFile = File(...),
) -> dict:
    """上传视频并按时间间隔抽帧入库(每帧创建空标注，便于打标)。"""
    if split not in dataset_mod.BROWSE_SPLITS:
        raise HTTPException(status_code=400, detail="split 非法")
    ds_dir = _dataset_dir(dataset)
    every_sec = max(0.1, float(every_sec))
    max_frames = max(1, min(500, int(max_frames)))

    import os
    import tempfile
    suffix = Path(encode_upload_name(file.filename or "video.mp4")).suffix or ".mp4"
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=settings.runtime_dir) as tmp:
            while True:
                chunk = await file.read(1 << 20)
                if not chunk:
                    break
                tmp.write(chunk)
            tmp_path = tmp.name
        import cv2
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="视频无法读取(格式不支持?)")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if fps <= 0:
            fps = 25.0
        step = max(1, int(round(fps * every_sec)))
        saved = 0
        idx = 0
        names: list[str] = []
        while saved < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                ok2, buf = cv2.imencode(".jpg", frame)
                if ok2:
                    nm = dataset_mod.add_single_image(ds_dir, split, "frame.jpg", buf.tobytes())
                    names.append(nm)
                    saved += 1
            idx += 1
        cap.release()
        return {"saved": saved, "split": split, "fps": round(fps, 1), "every_sec": every_sec, "names": names[:5]}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"切帧失败: {exc}") from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


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
def training_start(payload: dict | None = None) -> dict:
    name = str((payload or {}).get("dataset") or DEFAULT_DATASET)
    proj = registry.get(name)
    if proj is None:
        raise HTTPException(status_code=400, detail=f"未知数据集项目: {name}")
    try:
        trainer.start(_read_train_cfg(), _dataset_dir(name) / "data.yaml", out_model=proj["model"], project=name)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return trainer.status()


@app.get("/api/training/status")
def training_status() -> dict:
    st = trainer.status()
    st["active_model"] = registry.active_model()
    # 训练产出 models/<out_model>;是否切换为生效模型由用户在页面显式操作
    st["is_active"] = st.get("out_model") == registry.active_model()
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
