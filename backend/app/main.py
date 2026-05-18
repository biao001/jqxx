from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .analysis import AnalysisService, decode_image_bytes, encode_upload_name
from .config import get_settings


settings = get_settings()
settings.runtime_dir.mkdir(parents=True, exist_ok=True)
service = AnalysisService(settings)

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
