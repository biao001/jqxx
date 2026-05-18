# Real DMS Backend Design

## Goal

Add a local backend that connects the existing driving behavior and fatigue algorithms to the dashboard with real video upload analysis, real-time camera frame analysis, driving scores, detection lists, large-model analysis, and downloadable reports.

## Recommended Architecture

Use a Python FastAPI backend because the existing algorithm code is Python-based and depends on OpenCV, MediaPipe, PyTorch, and Ultralytics. The React dashboard remains the browser UI and calls backend HTTP/WebSocket APIs.

The backend owns all secret handling. SiliconFlow OpenAI-compatible API settings are read from `backend/.env` and are never bundled into frontend JavaScript.

## Data Flow

### Uploaded Video

1. Frontend uploads a real video through `POST /api/videos/analyze`.
2. Backend saves the upload into a local runtime directory.
3. Backend samples frames, calls the behavior detector, extracts fatigue features, runs fatigue inference when a checkpoint exists, and otherwise uses the existing fatigue heuristic preview path.
4. Backend aggregates raw frame outputs into:
   - driving state score
   - detection result list
   - component stats
   - large-model analysis text
   - report metadata
5. Frontend displays the returned result and enables `GET /api/reports/{job_id}` download.

### Real-Time Camera

1. Frontend opens the local camera with `getUserMedia`.
2. Frontend draws the video element to a hidden canvas at a configured frame rate.
3. Frames are JPEG-encoded and sent to `WS /ws/camera`.
4. Backend decodes each frame and runs lightweight per-frame behavior detection plus an in-memory fatigue window.
5. Backend returns unified JSON results per frame. The browser keeps local video preview for low latency and overlays the latest data in the existing dashboard panels.

## API Surface

- `GET /health`
- `POST /api/videos/analyze`
- `GET /api/reports/{job_id}`
- `WS /ws/camera`

## Algorithm Policy

There is no algorithm selector in the UI. The default pipeline always runs the final behavior detector and final fatigue detector/heuristic fallback.

If optional heavy model files are missing, the backend must still return structured results with clear capability metadata instead of crashing the UI.

## Scoring

The backend computes a 0-100 driving safety score from behavior risk, fatigue risk, camera validity, and driver presence. Higher score means safer driving. It also returns normalized stats for focus, reaction, compliance, fatigue, and stability.

## Large Model

The backend calls SiliconFlow through an OpenAI-compatible `/chat/completions` request. For camera streaming, LLM calls are throttled and based on an aggregated recent window rather than every frame. For uploaded video, the LLM analysis is generated once per completed analysis.

## Error Handling

HTTP endpoints return structured JSON errors. WebSocket frame errors return a per-frame error payload while keeping the socket open when possible.

## Verification

Backend tests cover result aggregation, scoring, report generation, and OpenAI-compatible request shaping without calling external services. Frontend verification uses TypeScript compilation and production build.
