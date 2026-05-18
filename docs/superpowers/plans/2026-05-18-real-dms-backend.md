# Real DMS Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local FastAPI backend and connect the React dashboard to real uploaded-video and real-time camera analysis.

**Architecture:** FastAPI provides HTTP upload/report APIs and a WebSocket camera stream API. A backend analysis service adapts existing Python algorithm modules into a unified result contract consumed by the dashboard. Frontend code removes mock data and algorithm selectors, sending real files and camera frames to backend endpoints.

**Tech Stack:** Python 3, FastAPI, OpenCV, NumPy, pytest, Vite, React, TypeScript, WebSocket.

---

## File Structure

- Create `backend/requirements.txt`: backend Python dependencies.
- Create `backend/.env.example`: non-secret configuration names for the local backend.
- Create `backend/app/__init__.py`: package marker.
- Create `backend/app/config.py`: environment-driven backend settings.
- Create `backend/app/schemas.py`: response type helpers and normalized labels.
- Create `backend/app/scoring.py`: deterministic driving-score aggregation.
- Create `backend/app/llm.py`: SiliconFlow OpenAI-compatible client.
- Create `backend/app/reports.py`: report file generation.
- Create `backend/app/analysis.py`: video and frame analysis orchestration.
- Create `backend/app/main.py`: FastAPI app, routes, and WebSocket endpoint.
- Create `backend/tests/test_scoring.py`: unit tests for scoring.
- Create `backend/tests/test_reports.py`: unit tests for report generation.
- Create `backend/tests/test_llm.py`: unit tests for request shaping with a fake transport.
- Modify `frontend/.env.example`: add backend URL configuration.
- Modify `frontend/src/types.ts`: replace mock-oriented types with backend contract types.
- Modify `frontend/src/components/ControlPanel.tsx`: remove algorithm selectors and add camera start/stop state.
- Modify `frontend/src/components/UploadModal.tsx`: use real file input.
- Modify `frontend/src/components/Sidebar.tsx`: render real stats, detections, LLM analysis, and report download.
- Modify `frontend/src/App.tsx`: manage backend calls, WebSocket camera streaming, upload analysis, and local video preview.

## Task 1: Backend Scoring Core

**Files:**
- Create: `backend/app/scoring.py`
- Test: `backend/tests/test_scoring.py`

- [ ] **Step 1: Write failing scoring tests**

```python
from backend.app.scoring import compute_driving_stats, normalize_detection


def test_compute_driving_stats_penalizes_high_risk_events():
    stats = compute_driving_stats(
        behavior_risk=70.0,
        fatigue_risk=80.0,
        driver_present=True,
        camera_ok=True,
        detections=[{"severity": "high"}, {"severity": "medium"}],
    )

    assert stats["score"] < 55
    assert stats["status"] in {"危险", "警告"}
    assert stats["fatigue"] == 20
    assert stats["compliance"] < 80


def test_normalize_detection_maps_backend_event_to_ui_shape():
    detection = normalize_detection(
        event={"type": "phone_use", "label_zh": "驾驶中使用手机", "confidence": 0.91, "severity": "high"},
        timestamp_seconds=12.4,
        prefix="behavior",
        index=2,
    )

    assert detection["id"] == "behavior-2"
    assert detection["type"] == "驾驶中使用手机"
    assert detection["timestamp"] == "00:12"
    assert detection["confidence"] == 0.91
    assert detection["severity"] == "high"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest backend/tests/test_scoring.py -v`

Expected: FAIL because `backend.app.scoring` does not exist.

- [ ] **Step 3: Implement scoring**

Create deterministic functions that:
- format timestamps as `MM:SS`
- normalize behavior/fatigue detections into UI shape
- compute score as `100 - weighted risk penalties`
- return stats keys: `score`, `status`, `focus`, `reaction`, `compliance`, `fatigue`, `stability`

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest backend/tests/test_scoring.py -v`

Expected: PASS.

## Task 2: Backend LLM and Report Helpers

**Files:**
- Create: `backend/app/config.py`
- Create: `backend/app/llm.py`
- Create: `backend/app/reports.py`
- Test: `backend/tests/test_llm.py`
- Test: `backend/tests/test_reports.py`

- [ ] **Step 1: Write failing tests**

Test LLM request building with a fake `urlopen` function and report generation with a temporary directory.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest backend/tests/test_llm.py backend/tests/test_reports.py -v`

Expected: FAIL because modules do not exist.

- [ ] **Step 3: Implement helpers**

`llm.py` should call `POST {base_url}/chat/completions` with `Authorization: Bearer {api_key}` and parse `choices[0].message.content`. If API info is missing or request fails, return a deterministic fallback analysis text.

`reports.py` should write a Markdown report under `backend/runtime/reports/{job_id}.md`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest backend/tests/test_llm.py backend/tests/test_reports.py -v`

Expected: PASS.

## Task 3: Backend Analysis and Routes

**Files:**
- Create: `backend/app/analysis.py`
- Create: `backend/app/main.py`
- Create: `backend/requirements.txt`
- Create: `backend/.env.example`

- [ ] **Step 1: Implement analysis service**

Add lazy algorithm loading. If heavy algorithm dependencies or model files are unavailable, return structured capability metadata and fallback heuristic detections instead of crashing.

- [ ] **Step 2: Implement routes**

Routes:
- `GET /health`
- `POST /api/videos/analyze`
- `GET /api/reports/{job_id}`
- `WS /ws/camera`

- [ ] **Step 3: Verify backend imports**

Run: `python -m compileall backend/app`

Expected: all backend files compile.

## Task 4: Frontend Real Data Integration

**Files:**
- Modify: `frontend/.env.example`
- Modify: `frontend/src/types.ts`
- Modify: `frontend/src/components/ControlPanel.tsx`
- Modify: `frontend/src/components/UploadModal.tsx`
- Modify: `frontend/src/components/Sidebar.tsx`
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Remove algorithm selector UI**

Control panel should show fixed algorithms as status text only: behavior recognition and fatigue detection.

- [ ] **Step 2: Replace mock upload modal**

Use `<input type="file" accept="video/*">` and pass the real `File` object to `App`.

- [ ] **Step 3: Add camera streaming**

Use `navigator.mediaDevices.getUserMedia`, canvas frame capture, and `WebSocket` binary JPEG frames.

- [ ] **Step 4: Add real API calls**

Use `VITE_BACKEND_URL` for HTTP and WebSocket endpoints. Update stats, detections, LLM analysis, and report download from backend responses.

- [ ] **Step 5: Verify frontend**

Run: `npm run lint` and `npm run build` from `frontend/`.

Expected: both pass.

## Task 5: End-to-End Verification

**Files:**
- Modify as needed from previous tasks only.

- [ ] **Step 1: Run backend tests**

Run: `python -m pytest backend/tests -v`

Expected: PASS.

- [ ] **Step 2: Run backend compile check**

Run: `python -m compileall backend/app`

Expected: PASS.

- [ ] **Step 3: Run frontend verification**

Run from `frontend/`: `npm run lint` and `npm run build`

Expected: PASS.

- [ ] **Step 4: Review git diff**

Run: `git diff --stat` and `git status --short --branch`.

Expected: only planned files changed.
