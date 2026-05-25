# DMS 环境配置教程

本文档用于配置和运行当前 DMS 前后端联调环境。推荐在 Windows + PowerShell 下按步骤执行。

## 1. 基础要求

- Python 3.11
- Conda 或 Miniconda
- Node.js 20+ 与 npm
- Git
- 可用摄像头权限

后端不要使用 Python 3.13。当前实时疲劳检测依赖 `mediapipe`，该依赖更适合在 Python 3.11 环境中运行。

## 2. 后端 Python 环境

在仓库根目录执行：

```powershell
conda create -n dms-demo python=3.11 -y
conda activate dms-demo
python -m pip install -r backend\requirements.txt
```

如果不想激活环境，也可以直接使用环境里的 Python：

```powershell
C:\Users\ASUS\.conda\envs\dms-demo\python.exe -m pip install -r backend\requirements.txt
```

验证关键依赖：

```powershell
python -c "import cv2, torch, mediapipe, ultralytics, fastapi, uvicorn; print('backend deps ok')"
```

## 3. 模型文件

疲劳检测模型需要：

```text
fatigue\outputs\fatigue_b\checkpoints\best_model.pt
```

MediaPipe 人脸模型默认使用：

```text
fatigue\models\mediapipe\face_landmarker.task
```

行为识别会使用 YOLO 权重：

```text
yolov8n.pt
yolov8n-pose.pt
```

如果本地没有这两个文件，`ultralytics` 首次运行时会自动下载。可选增强模型放在：

```text
behavior_algo\models\seatbelt.pt
behavior_algo\models\smoking.pt
behavior_algo\models\unified.pt
```

没有可选模型时，行为识别仍会使用基础 YOLO + pose + 规则分支运行。

## 4. 后端环境变量

复制示例文件：

```powershell
Copy-Item backend\.env.example backend\.env
```

编辑 `backend\.env`：

```env
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
SILICONFLOW_API_KEY=replace-with-your-key
SILICONFLOW_MODEL=Qwen/Qwen3-8B
LLM_TIMEOUT_SECONDS=60
DMS_ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001
DMS_FRAME_STRIDE=10
DMS_MAX_VIDEO_FRAMES=240
```

不要把真实 API Key 提交到 Git。

## 5. 启动后端

推荐显式使用 `dms-demo` 的 Python：

```powershell
C:\Users\ASUS\.conda\envs\dms-demo\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

启动后检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health | ConvertTo-Json -Depth 8
```

正常情况下应看到：

```json
{
  "ok": true,
  "capabilities": {
    "behavior": { "mode": "model" },
    "fatigue": {
      "mode": "model",
      "predictor": "model",
      "feature_extractor": "mediapipe"
    }
  }
}
```

如果 `behavior.mode` 是 `not_loaded`，通常表示行为识别尚未收到第一帧；发送相机/视频帧后会懒加载。若显示 `fallback`，查看 `error` 字段，一般是依赖或模型文件问题。

## 6. 前端环境

进入前端目录：

```powershell
cd frontend
npm install
```

复制环境变量：

```powershell
Copy-Item .env.example .env
```

`frontend\.env` 默认配置：

```env
VITE_BACKEND_URL=http://localhost:8000
VITE_CAMERA_FPS=3
```

启动前端：

```powershell
npm run dev
```

浏览器打开：

```text
http://localhost:3000
```

如果 3000 被占用，Vite 会提示其他端口，例如 `http://localhost:3001`。

## 7. 推荐启动顺序

1. 启动后端：

```powershell
C:\Users\ASUS\.conda\envs\dms-demo\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

2. 新开一个 PowerShell，启动前端：

```powershell
cd frontend
npm run dev
```

3. 打开前端页面，上传视频或启动本地相机。

上传视频会按相机模式逐帧发送到后端 WebSocket；本地相机也会实时逐帧检测。视频区域会显示行为识别实时标签和 bbox，疲劳检测会显示实时标签与置信度。

## 8. 验证命令

后端测试：

```powershell
C:\Users\ASUS\.conda\envs\dms-demo\python.exe -m pytest backend\tests -v
```

后端编译检查：

```powershell
C:\Users\ASUS\.conda\envs\dms-demo\python.exe -m compileall backend\app
```

前端类型检查与构建：

```powershell
cd frontend
npm run lint
npm run build
```

## 9. 常见问题

### `/health` 显示 `fatigue.feature_extractor=fallback`

说明 MediaPipe 特征提取没有加载成功。检查：

- 是否使用 Python 3.11 环境
- 是否安装 `mediapipe`
- `fatigue\models\mediapipe\face_landmarker.task` 是否存在

### `/health` 显示 `behavior.mode=fallback`

说明行为识别模型没有完整加载。检查：

- 是否安装 `ultralytics`
- `yolov8n.pt` 和 `yolov8n-pose.pt` 是否可下载或已存在
- 控制台里是否有模型加载错误

### 前端无法连接后端

检查后端是否监听 8000：

```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen
```

检查 `frontend\.env`：

```env
VITE_BACKEND_URL=http://localhost:8000
```

### 摄像头打不开

按顺序检查：

- 关闭 Teams、Zoom、浏览器等占用摄像头的软件
- Windows 设置中允许桌面应用访问摄像头
- 浏览器页面允许相机权限
- 重新插拔或重启摄像头设备

### 打哈欠没有标签

先确认后端健康状态：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health | ConvertTo-Json -Depth 8
```

需要看到：

```text
fatigue.mode = model
fatigue.feature_extractor = mediapipe
```

如果不是这个状态，说明没有走完整实时疲劳检测链路。

## 10. 停止服务

停止前端或后端：在对应终端按 `Ctrl+C`。

如果需要查找端口占用：

```powershell
Get-NetTCPConnection -LocalPort 8000,3000,3001 -State Listen
```

按进程号停止：

```powershell
Stop-Process -Id <PID>
```
