"""DMS 后端启动入口 —— 默认绑定 0.0.0.0(局域网可访问)。

用法:
    cd backend && python run.py
环境变量(可选):
    DMS_HOST   监听地址，默认 0.0.0.0(局域网) / 设 127.0.0.1 则仅本机
    DMS_PORT   端口，默认 8000
    DMS_RELOAD 设为 1 开启热重载(开发用)
"""
from __future__ import annotations

import os

import uvicorn

if __name__ == "__main__":
    host = os.getenv("DMS_HOST", "0.0.0.0")
    port = int(os.getenv("DMS_PORT", "8000"))
    reload = os.getenv("DMS_RELOAD", "") in ("1", "true", "True")
    print(f"[DMS] 后端启动: http://{host}:{port}  (局域网内其他设备用本机 IP:{port} 访问)")
    uvicorn.run("app.main:app", host=host, port=port, reload=reload)
