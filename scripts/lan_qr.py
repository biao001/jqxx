#!/usr/bin/env python3
"""打印局域网访问地址 + 终端二维码，供手机扫码访问 DMS 前端。

用法:
    python scripts/lan_qr.py            # 默认前端端口 3000
    python scripts/lan_qr.py 5173       # 指定端口
"""
from __future__ import annotations

import socket
import sys


def lan_ip() -> str:
    """获取本机在局域网中的出口 IP(无需真正联网)。"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def main() -> None:
    import pathlib

    port = sys.argv[1] if len(sys.argv) > 1 else "3000"
    ip = lan_ip()
    # 前端启用了自签证书则用 https(摄像头需要安全上下文)
    certs = pathlib.Path(__file__).resolve().parent.parent / "frontend" / "certs"
    scheme = "https" if (certs / "cert.pem").exists() else "http"
    url = f"{scheme}://{ip}:{port}"

    print("\n  局域网访问地址(确保手机与本机在同一 WiFi/局域网):")
    print(f"  \033[1;36m{url}\033[0m\n")

    try:
        import qrcode

        qr = qrcode.QRCode(border=1)
        qr.add_data(url)
        qr.make(fit=True)
        qr.print_ascii(invert=True)
    except ModuleNotFoundError:
        print("  (未安装 qrcode，无法显示二维码：pip install qrcode)")

    if scheme == "https":
        print("\n  已启用 HTTPS(自签证书)：首次访问浏览器会提示\"不安全\"，点\"高级 → 继续前往\"即可，")
        print("  之后摄像头/车主识别均可用。\n")
    else:
        print("\n  注意：当前是 HTTP，手机/局域网 IP 访问时摄像头被浏览器禁用。")
        print("  生成证书启用 HTTPS 即可用摄像头(见 frontend/certs 与 vite.config)。\n")


if __name__ == "__main__":
    main()
