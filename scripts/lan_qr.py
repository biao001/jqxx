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
    port = sys.argv[1] if len(sys.argv) > 1 else "3000"
    ip = lan_ip()
    url = f"http://{ip}:{port}"

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

    print("\n  注意：手机浏览器调用摄像头需要 HTTPS 或 localhost；通过 http://内网IP")
    print("  访问时摄像头/车主识别可能被浏览器拦截。仅看实时分析/看板/上传视频不受影响。")
    print("  如需手机端摄像头，请改用 HTTPS(见 README 说明)。\n")


if __name__ == "__main__":
    main()
