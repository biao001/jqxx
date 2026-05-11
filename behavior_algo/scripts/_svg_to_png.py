"""SVG → PNG 转换，用于视觉检查"""
import os
from pathlib import Path
from playwright.sync_api import sync_playwright


def convert(svg_dir: str = "docs/diagrams", width: int = 1600, height: int = 900):
    d = Path(svg_dir).resolve()
    svgs = sorted(d.glob("*.svg"))
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": width, "height": height})
        for svg in svgs:
            url = svg.as_uri()
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_timeout(500)
            png = svg.with_suffix(".png")
            try:
                # 用 clip 代替 full_page，规避字体等待
                page.screenshot(
                    path=str(png),
                    clip={"x": 0, "y": 0, "width": 1600, "height": 900},
                    timeout=15000,
                )
                print(f"[ok] {png.name}")
            except Exception as e:
                print(f"[err] {png.name}: {e}")
        browser.close()


if __name__ == "__main__":
    convert()
