from __future__ import annotations

from pathlib import Path
from typing import Any


def write_markdown_report(report_dir: Path, result: dict[str, Any]) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    job_id = str(result["job_id"])
    report_path = report_dir / f"{job_id}.md"
    source = result.get("source", {})
    stats = result.get("stats", {})
    detections = result.get("detections", [])

    lines = [
        "# 驾驶状态分析报告",
        "",
        f"- 任务 ID: {job_id}",
        f"- 数据来源: {source.get('kind', 'unknown')}",
        f"- 文件名称: {source.get('name', '--')}",
        f"- 驾驶状态评分: {stats.get('score', '--')}/100",
        f"- 状态等级: {stats.get('status', '--')}",
        "",
        "## 检测结果列表",
        "",
    ]

    if detections:
        lines.append("| 时间 | 类型 | 置信度 | 等级 |")
        lines.append("| --- | --- | --- | --- |")
        for detection in detections:
            lines.append(
                "| {timestamp} | {type} | {confidence:.2f} | {severity} |".format(
                    timestamp=detection.get("timestamp", "--"),
                    type=detection.get("type", "--"),
                    confidence=float(detection.get("confidence", 0.0)),
                    severity=detection.get("severity", "--"),
                )
            )
    else:
        lines.append("未检测到风险行为。")

    lines.extend(
        [
            "",
            "## 大模型分析",
            "",
            str(result.get("llm_analysis") or "暂无大模型分析。"),
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
