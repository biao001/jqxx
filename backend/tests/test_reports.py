from backend.app.reports import write_markdown_report


def test_write_markdown_report_contains_real_result_sections(tmp_path):
    result = {
        "job_id": "job-1",
        "source": {"kind": "upload", "name": "drive.mp4"},
        "stats": {"score": 82, "status": "正常"},
        "detections": [
            {"timestamp": "00:02", "type": "驾驶中使用手机", "confidence": 0.88, "severity": "high"}
        ],
        "llm_analysis": "驾驶状态总体平稳，但存在短时风险。",
    }

    report_path = write_markdown_report(tmp_path, result)

    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "驾驶状态分析报告" in content
    assert "drive.mp4" in content
    assert "驾驶中使用手机" in content
    assert "驾驶状态总体平稳" in content
