from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable
from urllib.request import Request, urlopen as default_urlopen


@dataclass(frozen=True)
class LlmSettings:
    base_url: str
    api_key: str
    model: str
    timeout_seconds: int = 20


def _fallback(summary: dict[str, Any], reason: str) -> str:
    score = summary.get("score") or summary.get("stats", {}).get("score", "--")
    detections = summary.get("detections") or []
    event_count = len(detections) if isinstance(detections, list) else 0
    return f"{reason}。当前驾驶状态评分为 {score}，检测到 {event_count} 条风险记录。建议结合检测列表复核驾驶行为。"


def build_prompt(summary: dict[str, Any]) -> list[dict[str, str]]:
    system = (
        "你是驾驶状态监测系统的安全分析助手。"
        "请基于结构化检测结果给出简洁、具体、可执行的中文分析，不要编造不存在的数据。"
    )
    user = "请分析以下驾驶状态检测结果：\n" + json.dumps(summary, ensure_ascii=False, indent=2)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def generate_llm_analysis(
    settings: LlmSettings,
    summary: dict[str, Any],
    urlopen: Callable[..., Any] = default_urlopen,
) -> str:
    if not settings.api_key:
        return _fallback(summary, "未配置大模型 API")
    if not settings.base_url or not settings.model:
        return _fallback(summary, "大模型配置不完整")

    endpoint = settings.base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": settings.model,
        "messages": build_prompt(summary),
        "temperature": 0.2,
        "max_tokens": 700,
        "stream": False,
    }
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {settings.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=settings.timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
        return str(content).strip() or _fallback(summary, "大模型返回内容为空")
    except Exception as exc:
        return _fallback(summary, f"大模型调用失败：{exc}")
