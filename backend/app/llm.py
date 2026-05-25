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

    try:
        content = _post_chat(settings, build_prompt(summary), max_tokens=700, temperature=0.2, urlopen=urlopen)
        return content or _fallback(summary, "大模型返回内容为空")
    except Exception as exc:
        return _fallback(summary, f"大模型调用失败：{exc}")


def _post_chat(
    settings: LlmSettings,
    messages: list[dict[str, str]],
    max_tokens: int = 700,
    temperature: float = 0.3,
    urlopen: Callable[..., Any] = default_urlopen,
) -> str:
    """调用 OpenAI 兼容 chat 接口，返回文本内容；失败抛异常。"""
    endpoint = settings.base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": settings.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
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
    with urlopen(request, timeout=settings.timeout_seconds) as response:
        data = json.loads(response.read().decode("utf-8"))
    return str(data["choices"][0]["message"]["content"]).strip()


REVIEW_SYSTEM = (
    "你是一位专业、友善的驾驶安全教练。基于一次行程的结构化驾驶监测数据，"
    "给出简洁的中文点评，包含：1) 总体评价(一句话 + 评级 优秀/良好/一般/较差) "
    "2) 主要问题(按严重度，最多3条) 3) 针对性改进建议(可执行，2-3条)。"
    "语气鼓励但具体，不要编造数据中没有的内容。用 Markdown 分点输出，控制在 300 字内。"
)

CHAT_SYSTEM = (
    "你是驾驶状态监测系统(DMS)的 AI 助手。根据提供的驾驶监测数据回答用户问题，"
    "简洁、专业、口语化，只依据数据作答，数据没有的不要编造。中文回答，控制在 150 字内。"
)


def generate_review(settings: LlmSettings, summary: dict[str, Any], urlopen: Callable[..., Any] = default_urlopen) -> str:
    if not settings.api_key:
        return _fallback(summary, "未配置大模型 API")
    messages = [
        {"role": "system", "content": REVIEW_SYSTEM},
        {"role": "user", "content": "以下是一次行程的驾驶监测汇总数据，请点评：\n" + json.dumps(summary, ensure_ascii=False, indent=2)},
    ]
    try:
        return _post_chat(settings, messages, max_tokens=800, temperature=0.4, urlopen=urlopen) or _fallback(summary, "点评为空")
    except Exception as exc:
        return _fallback(summary, f"点评生成失败：{exc}")


def generate_chat(
    settings: LlmSettings,
    question: str,
    context: dict[str, Any],
    history: list[dict[str, str]] | None = None,
    urlopen: Callable[..., Any] = default_urlopen,
) -> str:
    if not settings.api_key:
        return "未配置大模型 API，无法回答。"
    messages: list[dict[str, str]] = [
        {"role": "system", "content": CHAT_SYSTEM + "\n\n当前驾驶监测数据:\n" + json.dumps(context, ensure_ascii=False)},
    ]
    for h in (history or [])[-6:]:
        role = h.get("role")
        if role in ("user", "assistant") and h.get("content"):
            messages.append({"role": role, "content": str(h["content"])[:500]})
    messages.append({"role": "user", "content": question[:500]})
    try:
        return _post_chat(settings, messages, max_tokens=500, temperature=0.5, urlopen=urlopen) or "(无回答)"
    except Exception as exc:
        return f"回答失败：{exc}"
