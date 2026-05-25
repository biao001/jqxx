import json

from backend.app.llm import LlmSettings, generate_llm_analysis


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        payload = {"choices": [{"message": {"content": "真实大模型分析"}}]}
        return json.dumps(payload).encode("utf-8")


def test_generate_llm_analysis_uses_openai_compatible_payload():
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    text = generate_llm_analysis(
        settings=LlmSettings(
            base_url="https://api.siliconflow.cn/v1",
            api_key="secret",
            model="Qwen/Qwen3-8B",
            timeout_seconds=7,
        ),
        summary={"score": 80, "detections": [{"type": "疲劳驾驶"}]},
        urlopen=fake_urlopen,
    )

    assert text == "真实大模型分析"
    assert captured["url"] == "https://api.siliconflow.cn/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["body"]["model"] == "Qwen/Qwen3-8B"
    assert captured["body"]["messages"][0]["role"] == "system"
    assert captured["timeout"] == 7


def test_generate_llm_analysis_retries_once_after_read_timeout():
    attempts = 0

    def flaky_urlopen(request, timeout):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("The read operation timed out")
        return FakeResponse()

    text = generate_llm_analysis(
        settings=LlmSettings(
            base_url="https://api.siliconflow.cn/v1",
            api_key="secret",
            model="Qwen/Qwen3-8B",
            timeout_seconds=7,
        ),
        summary={"score": 80, "detections": []},
        urlopen=flaky_urlopen,
    )

    assert text == "真实大模型分析"
    assert attempts == 2


def test_generate_llm_analysis_falls_back_without_key():
    text = generate_llm_analysis(
        settings=LlmSettings(
            base_url="https://api.siliconflow.cn/v1",
            api_key="",
            model="Qwen/Qwen3-8B",
        ),
        summary={"score": 60, "detections": []},
    )

    assert "未配置大模型 API" in text
