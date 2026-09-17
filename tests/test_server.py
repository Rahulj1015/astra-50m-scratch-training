from server import get_default_model, get_provider_status


def test_default_model_uses_ollama_name():
    model = get_default_model()
    assert isinstance(model, str) and model.strip()
    assert model.lower() in {"qwen2.5:7b", "llama3.2", "llama3.1", "mistral"}


def test_provider_status_reports_local_fallback():
    status = get_provider_status(available=False, model="qwen2.5:7b")
    assert status["provider"] == "local"
    assert status["model"] == "qwen2.5:7b"
    assert status["ollama_available"] is False
