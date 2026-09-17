import ast
import json
import operator
import os
import re
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.scratchllm.runtime import ScratchAssistant


ROOT = Path(__file__).parent
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8001"))

app = FastAPI(title="Astra Local AI", version="0.1.0")
app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")
assistant = None
conversations: dict[str, list[dict[str, str]]] = {}
OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
MAX_CALC_OPERAND = 1_000_000_000_000
MAX_CALC_RESULT = 1_000_000_000_000_000


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = None
    temperature: float = Field(default=0.35, ge=0.1, le=2.0)


def get_default_model() -> str:
    configured = (os.getenv("OLLAMA_MODEL") or "").strip()
    if configured:
        return configured
    try:
        with urllib_request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8") or "{}")
        models = payload.get("models", []) if isinstance(payload, dict) else []
        if models:
            first_model = models[0].get("name")
            if isinstance(first_model, str) and first_model.strip():
                return first_model.strip()
    except (urllib_error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
        pass
    return "qwen2.5:7b"


def get_provider_status(available: bool = False, model: str | None = None) -> dict[str, object]:
    chosen_model = model or get_default_model()
    return {
        "provider": "ollama" if available else "local",
        "model": chosen_model,
        "ollama_available": bool(available),
        "local_model_loaded": assistant is not None,
        "host": APP_HOST,
        "port": APP_PORT,
    }


def ollama_available() -> bool:
    try:
        with urllib_request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=2) as response:
            return response.status == 200
    except (urllib_error.URLError, TimeoutError, ValueError):
        return False


def call_ollama(prompt: str, temperature: float = 0.35) -> str:
    if not ollama_available():
        raise RuntimeError("Ollama is not running or not reachable on the configured host.")
    body = json.dumps({
        "model": get_default_model(),
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }).encode("utf-8")
    request = urllib_request.Request(
        f"{OLLAMA_BASE_URL}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib_request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("response"), str):
        return payload["response"].strip()
    if isinstance(payload, list):
        pieces = [item.get("response", "") for item in payload if isinstance(item, dict)]
        text = "".join(pieces).strip()
        if text:
            return text
    raise RuntimeError("Ollama returned an unexpected payload.")


def calculate(expression: str) -> int | float:
    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            left, right = evaluate(node.left), evaluate(node.right)
            if abs(left) > MAX_CALC_OPERAND or abs(right) > MAX_CALC_OPERAND:
                raise ValueError("number too large")
            result = OPERATORS[type(node.op)](left, right)
            if abs(result) > MAX_CALC_RESULT:
                raise ValueError("result too large")
            return result
        raise ValueError("unsupported expression")

    return evaluate(ast.parse(expression, mode="eval"))


def known_answer(message: str) -> str | None:
    normalized = message.strip().lower().rstrip("?.!")
    if normalized in {"hi", "hello", "hii", "hey"}:
        return "Hello. I am Astra, a local AI assistant backed by Ollama when available and by a scratch-trained fallback model otherwise."
    if "your name" in normalized or "who are you" in normalized:
        return "My name is Astra. I run locally using a local scratch model and can also use Ollama if it is installed and running."
    if "what can you do" in normalized:
        return "I can answer supported local questions, run basic arithmetic, and route to Ollama for stronger local chat responses when available."
    expression = re.fullmatch(r"[0-9+\-*/().\s^]+", normalized)
    if expression:
        try:
            value = calculate(normalized.replace("^", "**"))
            return f"{value:g}" if isinstance(value, float) else str(value)
        except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
            return "I could not safely calculate that expression."
    return None


@app.get("/")
def home():
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/api/health")
def health():
    ollama_active = ollama_available()
    parameters = (
        sum(parameter.numel() for parameter in assistant.model.parameters())
        if assistant is not None else 0
    )
    provider = get_provider_status(available=ollama_active)
    return {
        "status": "ok",
        "provider": provider["provider"],
        "model": provider["model"],
        "ollama_available": ollama_active,
        "local_model_loaded": assistant is not None,
        "parameters": parameters,
        "host": APP_HOST,
        "port": APP_PORT,
    }


@app.post("/api/chat")
def chat(request: ChatRequest):
    conversation_id = request.conversation_id or str(uuid4())
    history = conversations.setdefault(conversation_id, [])
    history.append({"role": "user", "content": request.message})

    answer = known_answer(request.message)
    if answer is None:
        if ollama_available():
            try:
                answer = call_ollama(request.message, temperature=request.temperature)
            except Exception:
                answer = None
        if answer is None and assistant is not None:
            answer = assistant.generate(request.message, token_count=180, temperature=request.temperature)
        if answer is None:
            answer = (
                "I do not have reliable knowledge for that question yet. "
                "This app is configured to use Ollama first, then a local scratch checkpoint fallback."
            )

    history.append({"role": "assistant", "content": answer})
    provider_name = "ollama" if ollama_available() else "local"
    return {"conversation_id": conversation_id, "answer": answer, "model": get_default_model(), "provider": provider_name}


@app.get("/api/conversations/{conversation_id}")
def conversation(conversation_id: str):
    return {"conversation_id": conversation_id, "messages": conversations.get(conversation_id, [])}


checkpoint = ROOT / "checkpoints" / "astra_50m.pt"
if not checkpoint.exists():
    checkpoint = ROOT / "checkpoints" / "astra_30m.pt"
if not checkpoint.exists():
    checkpoint = ROOT / "checkpoints" / "tiny_lm.pt"
if checkpoint.exists():
    assistant = ScratchAssistant(str(checkpoint))