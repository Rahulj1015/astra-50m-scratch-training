from pathlib import Path
import ast
import operator
import re
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.scratchllm.runtime import ScratchAssistant


ROOT = Path(__file__).parent
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
        return "Hello. I am Astra, a local scratch-trained assistant."
    if "your name" in normalized or "who are you" in normalized:
        return "My name is Astra. I run locally using your scratch-trained model."
    if "what can you do" in normalized:
        return "I can answer supported local questions, run basic arithmetic, and generate text from my trained corpus."
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
    parameters = (
        sum(parameter.numel() for parameter in assistant.model.parameters())
        if assistant is not None else 0
    )
    return {"status": "ok", "model": "Astra", "loaded": assistant is not None, "parameters": parameters}


@app.post("/api/chat")
def chat(request: ChatRequest):
    if assistant is None:
        raise HTTPException(status_code=503, detail="Model checkpoint is not available")
    conversation_id = request.conversation_id or str(uuid4())
    history = conversations.setdefault(conversation_id, [])
    history.append({"role": "user", "content": request.message})
    answer = known_answer(request.message)
    if answer is None:
        answer = (
            "I do not have reliable knowledge for that question yet. "
            "This scratch-trained model currently knows only its small training corpus. "
            "Add documents or a larger dataset before trusting an answer."
        )
    history.append({"role": "assistant", "content": answer})
    return {"conversation_id": conversation_id, "answer": answer, "model": "Astra"}


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