import ast
import operator
from pathlib import Path

from langchain_core.tools import StructuredTool


ALLOWED_ROOT = Path.cwd().resolve()


def calculate(expression: str) -> str:
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
    }

    def evaluate(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            return operators[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in operators:
            return operators[type(node.op)](evaluate(node.operand))
        raise ValueError("Only arithmetic operations are allowed")

    try:
        return str(evaluate(ast.parse(expression, mode="eval").body))
    except (SyntaxError, ZeroDivisionError, ValueError, TypeError) as exc:
        return f"Calculator error: {exc}"


def read_local_file(path: str) -> str:
    candidate = (ALLOWED_ROOT / path).resolve()
    if candidate != ALLOWED_ROOT and ALLOWED_ROOT not in candidate.parents:
        return "Error: file access is restricted to the project workspace"
    if not candidate.is_file():
        return f"Error: file not found: {path}"
    if candidate.stat().st_size > 200_000:
        return "Error: file exceeds the 200 KB reading limit"
    try:
        return candidate.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return f"Error: could not read file: {exc}"


def build_core_tools():
    return [
        StructuredTool.from_function(
            func=calculate,
            name="calculator",
            description="Calculate a mathematical expression safely. Input: one arithmetic expression.",
        ),
        StructuredTool.from_function(
            func=read_local_file,
            name="local_file_reader",
            description="Read a UTF-8 text file inside the project workspace. Input: relative file path.",
        ),
    ]
