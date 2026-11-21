import ast
import math
import operator
from pathlib import Path

from langchain_core.tools import StructuredTool


ALLOWED_ROOT = (Path.cwd() / "workspace").resolve()
ALLOWED_ROOT.mkdir(parents=True, exist_ok=True)


def calculate(expression: str) -> str:
    if len(expression) > 200:
        return "Calculator error: expression is too long"

    functions = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "floor": math.floor,
        "ceil": math.ceil,
        "abs": abs,
        "round": round,
    }
    constants = {"pi": math.pi, "e": math.e}

    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
        ast.FloorDiv: operator.floordiv,
    }

    def evaluate(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            if isinstance(node.op, ast.Pow):
                right = evaluate(node.right)
                if not isinstance(right, (int, float)) or abs(float(right)) > 1000:
                    raise ValueError("Exponent is too large")
            return operators[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in operators:
            return operators[type(node.op)](evaluate(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in functions or node.keywords:
                raise ValueError("Function is not allowed")
            arguments = [evaluate(argument) for argument in node.args]
            if len(arguments) > 3:
                raise ValueError("Too many function arguments")
            return functions[node.func.id](*arguments)
        if isinstance(node, ast.Name) and node.id in constants:
            return constants[node.id]
        raise ValueError("Only arithmetic operations are allowed")

    try:
        return str(evaluate(ast.parse(expression, mode="eval").body))
    except (SyntaxError, ZeroDivisionError, ValueError, TypeError) as exc:
        return f"Calculator error: {exc}"


def read_local_file(path: str) -> str:
    relative = Path(path)
    if relative.is_absolute():
        return "Error: file access is restricted to the project workspace"
    denied_names = {".env", ".env.local", ".git", ".streamlit"}
    if relative.parts and (relative.parts[0] in denied_names or relative.name.startswith(".env")):
        return "Error: file access is restricted to the project workspace"

    candidate = (ALLOWED_ROOT / relative).resolve(strict=False)
    if not candidate.is_relative_to(ALLOWED_ROOT):
        return "Error: file access is restricted to the project workspace"
    if candidate.is_symlink():
        return "Error: symbolic links are not allowed"
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
