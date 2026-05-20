"""SymPy-backed math evaluator for arithmetic and symbolic expressions."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)


@dataclass
class EvaluationResult:
    expression: str
    answer: str
    success: bool
    confidence: float = 0.0
    error: str = ""


class SympyEvaluator:
    """Evaluates cleaned expressions without eval or local ML dependencies."""

    def __init__(self) -> None:
        self.x = sp.Symbol("x")
        self.local_dict = {
            "x": self.x,
            "pi": sp.pi,
            "E": sp.E,
            "sqrt": sp.sqrt,
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "log": sp.log,
            "ln": sp.log,
            "exp": sp.exp,
            "Abs": sp.Abs,
        }
        self.transformations = standard_transformations + (
            implicit_multiplication_application,
            convert_xor,
        )

    def evaluate(self, expression: str) -> EvaluationResult:
        expression = (expression or "").strip()
        if not expression:
            return EvaluationResult(expression="", answer="No expression", success=False, error="Empty expression")

        try:
            if "=" in expression:
                left, right = expression.split("=", 1)
                parsed_left = self._parse(left)
                parsed_right = self._parse(right)
                answer = sp.Eq(sp.simplify(parsed_left), sp.simplify(parsed_right))
            else:
                parsed = self._parse(expression)
                answer = sp.simplify(parsed)
            return EvaluationResult(expression=expression, answer=self._format_answer(answer), success=True, confidence=0.95)
        except Exception as exc:
            return EvaluationResult(expression=expression, answer=f"Error: {exc}", success=False, error=str(exc))

    def _parse(self, expression: str) -> sp.Expr:
        return parse_expr(
            expression,
            local_dict=self.local_dict,
            transformations=self.transformations,
            evaluate=True,
        )

    @staticmethod
    def _format_answer(value: sp.Expr) -> str:
        if isinstance(value, sp.Equality):
            return str(value)
        if value.free_symbols:
            return str(value)
        numeric = sp.N(value)
        if numeric.is_Integer:
            return str(int(numeric))
        if numeric.is_Rational:
            return str(value)
        as_float = float(numeric)
        if as_float.is_integer():
            return str(int(as_float))
        return f"{as_float:.8g}"
