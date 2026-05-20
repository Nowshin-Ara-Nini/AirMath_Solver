"""OCR text and LaTeX cleanup for SymPy-compatible expressions."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ParseResult:
    raw: str
    expression: str
    error: str = ""


class ExpressionParser:
    """Converts common OCR/LaTeX math output into a Python/SymPy expression."""

    _FUNCTIONS = ("sin", "cos", "tan", "log", "sqrt", "exp", "Abs")

    def clean(self, text: str) -> ParseResult:
        raw = text or ""
        expr = raw.strip()
        if not expr:
            return ParseResult(raw=raw, expression="", error="No expression recognized.")

        expr = self._strip_math_delimiters(expr)
        expr = self._latex_to_plain(expr)
        expr = self._normalize_symbols(expr)
        expr = self._normalize_functions(expr)
        expr = self._insert_implicit_multiplication(expr)
        expr = self._final_cleanup(expr)

        if not expr:
            return ParseResult(raw=raw, expression="", error="Expression became empty after cleanup.")
        return ParseResult(raw=raw, expression=expr)

    @staticmethod
    def _strip_math_delimiters(expr: str) -> str:
        expr = expr.replace("\\(", "").replace("\\)", "")
        expr = expr.replace("\\[", "").replace("\\]", "")
        expr = expr.strip("$ ")
        return expr

    def _latex_to_plain(self, expr: str) -> str:
        expr = expr.replace("\\left", "").replace("\\right", "")
        expr = expr.replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/")
        expr = expr.replace("\\pi", "pi")
        expr = re.sub(r"\\(sin|cos|tan|ln|log|sqrt|exp)\b", r"\1", expr)

        previous = None
        while previous != expr:
            previous = expr
            expr = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"((\1)/(\2))", expr)
            expr = re.sub(r"sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", expr)
            expr = re.sub(r"\^\s*\{([^{}]+)\}", r"**(\1)", expr)
        expr = expr.replace("{", "(").replace("}", ")")
        return expr

    @staticmethod
    def _normalize_symbols(expr: str) -> str:
        replacements = {
            "\u2212": "-",
            "\u2013": "-",
            "\u2014": "-",
            "\u00d7": "*",
            "\u00f7": "/",
            "\u221a": "sqrt",
            "\u00b2": "**2",
            "\u00b3": "**3",
            "\u201c": "",
            "\u201d": "",
            "\u2019": "'",
        }
        for old, new in replacements.items():
            expr = expr.replace(old, new)
        expr = expr.replace("^", "**")
        expr = re.sub(r"\bln\b", "log", expr)
        expr = re.sub(r"\s+", "", expr)
        return expr

    def _normalize_functions(self, expr: str) -> str:
        for name in ("sin", "cos", "tan", "log", "sqrt", "exp"):
            expr = re.sub(rf"\b{name}([A-Za-z]\w*)", rf"{name}(\1)", expr)
            expr = re.sub(rf"\b{name}(\d+(?:\.\d+)?)", rf"{name}(\1)", expr)
        return expr

    def _insert_implicit_multiplication(self, expr: str) -> str:
        function_pattern = "|".join(self._FUNCTIONS)
        expr = re.sub(r"(\d)([A-Za-z])", r"\1*\2", expr)
        expr = re.sub(r"(\d|\))(\()", r"\1*\2", expr)
        expr = re.sub(r"(\))(\d|[A-Za-z])", r"\1*\2", expr)
        expr = re.sub(rf"(\d)\*({function_pattern})\(", r"\1*\2(", expr)
        return expr

    @staticmethod
    def _final_cleanup(expr: str) -> str:
        expr = re.sub(r"[^0-9A-Za-z_+\-*/().=,]", "", expr)
        expr = re.sub(r"([+\-*/])+$", "", expr)
        expr = expr.replace("**(", "**(")
        return expr.strip()
