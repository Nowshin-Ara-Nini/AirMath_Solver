"""Replaceable OCR engines for Air Math Solver.

The default implementation calls Mathpix. No local deep-learning OCR is used.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv


@dataclass
class OCRResult:
    text: str = ""
    latex: str = ""
    confidence: float = 0.0
    raw_response: Optional[Dict[str, Any]] = None
    error: str = ""


class BaseOCREngine(ABC):
    @abstractmethod
    def recognize(self, image_path: Path) -> OCRResult:
        """Recognize math from an image path."""


class MathpixOCREngine(BaseOCREngine):
    """Mathpix image OCR backend using credentials from `.env`."""

    def __init__(self) -> None:
        load_dotenv()
        self.app_id = os.getenv("MATHPIX_APP_ID", "").strip()
        self.app_key = os.getenv("MATHPIX_APP_KEY", "").strip()
        self.api_url = os.getenv("MATHPIX_API_URL", "https://api.mathpix.com/v3/text").strip()
        self.timeout = float(os.getenv("MATHPIX_TIMEOUT", "15"))

    def is_configured(self) -> bool:
        return bool(self.app_id and self.app_key and not self._using_placeholder_credentials())

    def recognize(self, image_path: Path) -> OCRResult:
        if not self.app_id or not self.app_key:
            return OCRResult(error="OCR setup needed: create .env and add Mathpix keys.")
        if self._using_placeholder_credentials():
            return OCRResult(error="OCR setup needed: replace placeholder Mathpix keys in .env.")
        if not image_path.exists():
            return OCRResult(error=f"OCR image not found: {image_path}")

        options = {
            "math_inline_delimiters": ["$", "$"],
            "rm_spaces": True,
            "formats": ["text", "data"],
            "data_options": {"include_asciimath": True, "include_latex": True},
        }
        headers = {"app_id": self.app_id, "app_key": self.app_key}

        try:
            with image_path.open("rb") as image_file:
                response = requests.post(
                    self.api_url,
                    files={"file": image_file},
                    data={"options_json": json.dumps(options)},
                    headers=headers,
                    timeout=self.timeout,
                )
            payload = response.json()
        except requests.RequestException as exc:
            return OCRResult(error=f"Mathpix request failed: {exc}")
        except ValueError:
            return OCRResult(error="Mathpix returned a non-JSON response.")

        if response.status_code >= 400 or payload.get("error"):
            message = payload.get("error") or payload.get("message") or f"HTTP {response.status_code}"
            return OCRResult(raw_response=payload, error=f"Mathpix error: {message}")

        latex = self._extract_latex(payload)
        text = payload.get("text") or latex
        confidence = float(payload.get("confidence_rate") or payload.get("confidence") or 0.0)
        return OCRResult(text=text or "", latex=latex or "", confidence=confidence, raw_response=payload)

    def _using_placeholder_credentials(self) -> bool:
        placeholders = {"your_app_id_here", "your_app_key_here", "app_id", "app_key"}
        return self.app_id.lower() in placeholders or self.app_key.lower() in placeholders

    @staticmethod
    def _extract_latex(payload: Dict[str, Any]) -> str:
        data = payload.get("data") or []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get("type") == "latex":
                    value = item.get("value")
                    if value:
                        return str(value)
        text = payload.get("text", "")
        return str(text)
