"""Replaceable OCR engines for Air Math Solver."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
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


class LocalSimpleOCREngine(BaseOCREngine):
    """Small OpenCV recognizer for clean handwritten digits and simple operators."""

    _CHARS = "0123456789+-*/="

    def __init__(self) -> None:
        self._templates = self._build_templates()

    def recognize(self, image_path: Path) -> OCRResult:
        if not image_path.exists():
            return OCRResult(error=f"OCR image not found: {image_path}")

        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return OCRResult(error=f"Could not read OCR image: {image_path}")

        binary = self._prepare_binary(gray)
        segments = self._segment(binary)
        if not segments:
            return OCRResult(error="No readable handwriting found.")

        tokens: List[str] = []
        confidences: List[float] = []
        debug_segments = []
        for rect in segments:
            x, y, w, h = rect
            roi = binary[y : y + h, x : x + w]
            token, confidence = self._recognize_roi(roi, rect)
            if token:
                tokens.append(token)
                confidences.append(confidence)
                debug_segments.append({"token": token, "confidence": float(confidence), "rect": rect})

        expression = self._format_expression(tokens)
        if not expression:
            return OCRResult(error="No simple expression recognized.")

        confidence = float(np.mean(confidences)) if confidences else 0.0
        validation_error = self._validate_expression(tokens, confidences)
        if validation_error:
            return OCRResult(error=validation_error, confidence=confidence)

        return OCRResult(
            text=expression,
            confidence=max(0.0, min(1.0, confidence)),
            raw_response={"source": "local", "segments": debug_segments},
        )

    @staticmethod
    def _prepare_binary(gray: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        threshold_type = cv2.THRESH_BINARY_INV if float(blurred.mean()) > 127.0 else cv2.THRESH_BINARY
        _, binary = cv2.threshold(blurred, 0, 255, threshold_type + cv2.THRESH_OTSU)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        return binary

    def _segment(self, binary: np.ndarray) -> List[Tuple[int, int, int, int]]:
        count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        image_area = binary.shape[0] * binary.shape[1]
        rects: List[Tuple[int, int, int, int]] = []
        for index in range(1, count):
            x, y, w, h, area = (int(value) for value in stats[index])
            if area < max(24, image_area * 0.0008):
                continue
            if w < 3 or h < 3:
                continue
            rects.append((x, y, w, h))

        if not rects:
            return []

        rects.sort(key=lambda item: item[0])
        merged: List[Tuple[int, int, int, int]] = []
        for rect in rects:
            if not merged:
                merged.append(rect)
                continue

            px, py, pw, ph = merged[-1]
            x, y, w, h = rect
            gap = x - (px + pw)
            stacked_operator = abs((px + pw / 2) - (x + w / 2)) < max(pw, w) * 0.3
            stacked_operator = stacked_operator and gap < 12 and self._x_overlap(merged[-1], rect) > 0.55

            if stacked_operator:
                nx = min(px, x)
                ny = min(py, y)
                nx2 = max(px + pw, x + w)
                ny2 = max(py + ph, y + h)
                merged[-1] = (nx, ny, nx2 - nx, ny2 - ny)
            else:
                merged.append(rect)

        return self._split_wide_digits(binary, self._split_touching_symbols(binary, merged))

    def _split_touching_symbols(
        self, binary: np.ndarray, rects: Sequence[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        output: List[Tuple[int, int, int, int]] = []
        for rect in rects:
            x, y, w, h = rect
            if w < max(70, int(h * 0.85)):
                output.append(rect)
                continue

            roi = binary[y : y + h, x : x + w]
            cols = roi.mean(axis=0) / 255.0
            smooth = np.convolve(cols, np.ones(7) / 7.0, mode="same")
            left = max(10, int(w * 0.22))
            right = min(w - 10, int(w * 0.78))
            if right <= left:
                output.append(rect)
                continue

            valleys = [index for index in range(left, right) if smooth[index] < 0.035]
            if not valleys:
                output.append(rect)
                continue

            split = int(valleys[len(valleys) // 2])
            if split < max(12, w * 0.2) or w - split < max(12, w * 0.2):
                output.append(rect)
                continue
            output.append((x, y, split, h))
            output.append((x + split, y, w - split, h))
        return sorted(output, key=lambda item: item[0])

    def _split_wide_digits(
        self, binary: np.ndarray, rects: Sequence[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        output: List[Tuple[int, int, int, int]] = []
        for rect in rects:
            x, y, w, h = rect
            if w / max(h, 1) < 1.25 or w < 44:
                output.append(rect)
                continue

            roi = binary[y : y + h, x : x + w]
            cols = roi.mean(axis=0) / 255.0
            left = int(w * 0.3)
            right = int(w * 0.7)
            if right <= left:
                output.append(rect)
                continue
            split = left + int(np.argmin(cols[left:right]))
            if cols[split] > 0.08:
                output.append(rect)
                continue
            output.append((x, y, split, h))
            output.append((x + split, y, w - split, h))
        return output

    @staticmethod
    def _x_overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax, _, aw, _ = a
        bx, _, bw, _ = b
        overlap = min(ax + aw, bx + bw) - max(ax, bx)
        return max(0.0, overlap / max(1, min(aw, bw)))

    def _recognize_roi(self, roi: np.ndarray, rect: Tuple[int, int, int, int]) -> Tuple[str, float]:
        heuristic = self._operator_heuristic(roi, rect)
        if heuristic:
            return heuristic

        normalized = self._normalize_binary(roi)
        best_char = ""
        best_score = float("inf")
        best_by_char: Dict[str, float] = {}
        for char, templates in self._templates.items():
            if char not in "0123456789":
                continue
            char_score = float("inf")
            for template in templates:
                pixel_score = np.mean(cv2.absdiff(normalized, template)) / 255.0
                shape_score = self._shape_distance(normalized, template)
                score = pixel_score * 0.65 + shape_score * 0.35
                char_score = min(char_score, float(score))
                if score < best_score:
                    best_score = float(score)
                    best_char = char
            best_by_char[char] = char_score

        original_char = best_char
        if best_char in {"1", "2", "7"} and best_by_char.get("3", 1.0) <= best_score + 0.035:
            if self._looks_like_three(normalized):
                best_char = "3"
                best_score = best_by_char["3"]
        if best_char == original_char and best_char in {"1", "7"} and best_by_char.get("2", 1.0) <= best_score + 0.03:
            if self._looks_like_two(normalized):
                best_char = "2"
                best_score = best_by_char["2"]

        confidence = 1.0 / (1.0 + best_score * 3.2)
        return best_char, confidence

    def _operator_heuristic(self, roi: np.ndarray, rect: Tuple[int, int, int, int]) -> Optional[Tuple[str, float]]:
        _, _, w, h = rect
        aspect = w / max(h, 1)
        density = cv2.countNonZero(roi) / max(1, w * h)
        normalized = self._normalize_binary(roi, size=40)
        rows = normalized.mean(axis=1) / 255.0
        cols = normalized.mean(axis=0) / 255.0
        center_h = rows[16:24].mean()
        center_v = cols[16:24].mean()
        corner_density = (
            normalized[:10, :10].mean()
            + normalized[:10, 30:].mean()
            + normalized[30:, :10].mean()
            + normalized[30:, 30:].mean()
        ) / (4 * 255.0)

        if aspect > 2.6 and h < 42 and density > 0.08:
            return "-", 0.86
        if 0.55 <= aspect <= 1.8 and center_h > 0.18 and center_v > 0.18 and corner_density < 0.16:
            return "+", 0.78
        row_peaks = [index for index, value in enumerate(rows) if value > 0.18]
        if aspect > 1.3 and len(row_peaks) >= 2 and max(row_peaks) - min(row_peaks) > 10 and center_v < 0.18:
            return "=", 0.78
        if 0.45 <= aspect <= 1.8 and density > 0.08 and corner_density > 0.08:
            diagonal_score = self._diagonal_score(normalized)
            if diagonal_score > 0.13 and center_h < 0.34 and center_v < 0.34:
                return "*", 0.72
        if 0.18 <= aspect <= 0.45 and 0.04 <= density <= 0.18 and self._slash_score(normalized) > 0.18:
            return "/", 0.70
        return None

    @staticmethod
    def _format_expression(tokens: Sequence[str]) -> str:
        replacements = {"×": "*", "x": "*", "X": "*", "÷": "/"}
        normalized = [replacements.get(token, token) for token in tokens]
        text = "".join(normalized)
        for operator in "+-*/=":
            text = text.replace(operator, f" {operator} ")
        return " ".join(text.split())

    @staticmethod
    def _diagonal_score(normalized: np.ndarray) -> float:
        image = normalized.astype(np.float32) / 255.0
        main = float(np.mean(np.diag(image)))
        anti = float(np.mean(np.diag(np.fliplr(image))))
        return (main + anti) / 2.0

    @staticmethod
    def _slash_score(normalized: np.ndarray) -> float:
        image = normalized.astype(np.float32) / 255.0
        anti = float(np.mean(np.diag(np.fliplr(image))))
        main = float(np.mean(np.diag(image)))
        return max(0.0, anti - main * 0.5)

    @staticmethod
    def _validate_expression(tokens: Sequence[str], confidences: Sequence[float]) -> str:
        if not tokens:
            return "No simple expression recognized."
        if confidences and float(np.mean(confidences)) < 0.34:
            return "Local OCR confidence is too low. Please redraw more clearly."
        operators = {"+", "-", "*", "/", "="}
        has_operator = any(token in operators for token in tokens)
        if len(tokens) >= 3 and not has_operator and confidences and min(confidences) < 0.46:
            return "Local OCR saw multiple symbols but no operator. Please redraw with clearer spacing."
        return ""

    @staticmethod
    def _looks_like_two(normalized: np.ndarray) -> bool:
        image = normalized.astype(np.float32) / 255.0
        upper_center = image[:16, 16:32].mean()
        upper_right = image[:16, 32:].mean()
        middle_center = image[16:32, 16:32].mean()
        lower_left = image[32:, :16].mean()
        lower_center = image[32:, 16:32].mean()
        lower_right = image[32:, 32:].mean()
        has_top = upper_center + upper_right > 0.16
        has_middle_curve = middle_center > 0.10
        has_lower_sweep = lower_left + lower_center + lower_right > 0.13
        return bool(has_top and has_middle_curve and has_lower_sweep)

    @staticmethod
    def _looks_like_three(normalized: np.ndarray) -> bool:
        image = normalized.astype(np.float32) / 255.0
        upper_center = image[:16, 16:32].mean()
        middle_center = image[16:32, 16:32].mean()
        lower_center = image[32:, 16:32].mean()
        left_column = image[:, :14].mean()
        right_column = image[:, 34:].mean()
        has_three_bands = upper_center > 0.12 and middle_center > 0.26 and lower_center > 0.18
        not_left_heavy = left_column <= right_column + 0.05
        return bool(has_three_bands and not_left_heavy)

    def _build_templates(self) -> Dict[str, List[np.ndarray]]:
        templates: Dict[str, List[np.ndarray]] = {char: [] for char in self._CHARS}
        fonts = (cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_PLAIN)
        for char in self._CHARS:
            for font in fonts:
                for scale in (1.5, 1.8, 2.1):
                    for thickness in (2, 3, 4):
                        canvas = np.zeros((96, 96), dtype=np.uint8)
                        text_size, baseline = cv2.getTextSize(char, font, scale, thickness)
                        tx = max(2, (96 - text_size[0]) // 2)
                        ty = max(text_size[1] + 2, (96 + text_size[1] - baseline) // 2)
                        cv2.putText(canvas, char, (tx, ty), font, scale, 255, thickness, cv2.LINE_AA)
                        templates[char].append(self._normalize_binary(canvas))
        return templates

    @staticmethod
    def _normalize_binary(binary: np.ndarray, size: int = 48) -> np.ndarray:
        coords = cv2.findNonZero(binary)
        if coords is None:
            return np.zeros((size, size), dtype=np.uint8)

        x, y, w, h = cv2.boundingRect(coords)
        crop = binary[y : y + h, x : x + w]
        side = max(w, h)
        pad_x = (side - w) // 2 + max(2, side // 8)
        pad_y = (side - h) // 2 + max(2, side // 8)
        padded = cv2.copyMakeBorder(crop, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=0)
        resized = cv2.resize(padded, (size, size), interpolation=cv2.INTER_AREA)
        _, normalized = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return normalized

    @staticmethod
    def _shape_distance(a: np.ndarray, b: np.ndarray) -> float:
        contours_a, _ = cv2.findContours(a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_b, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours_a or not contours_b:
            return 1.0
        ca = max(contours_a, key=cv2.contourArea)
        cb = max(contours_b, key=cv2.contourArea)
        return min(float(cv2.matchShapes(ca, cb, cv2.CONTOURS_MATCH_I1, 0.0)), 1.0)


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
        payload["source"] = "mathpix"
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


class HybridOCREngine(BaseOCREngine):
    """Use Mathpix when configured, with local simple OCR as a graceful fallback."""

    def __init__(self) -> None:
        self.mathpix = MathpixOCREngine()
        self.local = LocalSimpleOCREngine()

    def is_configured(self) -> bool:
        return self.mathpix.is_configured()

    def recognize(self, image_path: Path) -> OCRResult:
        if not self.mathpix.is_configured():
            return self.local.recognize(image_path)

        mathpix_result = self.mathpix.recognize(image_path)
        if not mathpix_result.error and (mathpix_result.latex or mathpix_result.text):
            return mathpix_result

        local_result = self.local.recognize(image_path)
        if not local_result.error:
            local_result.raw_response = local_result.raw_response or {}
            local_result.raw_response["fallback_from"] = mathpix_result.error or "empty Mathpix response"
            return local_result
        return mathpix_result if mathpix_result.error else local_result
