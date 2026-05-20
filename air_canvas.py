"""Air drawing canvas with stroke history, undo, erasing, and OCR export."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from utils import COLOR_DRAW, CONFIG, clamp_point


@dataclass
class Stroke:
    points: List[Tuple[int, int]] = field(default_factory=list)
    color: Tuple[int, int, int] = COLOR_DRAW
    size: int = CONFIG.brush_size
    mode: str = "draw"


class AirCanvas:
    """Stores vector strokes and renders them into display/OCR images."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.strokes: List[Stroke] = []
        self.current: Optional[Stroke] = None
        self._last_point: Optional[Tuple[int, int]] = None
        self._overlay = np.zeros((height, width, 4), dtype=np.uint8)

    def begin_stroke(self, mode: str, color: Tuple[int, int, int] = COLOR_DRAW, size: int = CONFIG.brush_size) -> None:
        if self.current and self.current.mode == mode:
            return
        self.end_stroke()
        stroke_size = CONFIG.eraser_size if mode == "erase" else size
        self.current = Stroke(color=color, size=stroke_size, mode=mode)
        self._last_point = None

    def add_point(self, point: Tuple[int, int]) -> None:
        if self.current is None:
            return
        point = clamp_point(point, self.width, self.height)
        if self._last_point is None:
            self.current.points.append(point)
            self._last_point = point
            return

        lx, ly = self._last_point
        x, y = point
        distance = math.hypot(x - lx, y - ly)
        steps = max(1, int(distance / 4))
        for step in range(1, steps + 1):
            t = step / steps
            px = int(lx + (x - lx) * t)
            py = int(ly + (y - ly) * t)
            self.current.points.append((px, py))
        self._last_point = point

    def end_stroke(self) -> None:
        if self.current and len(self.current.points) > 1:
            self.strokes.append(self.current)
        self.current = None
        self._last_point = None

    def undo(self) -> None:
        self.end_stroke()
        if self.strokes:
            self.strokes.pop()

    def clear(self) -> None:
        self.strokes.clear()
        self.current = None
        self._last_point = None

    def has_content(self) -> bool:
        return any(stroke.mode == "draw" and len(stroke.points) > 1 for stroke in self.strokes) or (
            self.current is not None and self.current.mode == "draw" and len(self.current.points) > 1
        )

    def render_overlay(self) -> np.ndarray:
        self._overlay[:] = 0
        for stroke in self.strokes:
            self._paint_overlay(stroke)
        if self.current is not None:
            self._paint_overlay(self.current)
        return self._overlay

    def render_blackboard(self) -> np.ndarray:
        board = np.full((self.height, self.width, 3), (16, 28, 24), dtype=np.uint8)
        for stroke in self.strokes:
            self._paint_bgr(board, stroke)
        if self.current is not None:
            self._paint_bgr(board, self.current)
        return board

    def save_for_ocr(self, path: Path) -> bool:
        self.end_stroke()
        image = self._render_ocr_image()
        mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) < 245
        coords = cv2.findNonZero(mask.astype(np.uint8))
        if coords is None:
            return False

        x, y, w, h = cv2.boundingRect(coords)
        pad = 30
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(self.width, x + w + pad)
        y2 = min(self.height, y + h + pad)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return False
        path.parent.mkdir(exist_ok=True)
        return bool(cv2.imwrite(str(path), crop))

    def _render_ocr_image(self) -> np.ndarray:
        image = np.full((self.height, self.width, 3), 255, dtype=np.uint8)
        for stroke in self.strokes:
            color = (255, 255, 255) if stroke.mode == "erase" else (0, 0, 0)
            self._draw_polyline(image, stroke.points, color, stroke.size)
        return image

    def _paint_overlay(self, stroke: Stroke) -> None:
        if stroke.mode == "erase":
            self._draw_polyline(self._overlay, stroke.points, (0, 0, 0, 0), stroke.size)
        else:
            self._draw_polyline(self._overlay, stroke.points, (*stroke.color, 235), stroke.size)

    def _paint_bgr(self, image: np.ndarray, stroke: Stroke) -> None:
        color = (16, 28, 24) if stroke.mode == "erase" else stroke.color
        self._draw_polyline(image, stroke.points, color, stroke.size)

    @staticmethod
    def _draw_polyline(image: np.ndarray, points: List[Tuple[int, int]], color: Tuple[int, ...], size: int) -> None:
        if not points:
            return
        if len(points) == 1:
            cv2.circle(image, points[0], max(1, size // 2), color, -1, cv2.LINE_AA)
            return
        for index in range(1, len(points)):
            cv2.line(image, points[index - 1], points[index], color, size, cv2.LINE_AA)
        cv2.circle(image, points[0], max(1, size // 2), color, -1, cv2.LINE_AA)
        cv2.circle(image, points[-1], max(1, size // 2), color, -1, cv2.LINE_AA)
