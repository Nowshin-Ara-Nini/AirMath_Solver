"""OpenCV toolbar buttons with fingertip collision and debounce."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils import (
    COLOR_ACCENT,
    COLOR_BORDER,
    COLOR_DRAW,
    COLOR_ERASE,
    COLOR_MUTED,
    COLOR_PANEL,
    COLOR_TEXT,
    CONFIG,
    FONT,
    draw_text,
)


@dataclass
class Button:
    action: str
    label: str
    rect: Tuple[int, int, int, int]

    def contains(self, point: Optional[Tuple[int, int]]) -> bool:
        if point is None:
            return False
        x, y = point
        rx, ry, rw, rh = self.rect
        return rx <= x <= rx + rw and ry <= y <= ry + rh


class Toolbar:
    """Top toolbar used by both mouse-free fingertip clicks and keyboard shortcuts."""

    def __init__(
        self,
        width: int,
        height: int = CONFIG.toolbar_height,
        cooldown_s: float = CONFIG.click_cooldown_s,
        dwell_s: float = 0.42,
    ) -> None:
        self.width = width
        self.height = height
        self.cooldown_s = cooldown_s
        self.dwell_s = dwell_s
        self.buttons = self._layout_buttons()
        self._last_click: Dict[str, float] = {}
        self._hover_action: Optional[str] = None
        self._hover_started = 0.0

    def hovered_action(self, point: Optional[Tuple[int, int]]) -> Optional[str]:
        for button in self.buttons:
            if button.contains(point):
                return button.action
        return None

    def update(self, point: Optional[Tuple[int, int]]) -> Optional[str]:
        action = self.hovered_action(point)
        if action is None:
            self._hover_action = None
            self._hover_started = 0.0
            return None
        now = time.perf_counter()
        if action != self._hover_action:
            self._hover_action = action
            self._hover_started = now
            return None
        if now - self._hover_started < self.dwell_s:
            return None
        last = self._last_click.get(action, 0.0)
        if now - last < self.cooldown_s:
            return None
        self._last_click[action] = now
        self._hover_started = now
        return action

    def render(self, frame: np.ndarray, pointer: Optional[Tuple[int, int]], mode: str, status: str) -> None:
        cv2.rectangle(frame, (0, 0), (self.width, self.height), COLOR_PANEL, -1)
        cv2.line(frame, (0, self.height - 1), (self.width, self.height - 1), COLOR_BORDER, 1)
        hover = self.hovered_action(pointer)
        for button in self.buttons:
            self._draw_button(frame, button, hover == button.action, mode)

        mode_color = COLOR_ERASE if mode == "ERASER" else COLOR_DRAW
        cv2.circle(frame, (490, 31), 7, mode_color, -1, cv2.LINE_AA)
        draw_text(frame, f"MODE: {mode}", (504, 36), 0.42, COLOR_TEXT, 1)
        draw_text(frame, status[:24], (504, 55), 0.35, COLOR_MUTED, 1)

    def _layout_buttons(self) -> List[Button]:
        labels = [
            ("CLEAR", "CLEAR C", 76),
            ("UNDO", "UNDO Z", 78),
            ("ERASER", "ERASER R", 88),
            ("EVALUATE", "EVAL E/SPACE", 116),
            ("QUIT", "QUIT Q", 72),
        ]
        x = 12
        buttons: List[Button] = []
        for action, label, width in labels:
            buttons.append(Button(action=action, label=label, rect=(x, 14, width, 36)))
            x += width + 8
        return buttons

    def _draw_button(self, frame: np.ndarray, button: Button, hovered: bool, mode: str) -> None:
        x, y, w, h = button.rect
        active = button.action == "ERASER" and mode == "ERASER"
        fill = (54, 60, 70) if not hovered else (68, 76, 88)
        border = COLOR_ERASE if active else (COLOR_ACCENT if hovered else COLOR_BORDER)
        cv2.rectangle(frame, (x, y), (x + w, y + h), fill, -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), border, 2 if active or hovered else 1)
        if hovered and self._hover_action == button.action and self._hover_started:
            elapsed = min(1.0, (time.perf_counter() - self._hover_started) / max(self.dwell_s, 0.01))
            cv2.rectangle(frame, (x, y + h - 4), (x + int(w * elapsed), y + h), COLOR_ACCENT, -1)
        text_size = cv2.getTextSize(button.label, FONT, 0.5, 1)[0]
        tx = x + max(4, (w - text_size[0]) // 2)
        ty = y + (h + text_size[1]) // 2
        draw_text(frame, button.label, (tx, ty), 0.5, COLOR_TEXT, 1)
