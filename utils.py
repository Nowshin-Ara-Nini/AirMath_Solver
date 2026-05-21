"""Shared configuration and small helpers for Air Math Solver."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
TEMP_DIR = ROOT_DIR / "temp"
ASSETS_DIR = ROOT_DIR / "assets"
MODELS_DIR = ROOT_DIR / "models"


@dataclass(frozen=True)
class AppConfig:
    window_name: str = "Air Math Solver"
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    target_fps: int = 30
    toolbar_height: int = 64
    panel_height: int = 72
    brush_size: int = 7
    eraser_size: int = 28
    smoothing_alpha: float = 0.30
    click_cooldown_s: float = 0.75
    detection_confidence: float = 0.75
    tracking_confidence: float = 0.75
    jump_threshold_px: float = 55.0
    hold_missing_frames: int = 3
    enhance_low_light: bool = True
    denoise_frame: bool = False
    debug_enabled: bool = False


CONFIG = AppConfig()

FONT = cv2.FONT_HERSHEY_SIMPLEX
SMALL_FONT = cv2.FONT_HERSHEY_PLAIN

COLOR_BG = (18, 20, 24)
COLOR_PANEL = (30, 34, 40)
COLOR_BORDER = (72, 78, 88)
COLOR_TEXT = (238, 241, 245)
COLOR_MUTED = (158, 166, 176)
COLOR_ACCENT = (51, 214, 159)
COLOR_WARN = (64, 183, 255)
COLOR_ERROR = (72, 72, 230)
COLOR_DRAW = (42, 230, 156)
COLOR_ERASE = (72, 72, 235)


def ensure_project_dirs() -> None:
    """Create runtime folders without touching source files."""
    for path in (TEMP_DIR, ASSETS_DIR, MODELS_DIR):
        path.mkdir(exist_ok=True)


class FPSCounter:
    """Rolling FPS counter that avoids noisy one-frame spikes."""

    def __init__(self, window: int = 30) -> None:
        self.samples: deque[float] = deque(maxlen=window)
        self.last_time = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        dt = max(now - self.last_time, 1e-6)
        self.last_time = now
        self.samples.append(dt)
        avg = sum(self.samples) / len(self.samples)
        return 1.0 / max(avg, 1e-6)


def draw_text(
    image: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    scale: float = 0.55,
    color: Tuple[int, int, int] = COLOR_TEXT,
    thickness: int = 1,
) -> None:
    cv2.putText(image, text, origin, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_fitted_text(
    image: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    max_width: int,
    scale: float,
    color: Tuple[int, int, int],
    thickness: int = 1,
    min_scale: float = 0.35,
) -> None:
    text = str(text)
    while scale > min_scale and cv2.getTextSize(text, FONT, scale, thickness)[0][0] > max_width:
        scale -= 0.05
    cv2.putText(image, text, origin, FONT, scale, color, thickness, cv2.LINE_AA)


def wrap_text(text: str, max_chars: int) -> List[str]:
    words = str(text).split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word[:max_chars]
    if current:
        lines.append(current)
    return lines or [""]


def alpha_blend(base_bgr: np.ndarray, overlay_bgra: np.ndarray) -> np.ndarray:
    """Blend a BGRA overlay onto a BGR image."""
    alpha = overlay_bgra[:, :, 3:4].astype(np.float32) / 255.0
    blended = base_bgr.astype(np.float32) * (1.0 - alpha) + overlay_bgra[:, :, :3].astype(np.float32) * alpha
    return blended.astype(np.uint8)


def enhance_frame_for_tracking(frame_bgr: np.ndarray) -> np.ndarray:
    """Apply light CPU-only enhancement for dark/noisy webcam frames."""
    if not CONFIG.enhance_low_light:
        return frame_bgr

    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y = clahe.apply(y)
    enhanced = cv2.cvtColor(cv2.merge((y, cr, cb)), cv2.COLOR_YCrCb2BGR)

    mean_luma = float(y.mean())
    if mean_luma < 105.0:
        beta = 20 if mean_luma < 75.0 else 12
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.06, beta=beta)

    if CONFIG.denoise_frame:
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 4, 4, 7, 15)
    return enhanced


def latest_temp_image_path(prefix: str = "air_math_canvas") -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return TEMP_DIR / f"{prefix}_{timestamp}.png"


def clamp_point(point: Tuple[int, int], width: int, height: int) -> Tuple[int, int]:
    x, y = point
    return max(0, min(width - 1, int(x))), max(0, min(height - 1, int(y)))


def mean_bool(values: Iterable[bool]) -> int:
    return sum(1 for value in values if value)
