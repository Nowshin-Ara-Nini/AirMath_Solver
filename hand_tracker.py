"""CPU-friendly MediaPipe hand tracking and gesture detection."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from utils import CONFIG, mean_bool


INDEX_TIP = 8
INDEX_PIP = 6
MIDDLE_TIP = 12
MIDDLE_PIP = 10
RING_TIP = 16
RING_PIP = 14
PINKY_TIP = 20
PINKY_PIP = 18


@dataclass
class HandState:
    found: bool
    index_tip: Optional[Tuple[int, int]] = None
    smoothed_index_tip: Optional[Tuple[int, int]] = None
    fingers: Optional[Dict[str, bool]] = None
    drawing: bool = False
    erasing: bool = False
    fingers_up: int = 0
    landmarks: object = None
    tracking_lost: bool = False


class HandTracker:
    """Wraps MediaPipe Hands and exposes stable high-level gestures."""

    def __init__(
        self,
        max_num_hands: int = 1,
        detection_confidence: float = 0.6,
        tracking_confidence: float = 0.5,
        smoothing_alpha: float = CONFIG.smoothing_alpha,
    ) -> None:
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=0,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._smooth: Optional[Tuple[float, float]] = None
        self._alpha = smoothing_alpha
        self._gesture_buffer: Deque[str] = deque(maxlen=4)
        self._active_gesture = "idle"
        self._missed_frames = 0
        self._hold_missing_frames = 4

    def process(self, frame_bgr: np.ndarray) -> HandState:
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._hands.process(rgb)

        if not result.multi_hand_landmarks:
            self._missed_frames += 1
            self._gesture_buffer.clear()
            self._active_gesture = "idle"
            if self._smooth is not None and self._missed_frames <= self._hold_missing_frames:
                held_point = (int(self._smooth[0]), int(self._smooth[1]))
                return HandState(
                    found=True,
                    index_tip=held_point,
                    smoothed_index_tip=held_point,
                    fingers={},
                    drawing=False,
                    erasing=False,
                    fingers_up=0,
                    tracking_lost=True,
                )
            self._smooth = None
            return HandState(found=False)

        self._missed_frames = 0
        landmarks = result.multi_hand_landmarks[0]
        index = landmarks.landmark[INDEX_TIP]
        raw_point = (int(index.x * width), int(index.y * height))
        smooth_point = self._smooth_point(raw_point)
        fingers = self._finger_states(landmarks)
        fingers_up = mean_bool(fingers.values())

        raw_gesture = self._classify_gesture(fingers)
        active_gesture = self._stable_gesture(raw_gesture)
        drawing = active_gesture == "draw"
        erasing = active_gesture == "erase"

        return HandState(
            found=True,
            index_tip=raw_point,
            smoothed_index_tip=smooth_point,
            fingers=fingers,
            drawing=drawing,
            erasing=erasing,
            fingers_up=fingers_up,
            landmarks=landmarks,
        )

    def draw_landmarks(self, frame_bgr: np.ndarray, hand_state: HandState) -> None:
        if not hand_state.found or hand_state.landmarks is None:
            return
        mp.solutions.drawing_utils.draw_landmarks(
            frame_bgr,
            hand_state.landmarks,
            self._mp_hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style(),
        )

    def close(self) -> None:
        self._hands.close()

    def _smooth_point(self, point: Tuple[int, int]) -> Tuple[int, int]:
        x, y = point
        if self._smooth is None:
            self._smooth = (float(x), float(y))
        else:
            sx, sy = self._smooth
            self._smooth = (
                sx * (1.0 - self._alpha) + x * self._alpha,
                sy * (1.0 - self._alpha) + y * self._alpha,
            )
        return int(self._smooth[0]), int(self._smooth[1])

    @staticmethod
    def _is_up(landmarks: object, tip_id: int, pip_id: int) -> bool:
        tip = landmarks.landmark[tip_id]
        pip = landmarks.landmark[pip_id]
        return tip.y < pip.y - 0.015

    @staticmethod
    def _classify_gesture(fingers: Dict[str, bool]) -> str:
        if fingers["index"] and fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
            return "erase"
        if fingers["index"] and not fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
            return "draw"
        return "idle"

    def _stable_gesture(self, raw_gesture: str) -> str:
        self._gesture_buffer.append(raw_gesture)
        counts = Counter(self._gesture_buffer)
        required = 3 if len(self._gesture_buffer) >= 4 else 2
        candidate, count = counts.most_common(1)[0]

        if count >= required:
            self._active_gesture = candidate
            return self._active_gesture

        if raw_gesture != self._active_gesture:
            return "idle"
        return self._active_gesture

    def _finger_states(self, landmarks: object) -> Dict[str, bool]:
        return {
            "index": self._is_up(landmarks, INDEX_TIP, INDEX_PIP),
            "middle": self._is_up(landmarks, MIDDLE_TIP, MIDDLE_PIP),
            "ring": self._is_up(landmarks, RING_TIP, RING_PIP),
            "pinky": self._is_up(landmarks, PINKY_TIP, PINKY_PIP),
        }
