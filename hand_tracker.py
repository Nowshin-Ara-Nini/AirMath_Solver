"""CPU-friendly MediaPipe hand tracking and gesture detection."""

from __future__ import annotations

from collections import deque
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
INDEX_MCP = 5
MIDDLE_MCP = 9
RING_MCP = 13
PINKY_MCP = 17


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
    tracking_confidence: float = 0.0
    gesture: str = "idle"
    jump_rejected: bool = False
    lost_frames: int = 0


class HandTracker:
    """Wraps MediaPipe Hands and exposes stable high-level gestures."""

    def __init__(
        self,
        max_num_hands: int = 1,
        detection_confidence: float = CONFIG.detection_confidence,
        tracking_confidence: float = CONFIG.tracking_confidence,
        smoothing_alpha: float = CONFIG.smoothing_alpha,
        jump_threshold_px: float = CONFIG.jump_threshold_px,
    ) -> None:
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=1,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._smooth: Optional[Tuple[float, float]] = None
        self._alpha = smoothing_alpha
        self._jump_threshold_px = jump_threshold_px
        self._gesture_buffer: Deque[str] = deque(maxlen=6)
        self._active_gesture = "idle"
        self._missed_frames = 0
        self._jump_frames = 0
        self._hold_missing_frames = CONFIG.hold_missing_frames

    def process(self, frame_bgr: np.ndarray) -> HandState:
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._hands.process(rgb)

        if not result.multi_hand_landmarks:
            self._missed_frames += 1
            self._jump_frames = 0
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
                    tracking_confidence=0.0,
                    gesture="lost",
                    lost_frames=self._missed_frames,
                )
            self._smooth = None
            return HandState(found=False, tracking_lost=True, gesture="lost", lost_frames=self._missed_frames)

        self._missed_frames = 0
        landmarks = result.multi_hand_landmarks[0]
        tracking_confidence = self._tracking_confidence(result)
        index = landmarks.landmark[INDEX_TIP]
        raw_point = (int(index.x * width), int(index.y * height))

        if self._is_jump(raw_point):
            self._jump_frames += 1
            self._gesture_buffer.clear()
            self._active_gesture = "idle"
            held_point = self.smoothed_point
            if self._jump_frames >= 2:
                self._smooth = (float(raw_point[0]), float(raw_point[1]))
                held_point = raw_point
                self._jump_frames = 0
            return HandState(
                found=True,
                index_tip=raw_point,
                smoothed_index_tip=held_point,
                fingers={},
                drawing=False,
                erasing=False,
                fingers_up=0,
                landmarks=landmarks,
                tracking_lost=True,
                tracking_confidence=tracking_confidence,
                gesture="jump",
                jump_rejected=True,
                lost_frames=max(1, self._jump_frames),
            )

        self._jump_frames = 0
        smooth_point = self._smooth_point(raw_point)
        fingers = self._finger_states(landmarks)
        fingers_up = mean_bool(fingers.values())

        raw_gesture = self._classify_gesture(fingers)
        active_gesture = self._stable_gesture(raw_gesture)
        drawing = active_gesture == "draw"
        erasing = False

        return HandState(
            found=True,
            index_tip=raw_point,
            smoothed_index_tip=smooth_point,
            fingers=fingers,
            drawing=drawing,
            erasing=erasing,
            fingers_up=fingers_up,
            landmarks=landmarks,
            tracking_confidence=tracking_confidence,
            gesture=active_gesture,
            lost_frames=0,
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

    @property
    def smoothed_point(self) -> Optional[Tuple[int, int]]:
        if self._smooth is None:
            return None
        return int(self._smooth[0]), int(self._smooth[1])

    def _is_jump(self, point: Tuple[int, int]) -> bool:
        if self._smooth is None:
            return False
        sx, sy = self._smooth
        return float(np.hypot(point[0] - sx, point[1] - sy)) > self._jump_threshold_px

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
    def _tracking_confidence(result: object) -> float:
        try:
            handedness = result.multi_handedness or []
            if handedness and handedness[0].classification:
                return float(handedness[0].classification[0].score)
        except (AttributeError, IndexError, TypeError, ValueError):
            pass
        return 1.0

    @staticmethod
    def _is_up(landmarks: object, tip_id: int, pip_id: int, mcp_id: int, margin: float = 0.012) -> bool:
        tip = landmarks.landmark[tip_id]
        pip = landmarks.landmark[pip_id]
        mcp = landmarks.landmark[mcp_id]
        return tip.y < min(pip.y - margin, mcp.y - margin * 0.5)

    @staticmethod
    def _classify_gesture(fingers: Dict[str, bool]) -> str:
        if fingers["index"] and not fingers["ring"] and not fingers["pinky"]:
            return "draw"
        return "idle"

    def _stable_gesture(self, raw_gesture: str) -> str:
        self._gesture_buffer.append(raw_gesture)
        draw_votes = sum(1 for gesture in self._gesture_buffer if gesture == "draw")
        total = len(self._gesture_buffer)
        start_required = 3 if total >= 4 else 2
        stop_required = 4 if total >= 5 else 2

        if self._active_gesture == "draw":
            if total - draw_votes >= stop_required:
                self._active_gesture = "idle"
        elif draw_votes >= start_required:
            self._active_gesture = "draw"
        return self._active_gesture

    def _finger_states(self, landmarks: object) -> Dict[str, bool]:
        return {
            "index": self._is_up(landmarks, INDEX_TIP, INDEX_PIP, INDEX_MCP, margin=0.006),
            "middle": self._is_up(landmarks, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
            "ring": self._is_up(landmarks, RING_TIP, RING_PIP, RING_MCP),
            "pinky": self._is_up(landmarks, PINKY_TIP, PINKY_PIP, PINKY_MCP),
        }
