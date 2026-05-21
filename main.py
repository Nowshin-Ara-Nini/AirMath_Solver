"""Air Math Solver entrypoint.

Real-time webcam app:
webcam -> MediaPipe hand tracking -> air canvas -> OCR -> parser -> SymPy.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from air_canvas import AirCanvas
from buttons import Toolbar
from evaluator import SympyEvaluator
from expression_parser import ExpressionParser
from hand_tracker import HandState, HandTracker
from ocr_engine import HybridOCREngine, OCRResult
from utils import (
    COLOR_ACCENT,
    COLOR_BG,
    COLOR_DRAW,
    COLOR_ERASE,
    COLOR_ERROR,
    COLOR_MUTED,
    COLOR_TEXT,
    COLOR_WARN,
    CONFIG,
    FPSCounter,
    alpha_blend,
    draw_fitted_text,
    draw_text,
    enhance_frame_for_tracking,
    ensure_project_dirs,
    latest_temp_image_path,
)


class AirMathApp:
    """Coordinates camera, UI, gestures, OCR, parsing, and evaluation."""

    def __init__(self) -> None:
        ensure_project_dirs()
        self.canvas = AirCanvas(CONFIG.frame_width, CONFIG.frame_height)
        self.toolbar = Toolbar(CONFIG.frame_width)
        self.hand_tracker = HandTracker()
        self.ocr_engine = HybridOCREngine()
        self.parser = ExpressionParser()
        self.evaluator = SympyEvaluator()
        self.fps_counter = FPSCounter()

        self.mode = "DRAW"
        self.status = "Ready"
        self.raw_expression = "-"
        self.clean_expression = "-"
        self.answer = "Draw an expression"
        self.confidence = 0.0
        self.history: List[Tuple[str, str]] = []
        self.last_snapshot = ""
        self.debug_enabled = CONFIG.debug_enabled
        self.current_fps = 0.0
        self.jump_rejections = 0
        self._last_gesture = "idle"
        self.last_ocr_preview: np.ndarray | None = None
        if not self.ocr_engine.is_configured():
            self.status = "Local OCR ready"
            self.answer = "Draw simple math"

    def run(self) -> None:
        cap = cv2.VideoCapture(CONFIG.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG.frame_height)
        cap.set(cv2.CAP_PROP_FPS, CONFIG.target_fps)

        if not cap.isOpened():
            print("Could not open webcam. Check camera permissions or camera index.")
            return

        cv2.namedWindow(CONFIG.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(CONFIG.window_name, CONFIG.frame_width, CONFIG.frame_height)

        running = True
        try:
            while running:
                ok, frame = cap.read()
                if not ok:
                    self.status = "Camera frame unavailable"
                    continue

                frame = cv2.resize(frame, (CONFIG.frame_width, CONFIG.frame_height))
                frame = cv2.flip(frame, 1)
                display_frame = enhance_frame_for_tracking(frame)
                hand_state = self.hand_tracker.process(display_frame)
                if hand_state.jump_rejected:
                    self.jump_rejections += 1
                pointer = hand_state.smoothed_index_tip if hand_state.found and not hand_state.tracking_lost else None

                action = self.toolbar.update(pointer)
                if action:
                    running = self._handle_action(action)

                self._update_canvas_from_hand(hand_state)
                display = self._render(display_frame, hand_state)
                cv2.imshow(CONFIG.window_name, display)

                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    running = self._handle_key(key)
        finally:
            self.canvas.end_stroke()
            self.hand_tracker.close()
            cap.release()
            cv2.destroyAllWindows()

    def _update_canvas_from_hand(self, hand_state: HandState) -> None:
        if not hand_state.found or hand_state.smoothed_index_tip is None:
            self.canvas.end_stroke()
            self._last_gesture = "lost"
            return
        if hand_state.tracking_lost:
            self.canvas.end_stroke()
            self._last_gesture = hand_state.gesture
            if self.status in {"Drawing", "Erasing"}:
                self.status = "Tracking hand..."
            return

        if hand_state.gesture != self._last_gesture:
            self.canvas.end_stroke()
            self._last_gesture = hand_state.gesture

        point = hand_state.smoothed_index_tip
        hovering_toolbar = self.toolbar.hovered_action(point) is not None
        in_draw_area = CONFIG.toolbar_height < point[1] < CONFIG.frame_height - CONFIG.panel_height
        if hovering_toolbar or not in_draw_area:
            self.canvas.end_stroke()
            return

        should_erase = self.mode == "ERASER" and hand_state.drawing
        should_draw = self.mode == "DRAW" and hand_state.drawing

        if should_erase:
            self.canvas.begin_stroke("erase", size=CONFIG.eraser_size)
            self.canvas.add_point(point)
            self.status = "Erasing"
        elif should_draw:
            self.canvas.begin_stroke("draw", color=COLOR_DRAW, size=CONFIG.brush_size)
            self.canvas.add_point(point)
            self.status = "Drawing"
        else:
            self.canvas.end_stroke()
            if self.status in {"Drawing", "Erasing"}:
                self.status = "Ready"

    def _render(self, frame: np.ndarray, hand_state: HandState) -> np.ndarray:
        display = alpha_blend(frame, self.canvas.render_overlay())
        self._draw_pointer(display, hand_state)
        self._draw_info_panel(display, hand_state)
        toolbar_pointer = hand_state.smoothed_index_tip if hand_state.found and not hand_state.tracking_lost else None
        self.toolbar.render(display, toolbar_pointer, self.mode, self.status)
        self.current_fps = self.fps_counter.tick()
        draw_text(display, f"{self.current_fps:4.0f} FPS", (570, 36), 0.42, COLOR_MUTED, 1)
        if self.debug_enabled:
            self._draw_debug_overlay(display, hand_state)
        return display

    def _draw_pointer(self, frame: np.ndarray, hand_state: HandState) -> None:
        if not hand_state.found or hand_state.smoothed_index_tip is None:
            return
        point = hand_state.smoothed_index_tip
        if hand_state.tracking_lost:
            color = COLOR_MUTED
        else:
            color = COLOR_ERASE if self.mode == "ERASER" else COLOR_ACCENT
        confidence_radius = 11 + int(max(0.0, min(1.0, hand_state.tracking_confidence)) * 7)
        cv2.circle(frame, point, confidence_radius, color, 2, cv2.LINE_AA)
        cv2.circle(frame, point, 5, color, -1, cv2.LINE_AA)

    def _draw_info_panel(self, frame: np.ndarray, hand_state: HandState) -> None:
        top = CONFIG.frame_height - CONFIG.panel_height
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, top), (CONFIG.frame_width, CONFIG.frame_height), COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
        cv2.line(frame, (0, top), (CONFIG.frame_width, top), (68, 78, 90), 1)

        answer_lower = str(self.answer).lower()
        is_problem = (
            str(self.answer).startswith("Error")
            or "setup needed" in answer_lower
            or self.status in {"OCR error", "OCR setup needed", "Parse error", "Math error"}
        )
        result_color = COLOR_ACCENT if not is_problem else COLOR_ERROR

        draw_text(frame, "Expr:", (14, top + 21), 0.42, COLOR_MUTED, 1)
        draw_fitted_text(frame, self.clean_expression, (62, top + 21), 250, 0.48, COLOR_TEXT, 1)
        draw_text(frame, "Ans:", (14, top + 45), 0.42, COLOR_MUTED, 1)
        draw_fitted_text(frame, self.answer, (62, top + 45), 390, 0.5, result_color, 1)
        draw_text(frame, f"Conf {int(self.confidence * 100):02d}%", (500, top + 21), 0.42, COLOR_MUTED, 1)
        status = "Tracking..." if hand_state.tracking_lost else self.status
        draw_fitted_text(frame, status, (500, top + 45), 122, 0.4, COLOR_WARN if is_problem else COLOR_MUTED, 1)
        draw_text(frame, "Keys: C clear | R eraser | Z undo | D debug | E/Space eval | Q quit", (14, top + 66), 0.32, COLOR_WARN, 1)

    def _draw_debug_overlay(self, frame: np.ndarray, hand_state: HandState) -> None:
        lines = [
            f"DEBUG fps={self.current_fps:.0f}",
            f"gesture={hand_state.gesture} conf={hand_state.tracking_confidence:.2f}",
            f"lost={hand_state.lost_frames} jump={hand_state.jump_rejected} total={self.jump_rejections}",
            f"tip={hand_state.smoothed_index_tip}",
        ]
        x, y = 8, CONFIG.toolbar_height + 18
        for index, line in enumerate(lines):
            draw_text(frame, line, (x, y + index * 18), 0.38, COLOR_WARN, 1)

        if self.last_ocr_preview is not None:
            preview = self.last_ocr_preview
            ph, pw = preview.shape[:2]
            px = CONFIG.frame_width - pw - 8
            py = CONFIG.toolbar_height + 8
            frame[py : py + ph, px : px + pw] = preview
            cv2.rectangle(frame, (px, py), (px + pw, py + ph), COLOR_WARN, 1)

    def _handle_action(self, action: str) -> bool:
        if action == "CLEAR":
            self.canvas.clear()
            self.raw_expression = "-"
            self.clean_expression = "-"
            self.answer = "Canvas cleared"
            self.confidence = 0.0
            self.status = "Ready"
        elif action == "UNDO":
            self.canvas.undo()
            self.status = "Undo"
        elif action == "ERASER":
            self.canvas.end_stroke()
            self.mode = "DRAW" if self.mode == "ERASER" else "ERASER"
            self.status = f"{self.mode.title()} mode"
        elif action == "EVALUATE":
            self.evaluate_canvas()
        elif action == "QUIT":
            return False
        return True

    def _handle_key(self, key: int) -> bool:
        if key in (ord("q"), ord("Q"), 27):
            return False
        if key in (ord("c"), ord("C")):
            return self._handle_action("CLEAR")
        if key in (ord("z"), ord("Z")):
            return self._handle_action("UNDO")
        if key in (ord("e"), ord("E"), ord(" ")):
            self.evaluate_canvas()
            return True
        if key in (ord("r"), ord("R")):
            return self._handle_action("ERASER")
        if key in (ord("d"), ord("D")):
            self.debug_enabled = not self.debug_enabled
            self.status = "Debug on" if self.debug_enabled else "Debug off"
            return True
        return True

    def evaluate_canvas(self) -> None:
        if not self.canvas.has_content():
            self.status = "Nothing to evaluate"
            self.answer = "Draw something first"
            self.confidence = 0.0
            return

        self.status = "Saving canvas"
        image_path = latest_temp_image_path()
        if not self.canvas.save_for_ocr(image_path):
            self.status = "Save failed"
            self.answer = "Could not save drawing"
            self.confidence = 0.0
            return

        self.last_snapshot = image_path.name
        self._update_ocr_preview(image_path)
        self.status = "Calling Mathpix" if self.ocr_engine.is_configured() else "Reading locally"
        ocr_result = self.ocr_engine.recognize(image_path)
        self._apply_ocr_result(ocr_result)

    def _update_ocr_preview(self, image_path: object) -> None:
        preview = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if preview is None:
            self.last_ocr_preview = None
            return
        preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
        preview = cv2.resize(preview, (120, 80), interpolation=cv2.INTER_AREA)
        self.last_ocr_preview = preview

    def _apply_ocr_result(self, ocr_result: OCRResult) -> None:
        if ocr_result.error:
            self.status = "OCR error"
            self.raw_expression = "-"
            self.clean_expression = "-"
            self.answer = ocr_result.error
            self.confidence = 0.0
            return

        recognized = ocr_result.latex or ocr_result.text
        parse_result = self.parser.clean(recognized)
        self.raw_expression = recognized or "-"

        if parse_result.error:
            self.status = "Parse error"
            self.clean_expression = "-"
            self.answer = parse_result.error
            self.confidence = ocr_result.confidence
            return

        evaluation = self.evaluator.evaluate(parse_result.expression)
        self.clean_expression = parse_result.expression
        self.answer = evaluation.answer
        self.confidence = max(0.0, min(1.0, ocr_result.confidence * evaluation.confidence))
        source = ""
        if ocr_result.raw_response and ocr_result.raw_response.get("source"):
            source = f" ({ocr_result.raw_response['source']})"
        self.status = f"Ready{source}" if evaluation.success else "Math error"
        if evaluation.success:
            self.history.append((parse_result.expression, evaluation.answer))
            self.history = self.history[-8:]


def main() -> None:
    AirMathApp().run()


if __name__ == "__main__":
    main()
