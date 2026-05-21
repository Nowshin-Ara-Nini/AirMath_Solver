from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from air_canvas import AirCanvas
from buttons import Toolbar
from evaluator import SympyEvaluator
from expression_parser import ExpressionParser
from hand_tracker import HandTracker
from ocr_engine import LocalSimpleOCREngine


class ParserEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = ExpressionParser()
        self.evaluator = SympyEvaluator()

    def assert_evaluates_to(self, raw: str, expected: str) -> None:
        parsed = self.parser.clean(raw)
        self.assertFalse(parsed.error)
        result = self.evaluator.evaluate(parsed.expression)
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.answer, expected)

    def test_basic_arithmetic(self) -> None:
        cases = {
            "22": "22",
            "2+3": "5",
            "2+2": "4",
            "12-5": "7",
            "3*7": "21",
            "2×3": "6",
            "2x3": "6",
            "6/2": "3",
            "6÷2": "3",
            "2*3": "6",
            "sqrt(25)": "5",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assert_evaluates_to(raw, expected)

    def test_equality_expression(self) -> None:
        parsed = self.parser.clean("2=2")
        self.assertFalse(parsed.error)
        result = self.evaluator.evaluate(parsed.expression)
        self.assertTrue(result.success, result.error)
        self.assertIn(result.answer, {"True", "1"})

    def test_empty_and_malformed_inputs_fail_cleanly(self) -> None:
        self.assertTrue(self.parser.clean("").error)
        result = self.evaluator.evaluate("2+")
        self.assertFalse(result.success)
        self.assertTrue(result.answer.startswith("Error"))


class LocalOCRTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ocr = LocalSimpleOCREngine()

    def test_local_ocr_reads_synthetic_22(self) -> None:
        image = np.full((160, 260, 3), 255, np.uint8)
        cv2.putText(image, "22", (30, 115), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 0), 8, cv2.LINE_AA)
        path = Path(tempfile.gettempdir()) / "airmath_test_22.png"
        cv2.imwrite(str(path), image)

        result = self.ocr.recognize(path)

        self.assertFalse(result.error)
        self.assertEqual(result.text.replace(" ", ""), "22")

    def test_local_ocr_reads_black_background_export_style(self) -> None:
        image = np.zeros((160, 360, 3), np.uint8)
        cv2.putText(image, "2+3", (30, 115), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 8, cv2.LINE_AA)
        path = Path(tempfile.gettempdir()) / "airmath_test_black_2_plus_3.png"
        cv2.imwrite(str(path), image)

        result = self.ocr.recognize(path)

        self.assertFalse(result.error)
        self.assertEqual(result.text.replace(" ", ""), "2+3")

    def test_local_ocr_reads_saved_two_crop_when_available(self) -> None:
        path = Path("temp") / "air_math_canvas_20260521_174313.png"
        if not path.exists():
            self.skipTest("Saved regression crop is not present.")

        result = self.ocr.recognize(path)

        self.assertFalse(result.error)
        self.assertIn(result.text.replace(" ", ""), {"2", "22"})

    def test_local_ocr_reads_saved_two_plus_three_crop_when_available(self) -> None:
        path = Path("temp") / "air_math_canvas_20260521_180458.png"
        if not path.exists():
            self.skipTest("Saved 2+3 regression crop is not present.")

        result = self.ocr.recognize(path)

        self.assertFalse(result.error)
        self.assertEqual(result.text.replace(" ", ""), "2+3")

        parsed = ExpressionParser().clean(result.text)
        evaluation = SympyEvaluator().evaluate(parsed.expression)
        self.assertTrue(evaluation.success, evaluation.error)
        self.assertEqual(evaluation.answer, "5")

    def test_local_ocr_keeps_visible_minus_when_available(self) -> None:
        path = Path("temp") / "air_math_canvas_20260521_180246.png"
        if not path.exists():
            self.skipTest("Saved 2-3 regression crop is not present.")

        result = self.ocr.recognize(path)

        self.assertFalse(result.error)
        self.assertEqual(result.text.replace(" ", ""), "2-3")

    def test_local_ocr_rejects_suspicious_low_confidence_expression(self) -> None:
        error = LocalSimpleOCREngine._validate_expression(["2", "3", "4"], [0.30, 0.32, 0.33])
        self.assertTrue(error)


class TrackerStabilityTests(unittest.TestCase):
    def test_jump_rejection_uses_configured_threshold(self) -> None:
        tracker = object.__new__(HandTracker)
        tracker._smooth = (100.0, 100.0)
        tracker._jump_threshold_px = 55.0

        self.assertFalse(tracker._is_jump((130, 125)))
        self.assertTrue(tracker._is_jump((180, 100)))

    def test_exponential_smoothing_keeps_motion_stable(self) -> None:
        tracker = object.__new__(HandTracker)
        tracker._smooth = (100.0, 100.0)
        tracker._alpha = 0.30

        self.assertEqual(tracker._smooth_point((130, 100)), (109, 100))


class CanvasExportTests(unittest.TestCase):
    def test_ocr_export_is_black_background_with_white_strokes(self) -> None:
        canvas = AirCanvas(180, 120)
        canvas.begin_stroke("draw", size=7)
        for point in [(20, 50), (40, 50), (60, 50), (80, 50)]:
            canvas.add_point(point)

        path = Path(tempfile.gettempdir()) / "airmath_canvas_export_test.png"
        self.assertTrue(canvas.save_for_ocr(path))

        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(gray)
        self.assertLess(float(gray.mean()), 80.0)
        self.assertEqual(int(gray.max()), 255)
        self.assertEqual(int(gray.min()), 0)

    def test_canvas_does_not_connect_large_jumps(self) -> None:
        canvas = AirCanvas(180, 120)
        canvas.begin_stroke("draw", size=7)
        canvas.add_point((20, 60))
        canvas.add_point((40, 60))
        canvas.add_point((140, 60))
        canvas.add_point((160, 60))
        canvas.end_stroke()

        image = canvas._render_ocr_image()
        self.assertEqual(int(image[60, 90, 0]), 0)


class ToolbarTests(unittest.TestCase):
    def test_toolbar_contains_undo_action(self) -> None:
        toolbar = Toolbar(640)
        actions = [button.action for button in toolbar.buttons]
        labels = {button.action: button.label for button in toolbar.buttons}

        self.assertIn("UNDO", actions)
        self.assertEqual(labels["UNDO"], "UNDO Z")


if __name__ == "__main__":
    unittest.main()
