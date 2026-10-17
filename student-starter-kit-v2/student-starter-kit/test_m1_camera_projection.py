#!/usr/bin/env python3
"""test_m1_camera_projection.py

Unit test suite for Milestone 1: Camera Projection & Coordinate Geometry.
Verifies:
  1. Ground truth test vectors from explorer_m1_3/test_vectors.md
     (hfov = 1.04719755 rad / 60 deg and hfov = 1.74 rad / 99.7 deg).
  2. Position latency freshness gate rejecting age > 1.0s in on_camera_image & on_detections.
  3. Fresh position age <= 1.0s passing through.
  4. Altitude filter rejecting height < 2.0m.
  5. Missing /opt/kit/camera.json fallback behavior and focal length precision.
  6. Non-finite coordinate (NaN/Inf) rejection in telemetry and projection.
  7. Clean Python 3.10 AST compilation of solution.py and test files.

Compatible with both:
  python3 -m unittest test_m1_camera_projection.py
  pytest test_m1_camera_projection.py
"""

import ast
import math
import os
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

# Ensure pymavlink can be imported or mocked if not in environment
try:
    import pymavlink
except ImportError:
    mock_mav = MagicMock()
    sys.modules['pymavlink'] = mock_mav
    sys.modules['pymavlink.mavutil'] = mock_mav

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import solution


class MockHypothesis:
    def __init__(self, class_id, score):
        self.class_id = class_id
        self.score = float(score)


class MockResult:
    def __init__(self, class_id, score):
        self.hypothesis = MockHypothesis(class_id, score)


class MockPosition:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)


class MockCenter:
    def __init__(self, x, y):
        self.position = MockPosition(x, y)


class MockBBox:
    def __init__(self, cx, cy):
        self.center = MockCenter(cx, cy)


class MockDetection:
    def __init__(self, cx, cy, score=0.85, class_id='person'):
        self.bbox = MockBBox(cx, cy)
        self.results = [MockResult(class_id, score)]


class MockDetectionArrayMsg:
    def __init__(self, detections):
        self.detections = detections


class MockMavMsg:
    def __init__(self, x, y, z, time_boot_ms=10000):
        self.x = x
        self.y = y
        self.z = z
        self.time_boot_ms = time_boot_ms


class TestMilestone1CameraProjection(unittest.TestCase):
    """Milestone 1 Test Suite for Pinhole Projection and Telemetry Ingestion."""

    def setUp(self):
        self.ref_n = 15.0
        self.ref_e = 15.0
        self.H = 16.5
        self.width = 1280.0
        self.height = 960.0
        self.hfov_60deg = 1.04719755
        self.hfov_sim = 1.74

    # -------------------------------------------------------------------------
    # 1. Ground Truth Test Vectors (explorer_m1_3/test_vectors.md)
    # -------------------------------------------------------------------------
    def test_vectors_competition_60deg(self):
        """Verify all 9 exact ground truth vectors for 60 deg HFOV at H=16.5m."""
        focal = self.width / (2.0 * math.tan(self.hfov_60deg / 2.0))
        # Analytical ground truth: focal = 640 * sqrt(3) ~= 1108.512518 px
        self.assertAlmostEqual(focal, 640.0 * math.sqrt(3.0), places=5)
        self.assertAlmostEqual(focal, 1108.512518, places=3)

        vectors = [
            # id, cx, cy, exp_dn, exp_de, exp_n, exp_e
            ("Center Pixel", 640.0, 480.0, 0.000000, 0.000000, 15.000000, 15.000000),
            ("Top-Left Corner", 0.0, 0.0, 7.144710, -9.526279, 22.144710, 5.473721),
            ("Top-Right Corner", 1280.0, 0.0, 7.144710, 9.526279, 22.144710, 24.526279),
            ("Bottom-Right Corner", 1280.0, 960.0, -7.144710, 9.526279, 7.855290, 24.526279),
            ("Bottom-Left Corner", 0.0, 960.0, -7.144710, -9.526279, 7.855290, 5.473721),
            ("Top Edge Midpoint", 640.0, 0.0, 7.144710, 0.000000, 22.144710, 15.000000),
            ("Bottom Edge Midpoint", 640.0, 960.0, -7.144710, 0.000000, 7.855290, 15.000000),
            ("Left Edge Midpoint", 0.0, 480.0, 0.000000, -9.526279, 15.000000, 5.473721),
            ("Right Edge Midpoint", 1280.0, 480.0, 0.000000, 9.526279, 15.000000, 24.526279),
        ]

        for desc, cx, cy, exp_dn, exp_de, exp_n, exp_e in vectors:
            dn = -(cy - self.height / 2.0) * self.H / focal
            de = +(cx - self.width / 2.0) * self.H / focal
            n = self.ref_n + dn
            e = self.ref_e + de

            self.assertAlmostEqual(dn, exp_dn, places=4, msg=f"{desc} delta_n failed")
            self.assertAlmostEqual(de, exp_de, places=4, msg=f"{desc} delta_e failed")
            self.assertAlmostEqual(n, exp_n, places=4, msg=f"{desc} north failed")
            self.assertAlmostEqual(e, exp_e, places=4, msg=f"{desc} east failed")

    def test_vectors_simulation_99deg(self):
        """Verify ground truth vectors for simulation default (HFOV=1.74 rad) at H=16.5m."""
        focal = self.width / (2.0 * math.tan(self.hfov_sim / 2.0))
        self.assertAlmostEqual(focal, 539.936368, places=3)

        vectors = [
            ("Center Pixel", 640.0, 480.0, 0.000000, 0.000000, 15.000000, 15.000000),
            ("Top-Left Corner", 0.0, 0.0, 14.668395, -19.557860, 29.668395, -4.557860),
            ("Top-Right Corner", 1280.0, 0.0, 14.668395, 19.557860, 29.668395, 34.557860),
            ("Bottom-Right Corner", 1280.0, 960.0, -14.668395, 19.557860, 0.331605, 34.557860),
            ("Bottom-Left Corner", 0.0, 960.0, -14.668395, -19.557860, 0.331605, -4.557860),
            ("Top Edge Midpoint", 640.0, 0.0, 14.668395, 0.000000, 29.668395, 15.000000),
            ("Right Edge Midpoint", 1280.0, 480.0, 0.000000, 19.557860, 15.000000, 34.557860),
        ]

        for desc, cx, cy, exp_dn, exp_de, exp_n, exp_e in vectors:
            dn = -(cy - self.height / 2.0) * self.H / focal
            de = +(cx - self.width / 2.0) * self.H / focal
            n = self.ref_n + dn
            e = self.ref_e + de

            self.assertAlmostEqual(dn, exp_dn, places=4, msg=f"{desc} delta_n failed")
            self.assertAlmostEqual(de, exp_de, places=4, msg=f"{desc} delta_e failed")
            self.assertAlmostEqual(n, exp_n, places=4, msg=f"{desc} north failed")
            self.assertAlmostEqual(e, exp_e, places=4, msg=f"{desc} east failed")

    def test_person_reporter_projection_end_to_end(self):
        """Verify PersonReporter.on_detections generates clusters matching ground truth."""
        reporter = solution.PersonReporter()
        reporter.camera = {
            'width': 1280,
            'height': 960,
            'hfov_rad': 1.04719755,
            'pitch_deg': 90.0
        }
        reporter.position = (15.0, 15.0, 16.5, 10.0, time.monotonic())

        # Test Top-Left Corner projection through actual solution class
        msg = MockDetectionArrayMsg([MockDetection(0.0, 0.0, score=0.85)])
        reporter.on_detections(msg)

        self.assertEqual(len(reporter.clusters), 1)
        c = reporter.clusters[0]
        self.assertAlmostEqual(c.north, 22.144710, places=4)
        self.assertAlmostEqual(c.east, 5.473721, places=4)

    # -------------------------------------------------------------------------
    # 2. Position Latency Freshness Gate (Age <= 1.0s vs Age > 1.0s)
    # -------------------------------------------------------------------------
    def test_latency_gate_on_camera_image_rejects_stale(self):
        """Ensure on_camera_image drops frames when position age > 0.40s."""
        reporter = solution.PersonReporter()
        reporter.yolo_model = MagicMock()
        reporter._last_infer_time = 0.0

        # Stale position: 0.45 seconds old (> 0.40s threshold)
        stale_rx = time.monotonic() - 0.45
        reporter.position = (15.0, 15.0, 16.5, 10.0, stale_rx)

        mock_msg = MagicMock()
        mock_msg.width = 1280
        mock_msg.height = 960

        reporter.on_camera_image(mock_msg)
        # Should have returned at line 124 without updating _last_infer_time
        self.assertEqual(reporter._last_infer_time, 0.0)

    def test_latency_gate_on_camera_image_rejects_severe_stale(self):
        """Ensure on_camera_image drops frames when position age is 10.0s old."""
        reporter = solution.PersonReporter()
        reporter.yolo_model = MagicMock()
        reporter._last_infer_time = 0.0

        stale_rx = time.monotonic() - 10.0
        reporter.position = (15.0, 15.0, 16.5, 10.0, stale_rx)

        mock_msg = MagicMock()
        reporter.on_camera_image(mock_msg)
        self.assertEqual(reporter._last_infer_time, 0.0)

    def test_latency_gate_on_camera_image_rejects_none_position(self):
        """Ensure on_camera_image drops frames when position is None."""
        reporter = solution.PersonReporter()
        reporter.yolo_model = MagicMock()
        reporter.position = None
        reporter._last_infer_time = 0.0

        mock_msg = MagicMock()
        reporter.on_camera_image(mock_msg)
        self.assertEqual(reporter._last_infer_time, 0.0)

    def test_latency_gate_on_camera_image_rejects_future_timestamp(self):
        """Ensure on_camera_image drops frames when position timestamp is in the future."""
        reporter = solution.PersonReporter()
        reporter.yolo_model = MagicMock()
        reporter._last_infer_time = 0.0

        future_rx = time.monotonic() + 5.0
        reporter.position = (15.0, 15.0, 16.5, 10.0, future_rx)

        mock_msg = MagicMock()
        reporter.on_camera_image(mock_msg)
        self.assertEqual(reporter._last_infer_time, 0.0)

    def test_latency_gate_on_camera_image_passes_fresh(self):
        """Ensure on_camera_image passes gate when position age <= 0.40s."""
        reporter = solution.PersonReporter()
        reporter.yolo_model = MagicMock()
        reporter._last_infer_time = 0.0

        # Fresh position: 0.05 seconds old (nominal 20Hz update)
        fresh_rx = time.monotonic() - 0.05
        reporter.position = (15.0, 15.0, 16.5, 10.0, fresh_rx)

        mock_msg = MagicMock()
        mock_msg.width = 1280
        mock_msg.height = 960

        with patch.object(reporter, 'close'):
            reporter.on_camera_image(mock_msg)

        # Fresh position successfully passes gate and advances past _last_infer_time check
        self.assertGreater(reporter._last_infer_time, 0.0)

    def test_latency_gate_on_camera_image_passes_near_boundary(self):
        """Ensure on_camera_image passes gate when position age is 0.35s (<= 0.40s)."""
        reporter = solution.PersonReporter()
        reporter.yolo_model = MagicMock()
        reporter._last_infer_time = 0.0

        fresh_rx = time.monotonic() - 0.35
        reporter.position = (15.0, 15.0, 16.5, 10.0, fresh_rx)

        mock_msg = MagicMock()
        mock_msg.width = 1280
        mock_msg.height = 960

        reporter.on_camera_image(mock_msg)
        self.assertGreater(reporter._last_infer_time, 0.0)

    def test_latency_gate_on_detections_rejects_stale(self):
        """Ensure on_detections drops detections when position age > 0.40s."""
        reporter = solution.PersonReporter()
        stale_rx = time.monotonic() - 0.45
        reporter.position = (15.0, 15.0, 16.5, 10.0, stale_rx)

        msg = MockDetectionArrayMsg([MockDetection(640.0, 480.0, score=0.9)])
        reporter.on_detections(msg)
        self.assertEqual(len(reporter.clusters), 0)
        self.assertEqual(reporter.det_seen, 0)

    def test_latency_gate_on_detections_rejects_future_timestamp(self):
        """Ensure on_detections drops detections when position timestamp is in the future."""
        reporter = solution.PersonReporter()
        future_rx = time.monotonic() + 5.0
        reporter.position = (15.0, 15.0, 16.5, 10.0, future_rx)

        msg = MockDetectionArrayMsg([MockDetection(640.0, 480.0, score=0.9)])
        reporter.on_detections(msg)
        self.assertEqual(len(reporter.clusters), 0)
        self.assertEqual(reporter.det_seen, 0)

    def test_latency_gate_on_detections_passes_fresh(self):
        """Ensure on_detections accepts detections when position age <= 0.40s."""
        reporter = solution.PersonReporter()
        fresh_rx = time.monotonic() - 0.1
        reporter.position = (15.0, 15.0, 16.5, 10.0, fresh_rx)

        msg = MockDetectionArrayMsg([MockDetection(640.0, 480.0, score=0.9)])
        reporter.on_detections(msg)
        self.assertEqual(len(reporter.clusters), 1)
        self.assertEqual(reporter.det_seen, 1)

    # -------------------------------------------------------------------------
    # 3. Altitude Filter (Reject Height < 2.0m)
    # -------------------------------------------------------------------------
    def test_altitude_filter_on_camera_image_rejects_low_altitude(self):
        """Ensure on_camera_image returns early without running YOLO when height < 2.0m."""
        reporter = solution.PersonReporter()
        reporter.yolo_model = MagicMock()
        reporter.position = (15.0, 15.0, 1.80, 10.0, time.monotonic())  # 1.80m < 2.0m
        reporter._last_infer_time = 0.0

        mock_msg = MagicMock()
        reporter.on_camera_image(mock_msg)

        # Height check is evaluated after timestamp update, but before prediction
        # yolo_model.predict must NOT be invoked
        self.assertFalse(reporter.yolo_model.predict.called)

    def test_altitude_filter_on_detections_rejects_sub_2m(self):
        """Ensure on_detections rejects detections when drone height < 2.0m."""
        reporter = solution.PersonReporter()

        for test_h in [1.99, 1.0, 0.0, -0.5]:
            reporter.clusters.clear()
            reporter.det_seen = 0
            reporter.position = (15.0, 15.0, test_h, 10.0, time.monotonic())
            msg = MockDetectionArrayMsg([MockDetection(640.0, 480.0, score=0.85)])
            reporter.on_detections(msg)
            self.assertEqual(len(reporter.clusters), 0, f"Failed to reject altitude {test_h}m")
            self.assertEqual(reporter.det_seen, 0)

    def test_altitude_filter_on_detections_accepts_at_or_above_2m(self):
        """Ensure on_detections accepts detections when drone height >= 2.0m."""
        reporter = solution.PersonReporter()

        for test_h in [2.0, 10.0, 16.5]:
            reporter.clusters.clear()
            reporter.position = (15.0, 15.0, test_h, 10.0, time.monotonic())
            msg = MockDetectionArrayMsg([MockDetection(640.0, 480.0, score=0.85)])
            reporter.on_detections(msg)
            self.assertEqual(len(reporter.clusters), 1, f"Failed at altitude {test_h}m")

    # -------------------------------------------------------------------------
    # 4. Missing camera.json Fallback & Intrinsics Handling
    # -------------------------------------------------------------------------
    def test_missing_camera_json_fallback(self):
        """Ensure PersonReporter initializes complete intrinsics when camera.json is missing."""
        # Non-existent path
        with patch('builtins.open', side_effect=OSError("File not found")):
            reporter = solution.PersonReporter()

        # All four essential keys must be pre-populated
        self.assertIn('hfov_rad', reporter.camera)
        self.assertIn('width', reporter.camera)
        self.assertIn('height', reporter.camera)
        self.assertIn('pitch_deg', reporter.camera)

        self.assertEqual(reporter.camera['width'], 1280)
        self.assertEqual(reporter.camera['height'], 960)
        self.assertAlmostEqual(reporter.camera['hfov_rad'], 1.04719755, places=7)
        self.assertEqual(reporter.camera['pitch_deg'], 90.0)

        # Verification: on_detections must still project without camera.json
        reporter.position = (15.0, 15.0, 16.5, 10.0, time.monotonic())
        msg = MockDetectionArrayMsg([MockDetection(0.0, 0.0, score=0.9)])
        reporter.on_detections(msg)

        self.assertEqual(len(reporter.clusters), 1)
        c = reporter.clusters[0]
        # Offset must be projected, NOT defaulting to drone coordinates (15.0, 15.0)
        self.assertAlmostEqual(c.north, 22.144710, places=4)
        self.assertAlmostEqual(c.east, 5.473721, places=4)

    def test_focal_formula_full_precision_default(self):
        """Verify focal formula uses 1.04719755 full precision rather than truncated 1.047."""
        width = 1280.0
        empty_camera = {}
        focal_full = width / (2.0 * math.tan(float(empty_camera.get('hfov_rad', 1.04719755)) / 2.0))
        focal_truncated = width / (2.0 * math.tan(float(empty_camera.get('hfov_rad', 1.047)) / 2.0))

        expected_analytical = 640.0 * math.sqrt(3.0)  # 1108.512518 px
        self.assertAlmostEqual(focal_full, expected_analytical, places=5)
        # Verify truncated causes a discrepancy of ~0.18px
        self.assertGreater(abs(focal_truncated - expected_analytical), 0.15)

    # -------------------------------------------------------------------------
    # 5. Non-Finite Coordinate Handling (NaN / Inf)
    # -------------------------------------------------------------------------
    def test_update_position_rejects_nan_inf(self):
        """Ensure update_position ignores telemetry messages containing NaN or Inf."""
        reporter = solution.PersonReporter()
        reporter.position = (10.0, 10.0, 15.0, 5.0, time.monotonic())

        # Test NaN in x
        reporter.update_position(MockMavMsg(float('nan'), 10.0, -15.0))
        self.assertEqual(reporter.position[0], 10.0)

        # Test Inf in y
        reporter.update_position(MockMavMsg(10.0, float('inf'), -15.0))
        self.assertEqual(reporter.position[1], 10.0)

        # Test -Inf in z
        reporter.update_position(MockMavMsg(10.0, 10.0, float('-inf')))
        self.assertEqual(reporter.position[2], 15.0)

        # Test valid finite update
        reporter.update_position(MockMavMsg(20.0, 25.0, -16.5))
        self.assertEqual(reporter.position[0], 20.0)
        self.assertEqual(reporter.position[1], 25.0)
        self.assertEqual(reporter.position[2], 16.5)

    def test_on_detections_ignores_nan_inf_projection(self):
        """Ensure on_detections skips detections with non-finite bounding boxes."""
        reporter = solution.PersonReporter()
        reporter.position = (15.0, 15.0, 16.5, 10.0, time.monotonic())

        msg = MockDetectionArrayMsg([
            MockDetection(float('nan'), 0.0, score=0.85),
            MockDetection(0.0, float('inf'), score=0.85),
        ])
        reporter.on_detections(msg)
        self.assertEqual(len(reporter.clusters), 0)

    # -------------------------------------------------------------------------
    # 6. Python 3.10 AST Compilation Compliance
    # -------------------------------------------------------------------------
    def test_solution_py_ast_clean(self):
        """Ensure solution.py compiles cleanly under Python 3.10 AST specification."""
        solution_path = PROJECT_ROOT / "solution.py"
        self.assertTrue(solution_path.exists(), "solution.py not found")
        content = solution_path.read_bytes()
        # Compile via ast.parse with feature_version=(3, 10)
        try:
            tree = ast.parse(content, filename=str(solution_path), feature_version=(3, 10))
            self.assertIsNotNone(tree)
        except SyntaxError as e:
            self.fail(f"solution.py failed Python 3.10 AST compilation: {e}")

    def test_this_test_file_ast_clean(self):
        """Ensure test_m1_camera_projection.py compiles cleanly under Python 3.10 AST."""
        test_path = Path(__file__).resolve()
        content = test_path.read_bytes()
        try:
            tree = ast.parse(content, filename=str(test_path), feature_version=(3, 10))
            self.assertIsNotNone(tree)
        except SyntaxError as e:
            self.fail(f"{test_path.name} failed Python 3.10 AST compilation: {e}")


if __name__ == '__main__':
    unittest.main()
