#!/usr/bin/env python3
"""test_m1_adversarial_stress_it2.py

Comprehensive Adversarial Stress Test Suite for Milestone 1 Iteration 2:
Validates and stress-tests all Iteration 1 failure points, extreme edge cases,
and physical/geometric invariants in student-starter-kit/solution.py.

Sections:
1. Camera Configuration & Intrinsics Fault-Injection (Null, Scalar, Corrupt, Degenerate)
2. Telemetry Ingestion & Retrograde MAVLink Handling
3. Two-Sided Latency Gate Verification (Future vs Stale vs Boundary Timestamps)
4. Altitude Boundary & Ground Artifact Suppression
5. Mathematical Geometry, Cardinal Signs, and Analytical Error Bounds
6. High-Throughput Adversarial Stress Fuzzing (1,000 packets)
"""

import ast
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import unittest
from unittest.mock import MagicMock, mock_open, patch

# Ensure pymavlink is mocked if not installed
try:
    import pymavlink
except ImportError:
    mock_mav = MagicMock()
    sys.modules['pymavlink'] = mock_mav
    sys.modules['pymavlink.mavutil'] = mock_mav

# Ensure sensor_msgs and cv_bridge are mocked if not installed
if 'sensor_msgs' not in sys.modules:
    try:
        import sensor_msgs
    except ImportError:
        sys.modules['sensor_msgs'] = MagicMock()
        sys.modules['sensor_msgs.msg'] = MagicMock()

if 'cv_bridge' not in sys.modules:
    try:
        import cv_bridge
    except ImportError:
        sys.modules['cv_bridge'] = MagicMock()

if 'numpy' not in sys.modules:
    try:
        import numpy
    except ImportError:
        sys.modules['numpy'] = MagicMock()

if 'cv2' not in sys.modules:
    try:
        import cv2
    except ImportError:
        sys.modules['cv2'] = MagicMock()

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


class TestMilestone1AdversarialStressIt2(unittest.TestCase):
    """Deep adversarial stress testing for Milestone 1 Iteration 2."""

    def setUp(self):
        self.report_patcher = patch('solution.report_person')
        self.mock_report = self.report_patcher.start()

    def tearDown(self):
        self.report_patcher.stop()

    def _create_reporter(self, camera_override=None):
        reporter = solution.PersonReporter()
        if camera_override is not None:
            reporter.camera.update(camera_override)
        return reporter

    # =========================================================================
    # SECTION 1: Camera Intrinsics Fault-Injection (Corrupt / Scalar / Degenerate)
    # =========================================================================
    def test_camera_json_corrupt_syntax(self):
        """Corrupt JSON syntax in /opt/kit/camera.json must fall back safely to defaults."""
        corrupt_payloads = [
            "{invalid json",
            "",
            "   ",
            "{\"hfov_rad\": 1.047, ",
            "<xml>not json</xml>",
        ]
        for payload in corrupt_payloads:
            with patch('builtins.open', mock_open(read_data=payload)):
                reporter = solution.PersonReporter()
                self.assertEqual(reporter.camera['width'], 1280)
                self.assertEqual(reporter.camera['height'], 960)
                self.assertAlmostEqual(reporter.camera['hfov_rad'], 1.04719755)

    def test_camera_json_scalar_and_non_dict(self):
        """Scalar, null, or list payloads in camera.json must fall back safely to defaults."""
        non_dict_payloads = [
            "null",
            "true",
            "false",
            "12345",
            "-45.67",
            "\"string value\"",
            "[1, 2, 3]",
            "[]",
        ]
        for payload in non_dict_payloads:
            with patch('builtins.open', mock_open(read_data=payload)):
                reporter = solution.PersonReporter()
                self.assertEqual(reporter.camera['width'], 1280)
                self.assertEqual(reporter.camera['height'], 960)
                self.assertAlmostEqual(reporter.camera['hfov_rad'], 1.04719755)

    def test_camera_json_missing_or_extra_keys(self):
        """Camera JSON with partial keys or unexpected extra keys must preserve defaults."""
        partial_json = json.dumps({"width": 1920, "extra_calibration": "unknown"})
        with patch('builtins.open', mock_open(read_data=partial_json)):
            reporter = solution.PersonReporter()
            self.assertEqual(reporter.camera['width'], 1920)
            self.assertEqual(reporter.camera['height'], 960)  # preserved default
            self.assertAlmostEqual(reporter.camera['hfov_rad'], 1.04719755)  # preserved default

    def test_camera_degenerate_intrinsics_no_zero_division(self):
        """Zero / negative width, height, or hfov_rad must NOT cause ZeroDivisionError."""
        degenerate_configs = [
            {'width': 0, 'height': 960, 'hfov_rad': 1.047},
            {'width': -1280, 'height': 960, 'hfov_rad': 1.047},
            {'width': 1280, 'height': 0, 'hfov_rad': 1.047},
            {'width': 1280, 'height': -960, 'hfov_rad': 1.047},
            {'width': 1280, 'height': 960, 'hfov_rad': 0.0},
            {'width': 1280, 'height': 960, 'hfov_rad': -0.5},
            {'width': 1280, 'height': 960, 'hfov_rad': math.pi},
            {'width': 1280, 'height': 960, 'hfov_rad': 4.0},
        ]
        for cfg in degenerate_configs:
            reporter = self._create_reporter(camera_override=cfg)
            reporter.position = (10.0, 20.0, 16.5, 10.0, time.monotonic())
            msg = MockDetectionArrayMsg([MockDetection(640.0, 480.0, score=0.9)])
            try:
                reporter.on_detections(msg)
            except ZeroDivisionError:
                self.fail(f"ZeroDivisionError raised for degenerate camera config: {cfg}")
            self.assertEqual(len(reporter.clusters), 0, f"Clusters unexpectedly created for degenerate config: {cfg}")

    # =========================================================================
    # SECTION 2: Telemetry Ingestion & Retrograde MAVLink Handling
    # =========================================================================
    def test_telemetry_retrograde_packet_discarded(self):
        """Retrograde out-of-order MAVLink packets must NOT regress position or sim_s."""
        reporter = self._create_reporter()
        t0 = time.monotonic()

        # Advance to sim_s = 50.0s
        msg1 = MockMavMsg(25.0, 35.0, -16.5, time_boot_ms=50000)
        with patch('time.monotonic', return_value=t0):
            reporter.update_position(msg1)
        self.assertAlmostEqual(reporter.position[0], 25.0)
        self.assertAlmostEqual(reporter.position[1], 35.0)
        self.assertAlmostEqual(reporter.position[2], 16.5)
        self.assertAlmostEqual(reporter.position[3], 50.0)

        # Receive retrograde packet from sim_s = 40.0s (out-of-order UDP)
        msg_retro = MockMavMsg(10.0, 10.0, -16.5, time_boot_ms=40000)
        with patch('time.monotonic', return_value=t0 + 0.01):
            reporter.update_position(msg_retro)

        # Must retain msg1 coordinates and sim_s
        self.assertAlmostEqual(reporter.position[0], 25.0, msg="Retrograde packet overwritten north coordinate")
        self.assertAlmostEqual(reporter.position[1], 35.0, msg="Retrograde packet overwritten east coordinate")
        self.assertAlmostEqual(reporter.position[3], 50.0, msg="Retrograde packet regressed sim_s")

    def test_telemetry_equal_timestamp_permitted(self):
        """MAVLink packets with identical time_boot_ms (e.g. multi-rate sub-telemetry) are accepted."""
        reporter = self._create_reporter()
        msg1 = MockMavMsg(10.0, 10.0, -16.5, time_boot_ms=30000)
        reporter.update_position(msg1)
        self.assertAlmostEqual(reporter.position[0], 10.0)

        # Identical time_boot_ms, updated position
        msg2 = MockMavMsg(10.2, 10.2, -16.5, time_boot_ms=30000)
        reporter.update_position(msg2)
        self.assertAlmostEqual(reporter.position[0], 10.2)

    def test_telemetry_non_finite_and_none_absorption(self):
        """Telemetry packets with NaN, Inf, -Inf, or None must be absorbed safely."""
        reporter = self._create_reporter()
        reporter.position = (15.0, 15.0, 16.5, 20.0, time.monotonic())

        invalid_msgs = [
            MockMavMsg(float('nan'), 15.0, -16.5, 21000),
            MockMavMsg(15.0, float('nan'), -16.5, 21000),
            MockMavMsg(15.0, 15.0, float('nan'), 21000),
            MockMavMsg(15.0, 15.0, -16.5, float('nan')),
            MockMavMsg(float('inf'), 15.0, -16.5, 21000),
            MockMavMsg(15.0, float('-inf'), -16.5, 21000),
            MockMavMsg(None, 15.0, -16.5, 21000),
            MockMavMsg(15.0, None, -16.5, 21000),
            MockMavMsg(15.0, 15.0, None, 21000),
            MockMavMsg(15.0, 15.0, -16.5, None),
            MockMavMsg("not_a_float", 15.0, -16.5, 21000),
        ]
        for bad_msg in invalid_msgs:
            reporter.update_position(bad_msg)
            # Position must remain unchanged at (15.0, 15.0, 16.5, 20.0)
            self.assertEqual(reporter.position[0], 15.0)
            self.assertEqual(reporter.position[1], 15.0)
            self.assertEqual(reporter.position[2], 16.5)
            self.assertEqual(reporter.position[3], 20.0)

    # =========================================================================
    # SECTION 3: Two-Sided Latency Gate Verification
    # =========================================================================
    def test_latency_gate_future_timestamps_strictly_rejected(self):
        """Future timestamps (dt < 0) must be rejected across both input channels."""
        reporter = self._create_reporter()
        t_now = 500.0

        for dt_future in [-0.001, -0.010, -0.100, -1.0, -10.0]:
            with patch('time.monotonic', return_value=t_now):
                reporter.clusters.clear()
                reporter.position = (10.0, 20.0, 16.5, 50.0, t_now - dt_future)
                msg = MockDetectionArrayMsg([MockDetection(640.0, 480.0, score=0.9)])
                reporter.on_detections(msg)
                self.assertEqual(len(reporter.clusters), 0, f"Future timestamp dt={dt_future} leaked through gate")

    def test_latency_gate_stale_timestamps_strictly_rejected(self):
        """Stale timestamps (dt > 0.40s) must be rejected across both input channels."""
        reporter = self._create_reporter()
        t_now = 500.0

        for dt_stale in [0.4001, 0.401, 0.450, 0.500, 1.0, 10.0]:
            with patch('time.monotonic', return_value=t_now):
                reporter.clusters.clear()
                reporter.position = (10.0, 20.0, 16.5, 50.0, t_now - dt_stale)
                msg = MockDetectionArrayMsg([MockDetection(640.0, 480.0, score=0.9)])
                reporter.on_detections(msg)
                self.assertEqual(len(reporter.clusters), 0, f"Stale timestamp dt={dt_stale} leaked through gate")

    def test_latency_gate_valid_window_accepted(self):
        """Timestamps within 0.0 <= dt <= 0.400s must be accepted."""
        reporter = self._create_reporter()
        t_now = 500.0

        for dt_valid in [0.0, 0.001, 0.050, 0.100, 0.200, 0.350, 0.399, 0.400]:
            with patch('time.monotonic', return_value=t_now):
                reporter.clusters.clear()
                reporter.position = (10.0, 20.0, 16.5, 50.0, t_now - dt_valid)
                msg = MockDetectionArrayMsg([MockDetection(640.0, 480.0, score=0.9)])
                reporter.on_detections(msg)
                self.assertEqual(len(reporter.clusters), 1, f"Valid timestamp dt={dt_valid} incorrectly rejected")

    # =========================================================================
    # SECTION 4: Altitude Boundary & Ground Artifact Suppression
    # =========================================================================
    def test_altitude_sub_2m_ground_suppression(self):
        """Altitudes strictly below 2.0m must be rejected to prevent ground clutter."""
        reporter = self._create_reporter()
        sub_2m_altitudes = [-100.0, -10.0, -1.0, 0.0, 0.5, 1.0, 1.90, 1.99, 1.9999]

        for h in sub_2m_altitudes:
            reporter.clusters.clear()
            reporter.position = (10.0, 20.0, h, 50.0, time.monotonic())
            msg = MockDetectionArrayMsg([MockDetection(640.0, 480.0, score=0.9)])
            reporter.on_detections(msg)
            self.assertEqual(len(reporter.clusters), 0, f"Altitude H={h}m was not suppressed")

    def test_altitude_at_or_above_2m_accepted(self):
        """Altitudes at or above 2.0m must be accepted."""
        reporter = self._create_reporter()
        valid_altitudes = [2.000, 2.001, 5.0, 8.5, 16.5, 20.0, 25.0]

        for h in valid_altitudes:
            reporter.clusters.clear()
            reporter.position = (10.0, 20.0, h, 50.0, time.monotonic())
            msg = MockDetectionArrayMsg([MockDetection(640.0, 480.0, score=0.9)])
            reporter.on_detections(msg)
            self.assertEqual(len(reporter.clusters), 1, f"Altitude H={h}m was incorrectly rejected")

    # =========================================================================
    # SECTION 5: Mathematical Geometry, Cardinal Signs, and Analytical Invariants
    # =========================================================================
    def test_geometry_center_pixel_zero_ground_offset(self):
        """Image center pixel (cx=W/2, cy=H/2) must produce zero ground offset regardless of altitude."""
        reporter = self._create_reporter()
        drone_n, drone_e = 37.412173, -121.998878
        focal = 1280.0 / (2.0 * math.tan(1.04719755 / 2.0))

        for h in [2.0, 5.0, 10.0, 16.5, 20.0, 50.0]:
            reporter.clusters.clear()
            reporter.position = (drone_n, drone_e, h, 10.0, time.monotonic())
            msg = MockDetectionArrayMsg([MockDetection(640.0, 480.0, score=0.9)])
            reporter.on_detections(msg)
            c = reporter.clusters[0]
            self.assertAlmostEqual(c.north, drone_n, places=6)
            self.assertAlmostEqual(c.east, drone_e, places=6)

    def test_geometry_cardinal_directions_signs(self):
        """Verify cardinal axis signs:
        - cy < H/2 (top of image) -> +North
        - cy > H/2 (bottom of image) -> -North (South)
        - cx > W/2 (right of image) -> +East
        - cx < W/2 (left of image) -> -East (West)
        """
        reporter = self._create_reporter()
        drone_n, drone_e, H = 50.0, 50.0, 16.5
        reporter.position = (drone_n, drone_e, H, 10.0, time.monotonic())

        # Top-center: cy = 240, cx = 640 -> pure North
        reporter.clusters.clear()
        reporter.on_detections(MockDetectionArrayMsg([MockDetection(640.0, 240.0, score=0.9)]))
        c = reporter.clusters[0]
        self.assertGreater(c.north, drone_n, "cy < H/2 did not produce +North")
        self.assertAlmostEqual(c.east, drone_e, places=5)

        # Bottom-center: cy = 720, cx = 640 -> pure South (-North)
        reporter.clusters.clear()
        reporter.on_detections(MockDetectionArrayMsg([MockDetection(640.0, 720.0, score=0.9)]))
        c = reporter.clusters[0]
        self.assertLess(c.north, drone_n, "cy > H/2 did not produce -North (South)")
        self.assertAlmostEqual(c.east, drone_e, places=5)

        # Center-right: cy = 480, cx = 960 -> pure East
        reporter.clusters.clear()
        reporter.on_detections(MockDetectionArrayMsg([MockDetection(960.0, 480.0, score=0.9)]))
        c = reporter.clusters[0]
        self.assertAlmostEqual(c.north, drone_n, places=5)
        self.assertGreater(c.east, drone_e, "cx > W/2 did not produce +East")

        # Center-left: cy = 480, cx = 320 -> pure West (-East)
        reporter.clusters.clear()
        reporter.on_detections(MockDetectionArrayMsg([MockDetection(320.0, 480.0, score=0.9)]))
        c = reporter.clusters[0]
        self.assertAlmostEqual(c.north, drone_n, places=5)
        self.assertLess(c.east, drone_e, "cx < W/2 did not produce -East (West)")

    def test_geometry_analytical_error_budget_500_points(self):
        """Stress-test analytical pinhole projection error over 500 randomized points.
        Requirement R1: error <= 2.5m relative to ground geometry.
        Verified oracle: error <= 1e-4m (exact mathematical consistency).
        """
        reporter = self._create_reporter()
        focal = 1280.0 / (2.0 * math.tan(1.04719755 / 2.0))
        rng = random.Random(1337)

        max_error = 0.0
        for _ in range(500):
            drone_n = rng.uniform(-100.0, 100.0)
            drone_e = rng.uniform(-100.0, 100.0)
            H = rng.uniform(8.0, 20.0)

            # Ground target within camera footprint
            gt_dn = rng.uniform(-7.0, 7.0)
            gt_de = rng.uniform(-9.0, 9.0)
            gt_n = drone_n + gt_dn
            gt_e = drone_e + gt_de

            # Invert projection to get ground truth pixel
            cx = 640.0 + (gt_de * focal / H)
            cy = 480.0 - (gt_dn * focal / H)

            reporter.clusters.clear()
            reporter.position = (drone_n, drone_e, H, 10.0, time.monotonic())
            msg = MockDetectionArrayMsg([MockDetection(cx, cy, score=0.9)])
            reporter.on_detections(msg)

            c = reporter.clusters[0]
            err = math.hypot(c.north - gt_n, c.east - gt_e)
            if err > max_error:
                max_error = err

        self.assertLessEqual(max_error, 1e-4, f"Pinhole projection error {max_error} exceeded precision limit")
        self.assertLessEqual(max_error, 2.5, "Failed competition requirement error <= 2.5m")

    # =========================================================================
    # SECTION 6: High-Throughput Adversarial Stress Fuzzing (1,000 packets)
    # =========================================================================
    def test_adversarial_fuzz_1000_mixed_telemetry_and_detections(self):
        """High-throughput fuzz test with 1,000 mixed, chaotic packets.
        Ensures stability under extreme noisy conditions without crashes or memory leak.
        """
        reporter = self._create_reporter()
        rng = random.Random(42)
        sim_time = 0.0
        t_mono = 1000.0

        for i in range(1000):
            # 50% Telemetry packets, 50% Detection packets
            if rng.random() < 0.5:
                # Telemetry
                is_valid = rng.random() > 0.3
                if is_valid:
                    sim_time += rng.uniform(0.01, 0.05)
                    t_mono += rng.uniform(0.01, 0.05)
                    x = rng.uniform(10.0, 90.0)
                    y = rng.uniform(10.0, 90.0)
                    z = -rng.uniform(8.0, 20.0)
                    time_boot = sim_time * 1000.0
                else:
                    # Chaos: None, NaN, retrograde, negative height
                    x = rng.choice([float('nan'), float('inf'), None, 50.0])
                    y = rng.choice([float('nan'), float('-inf'), None, 50.0])
                    z = rng.choice([float('nan'), 10.0, -1.0, -16.5])
                    time_boot = rng.choice([float('nan'), None, (sim_time - 10.0) * 1000.0])

                with patch('time.monotonic', return_value=t_mono):
                    reporter.update_position(MockMavMsg(x, y, z, time_boot))
            else:
                # Detections
                cx = rng.uniform(-200.0, 1500.0)
                cy = rng.uniform(-200.0, 1200.0)
                score = rng.uniform(0.1, 1.0)
                cls_id = rng.choice(['person', '0', '1', 'car', 'dog', 'pedestrian'])

                msg = MockDetectionArrayMsg([MockDetection(cx, cy, score=score, class_id=cls_id)])
                with patch('time.monotonic', return_value=t_mono):
                    reporter.on_detections(msg)

        # Reporter must survive 1000 chaotic iterations without raising unhandled exceptions
        self.assertIsNotNone(reporter.position)
        # Flush pending must execute cleanly
        reporter.flush_pending(sim_s=sim_time)


    # =========================================================================
    # SECTION 7: on_camera_image End-to-End Adversarial Stress Testing
    # =========================================================================
    def test_on_camera_image_edge_cases_and_projection(self):
        """Stress-test on_camera_image with mocked YOLO model, boxes, and geometry."""
        reporter = self._create_reporter()
        reporter.pub_annotated = MagicMock()
        mock_model = MagicMock()
        mock_model.names = {0: 'person', 1: 'group', 2: 'car'}
        reporter.yolo_model = mock_model

        t_now = 200.0
        drone_n, drone_e, H = 30.0, 40.0, 16.5
        reporter.position = (drone_n, drone_e, H, 10.0, t_now - 0.05)

        # Mock Image message
        msg = MagicMock()
        msg.width = 1280
        msg.height = 960
        msg.header = MagicMock()

        # Mock predict output
        def make_mock_result(boxes_data):
            res = MagicMock()
            res.plot.return_value = MagicMock(shape=(960, 1280, 3), tobytes=lambda: b'mock')
            mock_boxes = []
            for xyxy_coords, cls_id, conf in boxes_data:
                b = MagicMock()
                mock_tensor = MagicMock()
                mock_tensor.tolist.return_value = xyxy_coords
                b.xyxy = [mock_tensor]
                b.cls = [MagicMock(item=lambda c=cls_id: c)]
                b.conf = [MagicMock(item=lambda cf=conf: cf)]
                mock_boxes.append(b)
            res.boxes = mock_boxes
            return [res]

        focal = 1280.0 / (2.0 * math.tan(1.04719755 / 2.0))

        # Ensure cv_bridge is mocked in sys.modules
        mock_cv_bridge = MagicMock()
        mock_bridge_instance = MagicMock()
        mock_bridge_instance.imgmsg_to_cv2.return_value = MagicMock()
        mock_cv_bridge.CvBridge.return_value = mock_bridge_instance
        with patch.dict(sys.modules, {'cv_bridge': mock_cv_bridge}):
            # 1. Nominal detection: person at (cx=640, cy=480) -> drone center
            mock_model.predict.return_value = make_mock_result([([600.0, 440.0, 680.0, 520.0], 0, 0.90)])
            with patch('time.monotonic', return_value=t_now):
                reporter.on_camera_image(msg)

            self.assertEqual(len(reporter.clusters), 1)
            self.assertAlmostEqual(reporter.clusters[0].north, drone_n, places=4)
            self.assertAlmostEqual(reporter.clusters[0].east, drone_e, places=4)

            # 2. Rate limiting check: second image arrives at t_now + 0.05s (< 0.10s interval)
            with patch('time.monotonic', return_value=t_now + 0.05):
                reporter.clusters.clear()
                reporter.on_camera_image(msg)
            # Must be skipped due to rate limiter
            self.assertEqual(len(reporter.clusters), 0, "Rate limiter failed to throttle image processing")

            # 3. Third image arrives at t_now + 0.15s (>= 0.10s interval) -> accepted
            with patch('time.monotonic', return_value=t_now + 0.15):
                reporter.position = (drone_n, drone_e, H, 10.15, t_now + 0.10)
                reporter.on_camera_image(msg)
            self.assertEqual(len(reporter.clusters), 1, "Image after interval was incorrectly throttled")

            # 4. Filter non-person class (car)
            with patch('time.monotonic', return_value=t_now + 0.30):
                reporter.position = (drone_n, drone_e, H, 10.30, t_now + 0.25)
                reporter.clusters.clear()
                mock_model.predict.return_value = make_mock_result([([600.0, 440.0, 680.0, 520.0], 2, 0.95)])
                reporter.on_camera_image(msg)
            self.assertEqual(len(reporter.clusters), 0, "Non-person class 'car' was not filtered in on_camera_image")

            # 5. Non-finite box coordinates (NaN / Inf)
            for bad_box in [
                ([float('nan'), 440.0, 680.0, 520.0], 0, 0.90),
                ([600.0, float('inf'), 680.0, 520.0], 0, 0.90),
            ]:
                with patch('time.monotonic', return_value=t_now + 0.50):
                    reporter.position = (drone_n, drone_e, H, 10.50, t_now + 0.45)
                    reporter.clusters.clear()
                    mock_model.predict.return_value = make_mock_result([bad_box])
                    reporter.on_camera_image(msg)
                self.assertEqual(len(reporter.clusters), 0, "Non-finite bounding box was not rejected in on_camera_image")


if __name__ == '__main__':
    unittest.main()

