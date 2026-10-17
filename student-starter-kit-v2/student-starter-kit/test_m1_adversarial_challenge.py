#!/usr/bin/env python3
"""test_m1_adversarial_challenge.py

Adversarial stress-test harness for Milestone 1:
Camera Projection Equations & Telemetry Latency Checks in solution.py.

Tests 6 Adversarial Dimensions:
1. Boundary Timestamps: pos[4] age = 0.999s vs 1.000s vs 1.001s, negative time delta, non-monotonic times.
2. Boundary Altitudes: H = 1.99m, H = 2.00m, H = 16.5m, H = 20.0m, H = 25.0m, H <= 0.
3. Non-Finite Telemetry: NaN, Inf, -Inf in (x, y, z, time_boot_ms), None/missing fields.
4. Extreme FOVs and Resolutions: 1920x1080 @ 1.74 rad, zero FOV, ultra-wide FOV, zero resolution.
5. Image Pixel Extremes: sub-pixel coords, negative coords, (0,0), (1280, 960), (2560, 1920), NaN/Inf coords.
6. Empirical Ground Plane Projection & Invariants Verification.
"""

import ast
import math
import os
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

# Ensure pymavlink is mocked if not available
try:
    import pymavlink
except ImportError:
    mock_mav = MagicMock()
    sys.modules['pymavlink'] = mock_mav
    sys.modules['pymavlink.mavutil'] = mock_mav

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


class TestMilestone1AdversarialChallenge(unittest.TestCase):
    """Adversarial stress-test suite targeting Milestone 1."""

    def setUp(self):
        # Isolate report_person calls
        self.report_patcher = patch('solution.report_person')
        self.mock_report = self.report_patcher.start()

    def tearDown(self):
        self.report_patcher.stop()

    def _create_reporter(self, camera_override=None):
        reporter = solution.PersonReporter()
        if camera_override:
            reporter.camera.update(camera_override)
        return reporter

    # =========================================================================
    # DIMENSION 1: Boundary Timestamps and Freshness Gates
    # =========================================================================
    def test_timestamp_boundary_0_399s_accepted(self):
        """Timestamp age 0.399s (< 0.40s) MUST be accepted in on_detections."""
        reporter = self._create_reporter()
        t_fixed = 1000.0
        with patch('time.monotonic', return_value=t_fixed):
            # Age is exactly 0.399s
            reporter.position = (10.0, 20.0, 16.5, 50.0, t_fixed - 0.399)
            msg = MockDetectionArrayMsg([MockDetection(640.0, 480.0, score=0.9)])
            reporter.on_detections(msg)

        self.assertEqual(len(reporter.clusters), 1, "0.399s age was incorrectly rejected")
        self.assertEqual(reporter.det_seen, 1)

    def test_timestamp_boundary_exact_0_400s_accepted(self):
        """Timestamp age exactly 0.400s (<= 0.40s) MUST be accepted."""
        reporter = self._create_reporter()
        t_fixed = 1000.0
        with patch('time.monotonic', return_value=t_fixed):
            # Age is exactly 0.400s: t_fixed - (t_fixed - 0.400) = 0.400 -> Accepted!
            reporter.position = (10.0, 20.0, 16.5, 50.0, t_fixed - 0.400)
            msg = MockDetectionArrayMsg([MockDetection(640.0, 480.0, score=0.9)])
            reporter.on_detections(msg)

        self.assertEqual(len(reporter.clusters), 1, "Exact 0.400s boundary was rejected")
        self.assertEqual(reporter.det_seen, 1)

    def test_timestamp_boundary_0_401s_rejected(self):
        """Timestamp age 0.401s (> 0.40s) MUST be rejected in on_detections."""
        reporter = self._create_reporter()
        t_fixed = 1000.0
        with patch('time.monotonic', return_value=t_fixed):
            # Age is 0.401s: t_fixed - (t_fixed - 0.401) = 0.401 > 0.40 -> Rejected!
            reporter.position = (10.0, 20.0, 16.5, 50.0, t_fixed - 0.401)
            msg = MockDetectionArrayMsg([MockDetection(640.0, 480.0, score=0.9)])
            reporter.on_detections(msg)

        self.assertEqual(len(reporter.clusters), 0, "0.401s age was NOT rejected (latency leak)")
        self.assertEqual(reporter.det_seen, 0)

    def test_timestamp_boundary_on_camera_image(self):
        """Verify latency gate in on_camera_image under boundary timestamps."""
        reporter = self._create_reporter()
        reporter.yolo_model = MagicMock()
        mock_msg = MagicMock()
        mock_msg.width = 1280
        mock_msg.height = 960

        t_fixed = 500.0
        with patch('time.monotonic', return_value=t_fixed):
            # 1. Age 0.401s -> rejected early at latency gate
            reporter._last_infer_time = 0.0
            reporter.position = (10.0, 20.0, 16.5, 50.0, t_fixed - 0.401)
            reporter.on_camera_image(mock_msg)
            self.assertEqual(reporter._last_infer_time, 0.0, "0.401s in on_camera_image was not rejected")

            # 2. Age 0.399s -> passes latency gate and updates _last_infer_time
            reporter._last_infer_time = 0.0
            reporter.position = (10.0, 20.0, 16.5, 50.0, t_fixed - 0.399)
            reporter.on_camera_image(mock_msg)
            self.assertGreater(reporter._last_infer_time, 0.0, "0.399s in on_camera_image was rejected")

    def test_timestamp_negative_time_delta_investigation(self):
        """Adversarial case: pos[4] is in the future (negative latency delta).

        In hardened solution.py: 0.0 <= time.monotonic() - pos[4] <= MAX_POSITION_AGE_S.
        Future timestamps produce negative time delta and MUST be rejected.
        """
        reporter = self._create_reporter()
        t_fixed = 1000.0
        with patch('time.monotonic', return_value=t_fixed):
            # Position received from future (+5.0s)
            reporter.position = (10.0, 20.0, 16.5, 50.0, t_fixed + 5.0)
            msg = MockDetectionArrayMsg([MockDetection(640.0, 480.0, score=0.9)])
            reporter.on_detections(msg)

        # Hardened behavior: Future timestamp strictly rejected by two-sided gate
        self.assertEqual(len(reporter.clusters), 0, "Future timestamp was NOT rejected by two-sided gate")
        self.assertEqual(reporter.det_seen, 0)

    def test_timestamp_non_monotonic_sim_time(self):
        """Adversarial case: Out-of-order MAVLink telemetry arrival.

        If MAVLink packet with time_boot_ms = 50000 arrives, followed by out-of-order packet
        with time_boot_ms = 10000, update_position must preserve monotonic sim_s.
        """
        reporter = self._create_reporter()
        msg1 = MockMavMsg(10.0, 10.0, -16.5, time_boot_ms=50000)
        reporter.update_position(msg1)
        self.assertAlmostEqual(reporter.position[3], 50.0)

        # Out-of-order retrograde packet
        msg2 = MockMavMsg(10.5, 10.5, -16.5, time_boot_ms=10000)
        reporter.update_position(msg2)
        # Hardened behavior: update_position discards retrograde packets, preserving latest sim_s
        self.assertAlmostEqual(reporter.position[3], 50.0, msg="sim_s should not regress on out-of-order packet")
        self.assertAlmostEqual(reporter.position[0], 10.0)

    # =========================================================================
    # DIMENSION 2: Boundary Altitudes
    # =========================================================================
    def test_altitude_boundary_1_99m_rejected(self):
        """Altitude H = 1.99m (< 2.0m) MUST be rejected."""
        reporter = self._create_reporter()
        reporter.position = (10.0, 20.0, 1.99, 50.0, time.monotonic())

        msg = MockDetectionArrayMsg([MockDetection(640.0, 480.0, score=0.9)])
        reporter.on_detections(msg)

        self.assertEqual(len(reporter.clusters), 0, "H=1.99m was NOT rejected")
        self.assertEqual(reporter.det_seen, 0)

    def test_altitude_boundary_2_00m_accepted(self):
        """Altitude H = 2.00m (>= 2.0m) MUST be accepted."""
        reporter = self._create_reporter()
        reporter.position = (10.0, 20.0, 2.00, 50.0, time.monotonic())

        msg = MockDetectionArrayMsg([MockDetection(640.0, 480.0, score=0.9)])
        reporter.on_detections(msg)

        self.assertEqual(len(reporter.clusters), 1, "H=2.00m was incorrectly rejected")

    def test_altitude_boundary_16_5m_accepted(self):
        """Altitude H = 16.5m (nominal corridor altitude) MUST be accepted."""
        reporter = self._create_reporter()
        reporter.position = (10.0, 20.0, 16.50, 50.0, time.monotonic())

        msg = MockDetectionArrayMsg([MockDetection(640.0, 480.0, score=0.9)])
        reporter.on_detections(msg)

        self.assertEqual(len(reporter.clusters), 1, "H=16.5m nominal altitude failed")

    def test_altitude_boundary_20_0m_accepted(self):
        """Altitude H = 20.0m (upper bound of competition corridor) MUST be accepted."""
        reporter = self._create_reporter()
        reporter.position = (10.0, 20.0, 20.00, 50.0, time.monotonic())

        msg = MockDetectionArrayMsg([MockDetection(640.0, 480.0, score=0.9)])
        reporter.on_detections(msg)

        self.assertEqual(len(reporter.clusters), 1, "H=20.00m upper bound failed")

    def test_altitude_beyond_corridor_unfiltered(self):
        """Investigate whether altitudes above 20m (e.g. 25m, 100m) are accepted or filtered.

        PROJECT.md specifies corridor [8.0m, 20.0m].
        In solution.py, only height < 2.0 is checked, so H=25.0m is accepted.
        """
        reporter = self._create_reporter()
        reporter.position = (10.0, 20.0, 25.00, 50.0, time.monotonic())

        msg = MockDetectionArrayMsg([MockDetection(640.0, 480.0, score=0.9)])
        reporter.on_detections(msg)

        self.assertEqual(len(reporter.clusters), 1, "H=25m behavior check: accepted because no upper gate")

    def test_altitude_negative_and_zero_rejected(self):
        """Altitude H <= 0 (on ground or below ground) MUST be rejected."""
        reporter = self._create_reporter()

        for bad_h in [0.0, -0.01, -5.0, -100.0]:
            reporter.clusters.clear()
            reporter.position = (10.0, 20.0, bad_h, 50.0, time.monotonic())
            msg = MockDetectionArrayMsg([MockDetection(640.0, 480.0, score=0.9)])
            reporter.on_detections(msg)
            self.assertEqual(len(reporter.clusters), 0, f"Altitude {bad_h}m was NOT rejected")

    # =========================================================================
    # DIMENSION 3: Non-Finite Telemetry (NaN / Inf / Non-numeric)
    # =========================================================================
    def test_telemetry_nan_in_x_rejected(self):
        """NaN in msg.x MUST NOT update position."""
        reporter = self._create_reporter()
        reporter.position = (10.0, 20.0, 16.5, 50.0, time.monotonic())

        reporter.update_position(MockMavMsg(float('nan'), 20.0, -16.5, 51000))
        self.assertEqual(reporter.position[0], 10.0)

    def test_telemetry_inf_in_y_rejected(self):
        """Inf in msg.y MUST NOT update position."""
        reporter = self._create_reporter()
        reporter.position = (10.0, 20.0, 16.5, 50.0, time.monotonic())

        reporter.update_position(MockMavMsg(10.0, float('inf'), -16.5, 51000))
        self.assertEqual(reporter.position[1], 20.0)

    def test_telemetry_neginf_in_z_rejected(self):
        """-Inf in msg.z MUST NOT update position."""
        reporter = self._create_reporter()
        reporter.position = (10.0, 20.0, 16.5, 50.0, time.monotonic())

        reporter.update_position(MockMavMsg(10.0, 20.0, float('-inf'), 51000))
        self.assertEqual(reporter.position[2], 16.5)

    def test_telemetry_nan_in_time_boot_ms_rejected(self):
        """NaN in msg.time_boot_ms MUST NOT update position."""
        reporter = self._create_reporter()
        reporter.position = (10.0, 20.0, 16.5, 50.0, time.monotonic())

        reporter.update_position(MockMavMsg(10.0, 20.0, -16.5, float('nan')))
        self.assertEqual(reporter.position[3], 50.0)

    def test_telemetry_inf_in_time_boot_ms_rejected(self):
        """Inf in msg.time_boot_ms MUST NOT update position."""
        reporter = self._create_reporter()
        reporter.position = (10.0, 20.0, 16.5, 50.0, time.monotonic())

        reporter.update_position(MockMavMsg(10.0, 20.0, -16.5, float('inf')))
        self.assertEqual(reporter.position[3], 50.0)

    def test_telemetry_none_field_absorbed_gracefully(self):
        """Adversarial edge case: msg contains None for a numeric field.

        update_position must absorb None gracefully without crashing.
        """
        reporter = self._create_reporter()
        msg_none_x = MockMavMsg(None, 20.0, -16.5, 50000)
        reporter.update_position(msg_none_x)
        self.assertIsNone(reporter.position)

        reporter.position = (10.0, 20.0, 16.5, 50.0, time.monotonic())
        msg_none_time = MockMavMsg(10.0, 20.0, -16.5, None)
        reporter.update_position(msg_none_time)
        self.assertEqual(reporter.position[3], 50.0)

    # =========================================================================
    # DIMENSION 4: Extreme FOVs and Resolutions
    # =========================================================================
    def test_fov_resolution_1920x1080_hfov_1_74(self):
        """Verify projection under Full HD (1920x1080) and wide simulation HFOV 1.74 rad."""
        cam = {
            'width': 1920,
            'height': 1080,
            'hfov_rad': 1.74,
            'pitch_deg': 90.0
        }
        reporter = self._create_reporter(camera_override=cam)
        ref_n, ref_e, H = 50.0, 50.0, 16.5
        reporter.position = (ref_n, ref_e, H, 10.0, time.monotonic())

        # Analytical focal length
        expected_focal = 1920.0 / (2.0 * math.tan(1.74 / 2.0))  # ~= 809.90455 px
        self.assertAlmostEqual(expected_focal, 809.90455, places=3)

        # Project Top-Left Corner: (0, 0)
        msg_tl = MockDetectionArrayMsg([MockDetection(0.0, 0.0, score=0.9)])
        reporter.on_detections(msg_tl)
        self.assertEqual(len(reporter.clusters), 1)

        c = reporter.clusters[0]
        expected_n = ref_n + (540.0 * H / expected_focal)
        expected_e = ref_e - (960.0 * H / expected_focal)

        self.assertAlmostEqual(c.north, expected_n, places=4)
        self.assertAlmostEqual(c.east, expected_e, places=4)

    def test_fov_degenerate_zero_guarded_against_zero_division(self):
        """Adversarial case: hfov_rad is 0 or negative.

        math.tan(0) = 0.0 -> guarded in on_detections without raising ZeroDivisionError.
        """
        cam = {'width': 1280, 'height': 960, 'hfov_rad': 0.0, 'pitch_deg': 90.0}
        reporter = self._create_reporter(camera_override=cam)
        reporter.position = (10.0, 10.0, 16.5, 10.0, time.monotonic())

        msg = MockDetectionArrayMsg([MockDetection(640.0, 480.0, score=0.9)])
        reporter.on_detections(msg)
        self.assertEqual(len(reporter.clusters), 0)
        self.assertEqual(reporter.det_seen, 0)

    def test_resolution_zero_guarded_against_zero_division(self):
        """Adversarial case: width = 0 in camera config.

        Guarded in on_detections without raising ZeroDivisionError.
        """
        cam = {'width': 0, 'height': 960, 'hfov_rad': 1.04719755, 'pitch_deg': 90.0}
        reporter = self._create_reporter(camera_override=cam)
        reporter.position = (10.0, 10.0, 16.5, 10.0, time.monotonic())

        msg = MockDetectionArrayMsg([MockDetection(0.0, 0.0, score=0.9)])
        reporter.on_detections(msg)
        self.assertEqual(len(reporter.clusters), 0)
        self.assertEqual(reporter.det_seen, 0)

    def test_camera_json_null_and_scalar_safe(self):
        """Verify PersonReporter.__init__ does not crash when camera.json is null or scalar."""
        with patch('builtins.open', unittest.mock.mock_open(read_data="null")):
            reporter_null = solution.PersonReporter()
            self.assertEqual(reporter_null.camera['width'], 1280)

        with patch('builtins.open', unittest.mock.mock_open(read_data="12345")):
            reporter_scalar = solution.PersonReporter()
            self.assertEqual(reporter_scalar.camera['width'], 1280)

        with patch('builtins.open', unittest.mock.mock_open(read_data='["not", "a", "dict"]')):
            reporter_list = solution.PersonReporter()
            self.assertEqual(reporter_list.camera['width'], 1280)

    def test_fov_ultra_wide_approaching_pi(self):
        """Adversarial case: hfov_rad approaches pi (179 deg).

        tan(hfov/2) -> tan(89.5 deg) ~= 114.59.
        focal length becomes very small (~5.5 px), causing massive ground coordinates.
        Verify projection remains finite.
        """
        hfov_near_pi = math.radians(179.0)
        cam = {'width': 1280, 'height': 960, 'hfov_rad': hfov_near_pi, 'pitch_deg': 90.0}
        reporter = self._create_reporter(camera_override=cam)
        reporter.position = (0.0, 0.0, 16.5, 10.0, time.monotonic())

        msg = MockDetectionArrayMsg([MockDetection(640.0, 480.0, score=0.9)])
        reporter.on_detections(msg)
        self.assertEqual(len(reporter.clusters), 1)
        self.assertAlmostEqual(reporter.clusters[0].north, 0.0, places=4)
        self.assertAlmostEqual(reporter.clusters[0].east, 0.0, places=4)

    # =========================================================================
    # DIMENSION 5: Image Pixel Extremes
    # =========================================================================
    def test_pixel_extremes_sub_pixel_precision(self):
        """Verify sub-pixel coordinates (e.g. from continuous neural net regressors)."""
        reporter = self._create_reporter()
        ref_n, ref_e, H = 0.0, 0.0, 16.5
        reporter.position = (ref_n, ref_e, H, 10.0, time.monotonic())
        focal = 1280.0 / (2.0 * math.tan(1.04719755 / 2.0))

        cx_sub = 640.123456
        cy_sub = 479.876543
        msg = MockDetectionArrayMsg([MockDetection(cx_sub, cy_sub, score=0.9)])
        reporter.on_detections(msg)

        c = reporter.clusters[0]
        exp_n = -(cy_sub - 480.0) * H / focal
        exp_e = +(cx_sub - 640.0) * H / focal
        self.assertAlmostEqual(c.north, exp_n, places=5)
        self.assertAlmostEqual(c.east, exp_e, places=5)

    def test_pixel_extremes_negative_coordinates(self):
        """Adversarial case: negative pixel coordinates (bounding box extends off-screen)."""
        reporter = self._create_reporter()
        ref_n, ref_e, H = 100.0, 100.0, 16.5
        reporter.position = (ref_n, ref_e, H, 10.0, time.monotonic())
        focal = 1280.0 / (2.0 * math.tan(1.04719755 / 2.0))

        # cx = -100.0, cy = -50.0
        msg = MockDetectionArrayMsg([MockDetection(-100.0, -50.0, score=0.9)])
        reporter.on_detections(msg)

        # In solution.py, negative coordinates are NOT filtered; they project beyond image footprint
        self.assertEqual(len(reporter.clusters), 1)
        c = reporter.clusters[0]
        exp_n = ref_n - (-50.0 - 480.0) * H / focal
        exp_e = ref_e + (-100.0 - 640.0) * H / focal
        self.assertAlmostEqual(c.north, exp_n, places=4)
        self.assertAlmostEqual(c.east, exp_e, places=4)

    def test_pixel_extremes_corners_and_double_resolution(self):
        """Verify (0, 0), (1280, 960), and 2x resolution (2560, 1920)."""
        reporter = self._create_reporter()
        focal = 1280.0 / (2.0 * math.tan(1.04719755 / 2.0))
        H = 16.5

        test_cases = [
            ("Origin (0,0)", 0.0, 0.0, +480.0 * H / focal, -640.0 * H / focal),
            ("Max (1280, 960)", 1280.0, 960.0, -480.0 * H / focal, +640.0 * H / focal),
            ("2x Frame (2560, 1920)", 2560.0, 1920.0, -1440.0 * H / focal, +1920.0 * H / focal),
        ]

        for desc, cx, cy, exp_dn, exp_de in test_cases:
            reporter.clusters.clear()
            reporter.position = (0.0, 0.0, H, 10.0, time.monotonic())
            msg = MockDetectionArrayMsg([MockDetection(cx, cy, score=0.9)])
            reporter.on_detections(msg)
            self.assertEqual(len(reporter.clusters), 1, f"Failed for {desc}")
            c = reporter.clusters[0]
            self.assertAlmostEqual(c.north, exp_dn, places=4, msg=f"North mismatch for {desc}")
            self.assertAlmostEqual(c.east, exp_de, places=4, msg=f"East mismatch for {desc}")

    def test_pixel_extremes_nan_inf_bounding_boxes(self):
        """Bounding box center with NaN or Inf MUST be ignored and produce zero clusters."""
        reporter = self._create_reporter()
        reporter.position = (10.0, 10.0, 16.5, 10.0, time.monotonic())

        cases = [
            (float('nan'), 480.0),
            (640.0, float('nan')),
            (float('inf'), 480.0),
            (640.0, float('-inf')),
        ]
        for cx, cy in cases:
            reporter.clusters.clear()
            msg = MockDetectionArrayMsg([MockDetection(cx, cy, score=0.9)])
            reporter.on_detections(msg)
            self.assertEqual(len(reporter.clusters), 0, f"Non-finite pixel ({cx}, {cy}) was not rejected")

    # =========================================================================
    # DIMENSION 6: Empirical Invariants & Ground Truth Error Bounds
    # =========================================================================
    def test_ground_plane_error_bound_less_than_2_5m(self):
        """Requirement R1: Projected coordinates must produce error <= 2.5m relative to ground geometry.

        Under known ground targets, verify the pinhole model has 0.000m systematic mathematical error.
        """
        reporter = self._create_reporter()
        drone_n, drone_e, H = 25.0, 45.0, 16.5
        reporter.position = (drone_n, drone_e, H, 10.0, time.monotonic())
        focal = 1280.0 / (2.0 * math.tan(1.04719755 / 2.0))

        # Test 100 randomly sampled ground targets across camera footprint
        import random
        random.seed(42)
        max_error = 0.0

        for _ in range(100):
            # Ground victim relative to drone
            gt_dn = random.uniform(-7.144, 7.144)
            gt_de = random.uniform(-9.526, 9.526)
            gt_n = drone_n + gt_dn
            gt_e = drone_e + gt_de

            # Invert projection to get exact pixel coordinate
            cx = 640.0 + (gt_de * focal / H)
            cy = 480.0 - (gt_dn * focal / H)

            reporter.clusters.clear()
            msg = MockDetectionArrayMsg([MockDetection(cx, cy, score=0.9)])
            reporter.on_detections(msg)

            c = reporter.clusters[0]
            err = math.hypot(c.north - gt_n, c.east - gt_e)
            if err > max_error:
                max_error = err

        self.assertLessEqual(max_error, 1e-4, f"Pinhole projection error {max_error} exceeded precision limit")
        self.assertLessEqual(max_error, 2.5, "Failed competition requirement error <= 2.5m")


if __name__ == '__main__':
    unittest.main()
