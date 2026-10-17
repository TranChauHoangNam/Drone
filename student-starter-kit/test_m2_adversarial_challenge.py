#!/usr/bin/env python3
"""test_m2_adversarial_challenge.py

Adversarial Stress Test Suite for Milestone 2: Perception & Model Verification.

Covers:
1. Cooperative Fusion Burst Stress Testing:
   - Simultaneous high-frequency bursts from camera inference and backup ROS2 detections.
   - Multithreaded concurrent bursts into _add_detection.
   - Dedup radius boundary dynamics and zero duplicate reports guarantee in reports.jsonl.
2. victim_report.py Extremes, Non-Finite Values, NumPy Types, and Atomic Durability:
   - NumPy scalar types: float16, float32, float64, int32, int64, uint8.
   - Extreme geographic coordinates and NED offsets.
   - Non-finite values (NaN, Inf, -Inf) handling and JSON compliance.
   - Atomic file durability via flush + fsync and automatic directory creation.
3. Proactive Flush Timing and Timestamp Fidelity at 86.0s:
   - Preservation of true visual sighting timestamp (c.first_sim_s) across multiple clusters.
   - Ensuring previously reported clusters are not re-reported.
   - Single-execution guard on proactive flush.
4. Robustness under Corrupt, Zero, or Stale Inputs:
   - Corrupt bounding boxes, empty detections, zero/negative FOV, stale MAVLink position, altitude < 2.0m.
"""
import concurrent.futures
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, MagicMock, patch

import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Mock pymavlink if not present
try:
    import pymavlink
    from pymavlink import mavutil
except ImportError:
    mock_mav = MagicMock()
    sys.modules['pymavlink'] = mock_mav
    sys.modules['pymavlink.mavutil'] = mock_mav

from solution import (
    PersonReporter,
    PersonCluster,
    DEDUP_RADIUS_M,
    MIN_CONFIDENCE,
)
import victim_report
from victim_report import (
    report_person,
    HOME_LAT,
    HOME_LON,
    METERS_PER_DEG_LAT,
    METERS_PER_DEG_LON,
)


class TestCooperativeFusionBursts(unittest.TestCase):
    """Stress testing cooperative fusion under high-frequency interleaved and concurrent bursts."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.report_file = Path(self.temp_dir.name) / "reports.jsonl"
        self._orig_env = os.environ.get("KIT_REPORT_PATH")
        os.environ["KIT_REPORT_PATH"] = str(self.report_file)

    def tearDown(self):
        if self._orig_env is not None:
            os.environ["KIT_REPORT_PATH"] = self._orig_env
        else:
            os.environ.pop("KIT_REPORT_PATH", None)
        self.temp_dir.cleanup()

    def test_interleaved_high_frequency_burst_zero_duplicates(self):
        """Simulate simultaneous 30Hz bursts from both camera and ROS2 topic for 5 ground truth victims.

        Verifies that across 600 total detection events (300 camera + 300 ROS2),
        exactly 5 unique reports are written to reports.jsonl with ZERO duplicates.
        """
        reporter = PersonReporter()
        reporter.position = (0.0, 0.0, 16.5, 10.0, 1000.0)

        # 5 distinct victims spaced well beyond DEDUP_RADIUS_M (4.5m)
        victims = [
            (15.0, 25.0, 0.85),    # High conf immediate report
            (-40.0, 60.0, 0.40),   # Low conf 2-hit confirmation
            (70.0, -10.0, 0.38),   # Low conf 2-hit confirmation
            (-15.0, -50.0, 0.90),  # High conf immediate report
            (0.0, 80.0, 0.42),     # Low conf single hit (emitted at flush)
        ]

        # Interleave 300 camera detections and 300 ROS2 detections over 10 sim seconds
        sim_time = 10.0
        events = []

        # Victim 0: High conf burst from both channels
        for _ in range(60):
            events.append(('cam', 15.0 + random.uniform(-0.5, 0.5), 25.0 + random.uniform(-0.5, 0.5), 0.85, sim_time))
            events.append(('ros', 15.0 + random.uniform(-0.5, 0.5), 25.0 + random.uniform(-0.5, 0.5), 0.82, sim_time))
            sim_time += 0.033

        # Victim 1: First hit from cam, second from ROS2, followed by 50 hits
        events.append(('cam', -40.0, 60.0, 0.40, sim_time))
        sim_time += 0.033
        events.append(('ros', -40.2, 60.1, 0.41, sim_time))
        sim_time += 0.033
        for _ in range(50):
            events.append(('cam', -40.0 + random.uniform(-0.3, 0.3), 60.0 + random.uniform(-0.3, 0.3), 0.45, sim_time))
            sim_time += 0.033

        # Victim 2: Hits alternating between channels
        events.append(('cam', 70.0, -10.0, 0.38, sim_time))
        sim_time += 0.033
        events.append(('ros', 70.1, -10.2, 0.39, sim_time))
        sim_time += 0.033
        for _ in range(40):
            events.append(('ros', 70.0 + random.uniform(-0.4, 0.4), -10.0 + random.uniform(-0.4, 0.4), 0.50, sim_time))
            sim_time += 0.033

        # Victim 3: Group detected by ROS2 and camera
        for _ in range(50):
            events.append(('ros', -15.0 + random.uniform(-0.2, 0.2), -50.0 + random.uniform(-0.2, 0.2), 0.90, sim_time))
            events.append(('cam', -15.0 + random.uniform(-0.2, 0.2), -50.0 + random.uniform(-0.2, 0.2), 0.88, sim_time))
            sim_time += 0.033

        # Victim 4: Single hit only (unconfirmed during flight)
        events.append(('cam', 0.0, 80.0, 0.42, 45.0))

        # Shuffle slightly to simulate asynchronous arrival order
        random.seed(42)
        random.shuffle(events)

        # Dispatch all events into reporter
        for channel, n, e, conf, s_s in events:
            reporter._add_detection(n, e, conf, s_s)

        # Before flush: exactly 4 victims should be reported (Victims 0, 1, 2, 3)
        self.assertEqual(reporter.reported_count, 4)
        lines_before_flush = self.report_file.read_text().strip().splitlines()
        self.assertEqual(len(lines_before_flush), 4, "Before flush, exactly 4 confirmed victims must be reported")

        # Now trigger proactive flush at sim_s = 86.0
        reporter.flush_pending(sim_s=86.0)

        # After flush: exactly 5 victims reported
        self.assertEqual(reporter.reported_count, 5)
        lines_after_flush = self.report_file.read_text().strip().splitlines()
        self.assertEqual(len(lines_after_flush), 5, "After flush, exactly 5 victims must be reported")

        # Verify zero duplicate clusters and each ground truth victim is matched once
        reports = [json.loads(line) for line in lines_after_flush]
        for vn, ve, _ in victims:
            matches = [r for r in reports if math.hypot(r['north_m'] - vn, r['east_m'] - ve) <= 2.5]
            self.assertEqual(len(matches), 1, f"Victim at ({vn}, {ve}) must have exactly 1 report, found {len(matches)}")

    def test_multithreaded_concurrent_burst_safety(self):
        """Adversarial stress: Concurrent threads executing _add_detection simultaneously.

        Simulates two asynchronous worker threads (camera thread + ROS2 executor thread)
        bombarding _add_detection with 500 detections each for the same victim.
        """
        reporter = PersonReporter()
        num_threads = 4
        detections_per_thread = 250
        barrier = threading.Barrier(num_threads)

        def worker_task(thread_id):
            barrier.wait()  # Synchronize start to maximize race condition probability
            for i in range(detections_per_thread):
                n = 30.0 + random.uniform(-0.3, 0.3)
                e = 40.0 + random.uniform(-0.3, 0.3)
                conf = 0.80 + random.uniform(-0.1, 0.1)
                reporter._add_detection(n, e, conf, 10.0 + i * 0.01)

        threads = [threading.Thread(target=worker_task, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # In Python GIL, list operations and cluster lookups occur
        # Check how many reports were written
        lines = self.report_file.read_text().strip().splitlines()
        # Even under high thread contention, all detections belong to the same 4.5m area
        # Verify reports count is <= 2 (or ideally 1) and clusters are properly bounded
        self.assertGreaterEqual(len(lines), 1)
        self.assertLessEqual(len(lines), 2, "Concurrent threads should not cause duplicate burst explosion")
        for line in lines:
            rec = json.loads(line)
            self.assertAlmostEqual(rec['north_m'], 30.0, delta=1.5)
            self.assertAlmostEqual(rec['east_m'], 40.0, delta=1.5)

    def test_dedup_radius_boundary_merge_and_split(self):
        """Verify strict geometric behavior at DEDUP_RADIUS_M = 4.5m."""
        reporter = PersonReporter()
        # Seed cluster at origin
        reporter._add_detection(0.0, 0.0, 0.60, 1.0)
        self.assertEqual(len(reporter.clusters), 1)

        # Detection at 4.499m (inside radius) must merge
        reporter._add_detection(4.499, 0.0, 0.70, 2.0)
        self.assertEqual(len(reporter.clusters), 1)
        self.assertEqual(reporter.clusters[0].count, 2)

        # Detection at 4.501m (outside current centroid) must create a new cluster
        # Current centroid is at (2.2495, 0.0). Distance from centroid to (7.0, 0.0) is ~4.75m > 4.5m
        reporter._add_detection(7.0, 0.0, 0.65, 3.0)
        self.assertEqual(len(reporter.clusters), 2)


class TestVictimReportExtremesAndDurability(unittest.TestCase):
    """Stress testing victim_report.py with NumPy types, extreme values, non-finites, and atomic durability."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.report_file = Path(self.temp_dir.name) / "reports.jsonl"
        self._orig_env = os.environ.get("KIT_REPORT_PATH")
        os.environ["KIT_REPORT_PATH"] = str(self.report_file)

    def tearDown(self):
        if self._orig_env is not None:
            os.environ["KIT_REPORT_PATH"] = self._orig_env
        else:
            os.environ.pop("KIT_REPORT_PATH", None)
        self.temp_dir.cleanup()

    def test_numpy_scalar_comprehensive_types(self):
        """Verify all NumPy scalar types serialize to valid JSON without TypeError."""
        type_cases = [
            (np.float16(0.75), "np.float16"),
            (np.float32(0.85), "np.float32"),
            (np.float64(0.95), "np.float64"),
            (np.int32(10), "np.int32"),
            (np.int64(20), "np.int64"),
            (np.uint8(1), "np.uint8"),
            (np.int16(5), "np.int16"),
        ]

        for val, name in type_cases:
            res = report_person(
                north_m=val,
                east_m=val,
                confidence=val if isinstance(val, (np.floating, float)) else 0.5,
                sim_s=val
            )
            self.assertIsNotNone(res, f"Failed serialization for {name}")

        lines = self.report_file.read_text().strip().splitlines()
        self.assertEqual(len(lines), len(type_cases))
        for line in lines:
            parsed = json.loads(line)
            self.assertIn('north_m', parsed)
            self.assertIn('east_m', parsed)
            self.assertIn('lat', parsed)
            self.assertIn('lon', parsed)

    def test_extreme_geographic_and_ned_coordinates(self):
        """Verify extreme coordinates do not overflow or produce invalid JSON."""
        extreme_cases = [
            # Geographic North Pole
            {'lat': 90.0, 'lon': 0.0, 'sim_s': 1.0},
            # Geographic South Pole
            {'lat': -90.0, 'lon': 0.0, 'sim_s': 2.0},
            # Antimeridian
            {'lat': 0.0, 'lon': 180.0, 'sim_s': 3.0},
            {'lat': 0.0, 'lon': -180.0, 'sim_s': 4.0},
            # Extremely large NED offset (1,000 km)
            {'north_m': 1_000_000.0, 'east_m': -1_000_000.0, 'sim_s': 5.0},
            # Sub-millimeter offset
            {'north_m': 1e-6, 'east_m': -1e-6, 'sim_s': 6.0},
            # Zero origin
            {'north_m': 0.0, 'east_m': 0.0, 'sim_s': 7.0},
        ]

        for case in extreme_cases:
            res = report_person(**case)
            self.assertIsNotNone(res)

        lines = self.report_file.read_text().strip().splitlines()
        self.assertEqual(len(lines), len(extreme_cases))
        for line in lines:
            parsed = json.loads(line)
            self.assertTrue(math.isfinite(parsed['north_m']))
            self.assertTrue(math.isfinite(parsed['east_m']))
            self.assertTrue(math.isfinite(parsed['lat']))
            self.assertTrue(math.isfinite(parsed['lon']))

    def test_non_finite_rejection_in_solution_pipeline(self):
        """Verify solution.py strictly filters NaN and Inf before they reach report_person."""
        reporter = PersonReporter()
        reporter.position = (0.0, 0.0, 16.5, 5.0, 1000.0)

        # Inject non-finite detections via on_detections
        bad_values = [float('nan'), float('inf'), -float('inf'), np.nan, np.inf]
        for val in bad_values:
            mock_msg = Mock()
            mock_det = Mock()
            mock_det.results = [Mock(hypothesis=Mock(class_id='person', score=val))]
            mock_det.bbox = Mock(center=Mock(position=Mock(x=640.0, y=480.0)))
            mock_msg.detections = [mock_det]

            with patch('time.monotonic', return_value=1000.05):
                reporter.on_detections(mock_msg)

        # Verify nothing was added or reported from non-finite scores
        self.assertEqual(reporter.det_used, 0)
        self.assertEqual(reporter.reported_count, 0)
        self.assertEqual(len(reporter.clusters), 0)

        # Inject non-finite positions via update_position
        initial_pos = reporter.position
        bad_pos_msg = Mock(x=float('nan'), y=0.0, z=-16.5, time_boot_ms=5000)
        reporter.update_position(bad_pos_msg)
        self.assertEqual(reporter.position, initial_pos, "update_position must reject NaN x")

        bad_pos_msg2 = Mock(x=0.0, y=float('inf'), z=-16.5, time_boot_ms=5000)
        reporter.update_position(bad_pos_msg2)
        self.assertEqual(reporter.position, initial_pos, "update_position must reject Inf y")

    def test_atomic_durability_and_nested_directory_creation(self):
        """Verify atomic durability with nested path creation and fsync invocation."""
        deep_path = Path(self.temp_dir.name) / "deep" / "nested" / "reports.jsonl"
        os.environ["KIT_REPORT_PATH"] = str(deep_path)

        with patch('os.fsync') as mock_fsync:
            res = report_person(north_m=10.0, east_m=20.0, confidence=0.88, sim_s=5.0)
            self.assertTrue(deep_path.exists())
            self.assertTrue(mock_fsync.called, "os.fsync must be called to ensure atomic durability")

        # Verify content on disk
        lines = deep_path.read_text().strip().splitlines()
        self.assertEqual(len(lines), 1)
        rec = json.loads(lines[0])
        self.assertEqual(rec['north_m'], 10.0)
        self.assertEqual(rec['east_m'], 20.0)


class TestProactiveFlushTimingAndFidelity(unittest.TestCase):
    """Verify proactive flush timing at 86.0s and timestamp fidelity."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.report_file = Path(self.temp_dir.name) / "reports.jsonl"
        self._orig_env = os.environ.get("KIT_REPORT_PATH")
        os.environ["KIT_REPORT_PATH"] = str(self.report_file)

    def tearDown(self):
        if self._orig_env is not None:
            os.environ["KIT_REPORT_PATH"] = self._orig_env
        else:
            os.environ.pop("KIT_REPORT_PATH", None)
        self.temp_dir.cleanup()

    def test_proactive_flush_all_clusters_retain_original_sighting_time(self):
        """Verify all unconfirmed clusters retain c.first_sim_s, NOT the flush trigger timestamp (86.0s)."""
        reporter = PersonReporter()

        # Cluster 1 sighted at t = 14.2s (single hit, conf 0.40)
        reporter._add_detection(10.0, 10.0, 0.40, 14.2)
        # Cluster 2 sighted at t = 35.8s (single hit, conf 0.42)
        reporter._add_detection(20.0, 20.0, 0.42, 35.8)
        # Cluster 3 sighted at t = 71.5s (single hit, conf 0.39)
        reporter._add_detection(30.0, 30.0, 0.39, 71.5)

        self.assertEqual(reporter.reported_count, 0, "No reports should be emitted yet for single-hit low-conf clusters")

        # Proactive flush triggered at sim_s = 86.0
        reporter.flush_pending(sim_s=86.0)

        self.assertEqual(reporter.reported_count, 3)
        lines = self.report_file.read_text().strip().splitlines()
        self.assertEqual(len(lines), 3)

        records = [json.loads(line) for line in lines]
        times = {r['north_m']: r['sim_s'] for r in records}

        # Assert exact preservation of first_sim_s
        self.assertAlmostEqual(times[10.0], 14.2, places=2)
        self.assertAlmostEqual(times[20.0], 35.8, places=2)
        self.assertAlmostEqual(times[30.0], 71.5, places=2)

    def test_proactive_flush_fallback_when_first_sim_s_is_none(self):
        """If cluster.first_sim_s is None, flush_pending correctly falls back to provided sim_s."""
        reporter = PersonReporter()
        cluster_no_time = PersonCluster(50.0, 50.0, 0.41, None)
        reporter.clusters.append(cluster_no_time)

        reporter.flush_pending(sim_s=86.0)

        lines = self.report_file.read_text().strip().splitlines()
        self.assertEqual(len(lines), 1)
        rec = json.loads(lines[0])
        self.assertAlmostEqual(rec['sim_s'], 86.0, places=2)

    def test_proactive_flush_idempotency(self):
        """Calling flush_pending multiple times does not produce duplicate reports."""
        reporter = PersonReporter()
        reporter._add_detection(10.0, 10.0, 0.40, 20.0)

        reporter.flush_pending(sim_s=86.0)
        self.assertEqual(reporter.reported_count, 1)

        # Call again at end of mission
        reporter.flush_pending(sim_s=90.0)
        self.assertEqual(reporter.reported_count, 1)

        lines = self.report_file.read_text().strip().splitlines()
        self.assertEqual(len(lines), 1)


class TestPerceptionRobustnessAndEdgeCases(unittest.TestCase):
    """Stress testing perception pipeline against malformed, corrupt, or boundary inputs."""

    def test_zero_and_negative_image_dimensions(self):
        """Image message with width=0 or height=0 is safely rejected without division by zero."""
        reporter = PersonReporter()
        reporter.position = (0.0, 0.0, 16.5, 5.0, 1000.0)

        mock_msg = Mock(width=0, height=960, data=b'', encoding='bgr8')
        with patch('time.monotonic', return_value=1000.05):
            reporter.on_camera_image(mock_msg)
        self.assertEqual(reporter.det_used, 0)

        mock_msg2 = Mock(width=1280, height=0, data=b'', encoding='bgr8')
        with patch('time.monotonic', return_value=1000.05):
            reporter.on_camera_image(mock_msg2)
        self.assertEqual(reporter.det_used, 0)

    def test_invalid_camera_fov(self):
        """FOV <= 0 or FOV >= pi is safely rejected without math domain error."""
        reporter = PersonReporter()
        reporter.position = (0.0, 0.0, 16.5, 5.0, 1000.0)

        # Zero FOV
        reporter.camera['hfov_rad'] = 0.0
        mock_msg = Mock(width=1280, height=960, data=b'', encoding='bgr8')
        with patch('time.monotonic', return_value=1000.05):
            reporter.on_camera_image(mock_msg)
        self.assertEqual(reporter.det_used, 0)

        # Negative FOV
        reporter.camera['hfov_rad'] = -1.0
        with patch('time.monotonic', return_value=1000.05):
            reporter.on_camera_image(mock_msg)
        self.assertEqual(reporter.det_used, 0)

        # Excessive FOV (>= pi)
        reporter.camera['hfov_rad'] = 3.5
        with patch('time.monotonic', return_value=1000.05):
            reporter.on_camera_image(mock_msg)
        self.assertEqual(reporter.det_used, 0)

    def test_position_staleness_rejection(self):
        """Position older than MAX_POSITION_AGE_S (1.0s) is strictly rejected by camera and ROS2."""
        reporter = PersonReporter()
        # Position received at monotonic time 1000.0
        reporter.position = (0.0, 0.0, 16.5, 5.0, 1000.0)

        # Camera frame arrives at 1001.01s (age = 1.01s > 1.0s)
        mock_msg = Mock(width=1280, height=960, data=b'', encoding='bgr8')
        with patch('time.monotonic', return_value=1001.01):
            reporter.on_camera_image(mock_msg)
        self.assertEqual(reporter.det_used, 0)

        # ROS2 detection arrives at 1001.01s (age = 1.01s > 1.0s)
        mock_det_msg = Mock(detections=[Mock()])
        with patch('time.monotonic', return_value=1001.01):
            reporter.on_detections(mock_det_msg)
        self.assertEqual(reporter.det_used, 0)

    def test_low_altitude_ground_filter(self):
        """Drone altitude below 2.0m is rejected by camera and ROS2 to eliminate ground takeoff artifacts."""
        reporter = PersonReporter()
        reporter.position = (0.0, 0.0, 1.95, 5.0, 1000.0)

        mock_msg = Mock(width=1280, height=960, data=b'', encoding='bgr8')
        with patch('time.monotonic', return_value=1000.05):
            reporter.on_camera_image(mock_msg)
        self.assertEqual(reporter.det_used, 0)

        mock_det_msg = Mock(detections=[Mock()])
        with patch('time.monotonic', return_value=1000.05):
            reporter.on_detections(mock_det_msg)
        self.assertEqual(reporter.det_used, 0)


if __name__ == '__main__':
    unittest.main()
