#!/usr/bin/env python3
"""
test_m2_challenger_harness.py — Empirical Challenger Harness for Milestone 2: Perception & Model Verification.

Adversarially probes:
1. Spatial deduplication at the exact 4.5m boundary (4.49m merge vs 4.51m split across 16 headings).
2. Centroid chaining stress tests and multi-hit inertia defense against over-merging.
3. Multi-path YOLO model resolution across arbitrary working directories (/tmp, /dev/shm, /var/tmp, nested dirs).
4. Fuzz testing of on_detections with malformed Detection2DArray messages.
5. Sighting timestamp fidelity and two-tier confirmation under adversarial burst sequences.
"""
import ast
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np

# Ensure pymavlink is available or mocked
try:
    import pymavlink
    from pymavlink import mavutil
except ImportError:
    mock_mav = MagicMock()
    sys.modules['pymavlink'] = mock_mav
    sys.modules['pymavlink.mavutil'] = mock_mav

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import solution
from solution import (
    PersonCluster,
    PersonReporter,
    DEDUP_RADIUS_M,
    MIN_CONFIDENCE,
)
import victim_report
from victim_report import report_person


# Concrete ROS2 Message Mocks avoiding dynamic Mock() attribute synthesis
class MockRos2Point2D:
    def __init__(self, x=640.0, y=480.0):
        self.x = x
        self.y = y


class MockRos2Pose2D:
    def __init__(self, x=640.0, y=480.0):
        self.position = MockRos2Point2D(x, y)


class MockRos2CenterDirect:
    def __init__(self, x=640.0, y=480.0):
        self.x = x
        self.y = y


class MockRos2BBoxJazzy:
    def __init__(self, x=640.0, y=480.0):
        self.center = MockRos2Pose2D(x, y)


class MockRos2BBoxHumble:
    def __init__(self, x=640.0, y=480.0):
        self.center = MockRos2CenterDirect(x, y)


class MockRos2Hypothesis:
    def __init__(self, class_id='person', score=0.85):
        self.class_id = class_id
        self.score = score


class MockRos2ResultWithHypo:
    def __init__(self, class_id='person', score=0.85):
        self.hypothesis = MockRos2Hypothesis(class_id, score)


class MockRos2ResultDirect:
    def __init__(self, class_id='person', score=0.85):
        self.class_id = class_id
        self.score = score


class MockRos2Detection:
    def __init__(self, results=None, bbox=None):
        self.results = results if results is not None else [MockRos2ResultWithHypo()]
        self.bbox = bbox if bbox is not None else MockRos2BBoxJazzy()


class MockRos2DetectionArray:
    def __init__(self, detections=None):
        self.detections = detections if detections is not None else []


class TestExactBoundarySpatialDeduplication(unittest.TestCase):
    """Mission Item 1: Empirically probe spatial deduplication at exact 4.5m boundary."""

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

    def test_exact_4_49m_merge_across_16_headings(self):
        """At 4.49m (< 4.50m), detections MUST merge into a single cluster across all 16 headings."""
        num_headings = 16
        headings = [
            (math.cos(2 * math.pi * i / num_headings), math.sin(2 * math.pi * i / num_headings))
            for i in range(num_headings)
        ]
        dist = 4.49

        for dn, de in headings:
            reporter = PersonReporter()
            base_n, base_e = 50.0, 50.0
            reporter._add_detection(base_n, base_e, 0.50, 1.0)
            self.assertEqual(len(reporter.clusters), 1)

            offset_n = base_n + dist * dn
            offset_e = base_e + dist * de
            reporter._add_detection(offset_n, offset_e, 0.60, 2.0)

            self.assertEqual(
                len(reporter.clusters), 1,
                f"Dist {dist}m at heading ({dn:.3f}, {de:.3f}) MUST merge into 1 cluster"
            )
            self.assertEqual(reporter.clusters[0].count, 2)
            self.assertAlmostEqual(reporter.clusters[0].north, (base_n + offset_n) / 2.0, places=4)
            self.assertAlmostEqual(reporter.clusters[0].east, (base_e + offset_e) / 2.0, places=4)
            self.assertEqual(reporter.clusters[0].max_conf, 0.60)

    def test_exact_4_51m_split_across_16_headings(self):
        """At 4.51m (> 4.50m), detections MUST separate into two clusters across all 16 headings."""
        num_headings = 16
        headings = [
            (math.cos(2 * math.pi * i / num_headings), math.sin(2 * math.pi * i / num_headings))
            for i in range(num_headings)
        ]
        dist = 4.51

        for dn, de in headings:
            reporter = PersonReporter()
            base_n, base_e = 50.0, 50.0
            reporter._add_detection(base_n, base_e, 0.50, 1.0)
            self.assertEqual(len(reporter.clusters), 1)

            offset_n = base_n + dist * dn
            offset_e = base_e + dist * de
            reporter._add_detection(offset_n, offset_e, 0.60, 2.0)

            self.assertEqual(
                len(reporter.clusters), 2,
                f"Dist {dist}m at heading ({dn:.3f}, {de:.3f}) MUST create 2 separate clusters"
            )
            self.assertEqual(reporter.clusters[0].count, 1)
            self.assertEqual(reporter.clusters[1].count, 1)

    def test_high_precision_boundary_delta(self):
        """Micro-delta testing: 4.499m (inside) vs 4.501m (outside)."""
        reporter = PersonReporter()
        reporter._add_detection(0.0, 0.0, 0.50, 1.0)
        reporter._add_detection(4.499, 0.0, 0.50, 2.0)
        self.assertEqual(len(reporter.clusters), 1, "4.499m MUST merge")

        # Now test 4.501m against fresh cluster
        reporter2 = PersonReporter()
        reporter2._add_detection(0.0, 0.0, 0.50, 1.0)
        reporter2._add_detection(4.501, 0.0, 0.50, 2.0)
        self.assertEqual(len(reporter2.clusters), 2, "4.501m MUST separate")


class TestCentroidChainingAndInertia(unittest.TestCase):
    """Mission Item 2: Stress test centroid chaining and multi-hit inertia."""

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

    def test_single_hit_unanchored_drift_vulnerability(self):
        """Demonstrate that WITHOUT multi-hit inertia, single sequential hits spaced 3.0m apart could walk.

        This test empirically validates why multi-hit inertia is essential in real flight.
        """
        reporter = PersonReporter()
        # Hit 1 at (0, 0)
        reporter._add_detection(0.0, 0.0, 0.50, 1.0)
        self.assertEqual(reporter.clusters[0].north, 0.0)

        # Hit 2 at (3.0, 0.0): distance 3.0 <= 4.5 -> merges, new centroid is (1.5, 0.0)
        reporter._add_detection(3.0, 0.0, 0.50, 2.0)
        self.assertEqual(reporter.clusters[0].north, 1.5)

        # Hit 3 at (5.5, 0.0): distance from original (0,0) is 5.5m (> 4.5m), but from centroid (1.5, 0) is 4.0m <= 4.5m!
        # It merges because centroid shifted!
        reporter._add_detection(5.5, 0.0, 0.50, 3.0)
        self.assertEqual(len(reporter.clusters), 1)

    def test_multi_hit_inertia_prevents_overmerging_separate_victims(self):
        """Verify that when a real victim has multi-hit inertia (5 hits), adjacent victim at 6.0m CANNOT merge."""
        reporter = PersonReporter()

        # 5 hits on Victim 1 at (0, 0) representing a real drone pass over target
        v1_detections = [
            (0.0, 0.0),
            (0.1, -0.1),
            (-0.1, 0.1),
            (0.1, 0.1),
            (-0.1, -0.1),
        ]
        for idx, (n, e) in enumerate(v1_detections):
            reporter._add_detection(n, e, 0.70, 5.0 + idx * 0.1)

        self.assertEqual(len(reporter.clusters), 1)
        self.assertEqual(reporter.clusters[0].count, 5)
        self.assertAlmostEqual(reporter.clusters[0].north, 0.0, places=3)
        self.assertAlmostEqual(reporter.clusters[0].east, 0.0, places=3)

        # Now simulate an intermediate noise detection at (3.5, 0.0)
        reporter._add_detection(3.5, 0.0, 0.40, 5.8)
        self.assertEqual(reporter.clusters[0].count, 6)
        # With 5-hit inertia, new centroid is (0.0 * 5 + 3.5) / 6 = 0.583m
        self.assertAlmostEqual(reporter.clusters[0].north, 3.5 / 6.0, places=3)

        # Now Victim 2 arrives at (6.0, 0.0)
        # Distance from Victim 2 to Cluster 1 centroid is 6.0 - 0.583 = 5.417m > 4.5m!
        # It CANNOT merge into Victim 1!
        reporter._add_detection(6.0, 0.0, 0.75, 6.5)
        self.assertEqual(
            len(reporter.clusters), 2,
            "Multi-hit inertia MUST prevent Victim 2 at 6.0m from merging into Victim 1"
        )
        self.assertEqual(reporter.reported_count, 2)

    def test_dense_5_victim_sweep_isolation(self):
        """Simulate 5 realistic victim locations across a lawnmower sweep with multiple hits each."""
        reporter = PersonReporter()

        # 5 distinct victims spaced 6.0m to 15.0m apart
        victims_ground_truth = [
            (10.0, 10.0),
            (16.5, 10.0),   # 6.5m North of V1
            (16.5, 17.0),   # 7.0m East of V2
            (25.0, 25.0),   # ~11.7m from V3
            (35.0, 40.0),   # ~18m from V4
        ]

        np.random.seed(1337)
        sim_time = 10.0
        for v_idx, (vn, ve) in enumerate(victims_ground_truth):
            # 6 detections per victim with gaussian noise sigma=0.35m
            for h in range(6):
                noise_n = float(np.random.normal(0, 0.35))
                noise_e = float(np.random.normal(0, 0.35))
                conf = 0.50 + float(np.random.uniform(0.1, 0.35))
                reporter._add_detection(vn + noise_n, ve + noise_e, conf, sim_time)
                sim_time += 0.1
            sim_time += 2.0

        self.assertEqual(
            len(reporter.clusters), 5,
            f"Expected exactly 5 clusters for 5 distinct victims, got {len(reporter.clusters)}"
        )
        self.assertEqual(
            reporter.reported_count, 5,
            f"Expected exactly 5 reported victims, got {reporter.reported_count}"
        )

        for v_idx, (vn, ve) in enumerate(victims_ground_truth):
            min_dist = min(
                math.hypot(c.north - vn, c.east - ve)
                for c in reporter.clusters
            )
            self.assertLess(min_dist, 0.50, f"Victim {v_idx} error must be < 0.50m")


class TestMultiPathYoloResolution(unittest.TestCase):
    """Mission Item 3: Test multi-path YOLO model resolution from arbitrary CWDs."""

    def test_resolution_from_arbitrary_cwds(self):
        candidate_cwds = ['/tmp', '/dev/shm', '/var/tmp']
        orig_cwd = os.getcwd()

        for cwd in candidate_cwds:
            if not os.path.isdir(cwd):
                continue
            try:
                os.chdir(cwd)
                reporter = PersonReporter()
                self.assertIsNotNone(
                    reporter.yolo_model,
                    f"YOLO model failed to resolve when CWD={cwd}"
                )
            finally:
                os.chdir(orig_cwd)

    def test_resolution_from_nested_temporary_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            orig_cwd = os.getcwd()
            try:
                nested_dir = Path(temp_dir) / "sandbox" / "evaluation" / "harness"
                nested_dir.mkdir(parents=True)
                os.chdir(str(nested_dir))

                reporter = PersonReporter()
                self.assertIsNotNone(reporter.yolo_model, "Model must resolve from nested temp dir")

                # Test forward pass on dummy BGR image
                dummy_img = np.zeros((960, 1280, 3), dtype=np.uint8)
                res = reporter.yolo_model.predict(dummy_img, conf=MIN_CONFIDENCE, verbose=False)
                self.assertIsNotNone(res)
                self.assertTrue(len(res) > 0)
            finally:
                os.chdir(orig_cwd)


class TestFuzzOnDetectionsMalformedMessages(unittest.TestCase):
    """Mission Item 4: Fuzz test on_detections with malformed Detection2DArray messages."""

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

    def test_fuzz_missing_attributes_and_none_fields(self):
        reporter = PersonReporter()
        reporter.position = (50.0, 50.0, 16.5, 10.0, 1000.0)

        with patch('time.monotonic', return_value=1000.05):
            # Message is None, empty, or has corrupted detections list
            reporter.on_detections(None)
            reporter.on_detections(MockRos2DetectionArray(detections=None))
            reporter.on_detections(MockRos2DetectionArray(detections=[]))
            reporter.on_detections(MockRos2DetectionArray(detections=[None]))
            reporter.on_detections(MockRos2DetectionArray(detections=["corrupt_string_item"]))

        self.assertEqual(len(reporter.clusters), 0)

        # Detection missing bbox (falls back to nadir position)
        det_no_bbox = MockRos2Detection(
            results=[MockRos2ResultWithHypo(class_id='person', score=0.90)],
            bbox=None
        )

        # Detection with results=None
        det_no_results = MockRos2Detection(
            results=None,
            bbox=MockRos2BBoxJazzy(640.0, 480.0)
        )

        # Detection with None elements inside results
        det_results_with_nones = MockRos2Detection(
            results=[None, MockRos2ResultWithHypo(class_id='car', score=0.95)],
            bbox=MockRos2BBoxJazzy(640.0, 480.0)
        )

        with patch('time.monotonic', return_value=1000.05):
            reporter.on_detections(MockRos2DetectionArray(
                detections=[det_no_bbox, det_no_results, det_results_with_nones]
            ))

        # Only det_no_bbox has a valid person hypothesis and gets clustered at nadir
        self.assertEqual(len(reporter.clusters), 1)
        self.assertEqual(reporter.clusters[0].north, 50.0)
        self.assertEqual(reporter.clusters[0].east, 50.0)

    def test_fuzz_string_scores_and_extreme_values(self):
        reporter = PersonReporter()
        reporter.position = (50.0, 50.0, 16.5, 10.0, 1000.0)

        test_cases = [
            ("0.85", True),           # String float parsable -> accepted
            ("not_a_number", False),  # ValueError -> rejected
            ("NaN", False),           # Non-finite -> rejected
            (float('nan'), False),
            (float('inf'), False),
            (-float('inf'), False),
            (None, False),
            (-0.5, False),            # Negative -> below MIN_CONFIDENCE
            (0.10, False),            # Below MIN_CONFIDENCE
            (1.5, True),              # > 1.0 -> finite and >= MIN_CONFIDENCE
        ]

        with patch('time.monotonic', return_value=1000.05):
            for score_val, expected_accepted in test_cases:
                det = MockRos2Detection(
                    results=[MockRos2ResultWithHypo(class_id='person', score=score_val)],
                    bbox=MockRos2BBoxHumble(640.0, 480.0)
                )
                prev_used = reporter.det_used
                reporter.on_detections(MockRos2DetectionArray(detections=[det]))
                if expected_accepted:
                    self.assertEqual(reporter.det_used, prev_used + 1, f"Score {score_val} should be accepted")
                else:
                    self.assertEqual(reporter.det_used, prev_used, f"Score {score_val} should be rejected")

    def test_fuzz_negative_and_extreme_bounding_boxes(self):
        """Bounding boxes outside frame (negative coordinates, extreme pixels, NaNs)."""
        reporter = PersonReporter()
        reporter.position = (100.0, 100.0, 16.5, 10.0, 1000.0)

        det_negative = MockRos2Detection(
            results=[MockRos2ResultWithHypo(class_id='person', score=0.80)],
            bbox=MockRos2BBoxHumble(x=-200.0, y=-300.0)
        )
        det_extreme = MockRos2Detection(
            results=[MockRos2ResultWithHypo(class_id='person', score=0.80)],
            bbox=MockRos2BBoxHumble(x=5000.0, y=6000.0)
        )
        det_nan = MockRos2Detection(
            results=[MockRos2ResultWithHypo(class_id='person', score=0.80)],
            bbox=MockRos2BBoxHumble(x=float('nan'), y=480.0)
        )

        with patch('time.monotonic', return_value=1000.05):
            reporter.on_detections(MockRos2DetectionArray(detections=[det_negative, det_extreme, det_nan]))

        # det_seen and det_used count accepted hypotheses (3 total)
        # but only the 2 finite detections are clustered, while NaN coordinate is rejected
        self.assertEqual(reporter.det_used, 3)
        self.assertEqual(len(reporter.clusters), 2)
        self.assertTrue(all(math.isfinite(c.north) and math.isfinite(c.east) for c in reporter.clusters))

    def test_fuzz_burst_stream_100_random_corrupted_messages(self):
        """High-throughput fuzzing: 100 random malformed messages in rapid succession."""
        reporter = PersonReporter()
        reporter.position = (0.0, 0.0, 16.5, 10.0, 1000.0)

        random.seed(999)
        with patch('time.monotonic', return_value=1000.05):
            for _ in range(100):
                # Randomly mutate fields
                cid = random.choice(['person', 'car', 'group', None, 0, 1, 'invalid'])
                score = random.choice([0.8, -1.0, '0.7', 'bad', float('nan'), None, 0.1])
                x = random.choice([640.0, -100.0, 2000.0, float('nan'), float('inf')])
                y = random.choice([480.0, -50.0, 1500.0, float('nan')])

                det = MockRos2Detection(
                    results=[MockRos2ResultWithHypo(class_id=cid, score=score)],
                    bbox=MockRos2BBoxHumble(x=x, y=y)
                )
                # Must not raise any unhandled exceptions
                try:
                    reporter.on_detections(MockRos2DetectionArray(detections=[det]))
                except Exception as e:
                    self.fail(f"on_detections raised unhandled exception on fuzz input: {e}")


if __name__ == '__main__':
    unittest.main()
