#!/usr/bin/env python3
"""
Unit Test Suite for Milestone 2: Perception & Model Verification.

Verifies:
1. YOLO SAR model weights class mapping (class 0/1 dual-class person mapping)
2. Multi-path weights resolution when invoked from external working directories
3. Spatial clustering at 4.5m boundary (4.49m merge vs 4.51m split across 8 compass headings)
4. Two-tier confirmation policy (conf >= 0.45 immediate, conf < 0.45 2-hit confirmation)
5. Polymorphic ROS2 detection parsing (Jazzy Point2D vs Humble direct x/y, hypothesis extraction)
6. Video stream fallback passthrough when YOLO model is None
7. victim_report.py schema, sim_time_s alias, NumPy scalar serialization, and bidirectional GPS math
8. Proactive flush timestamp fidelity (preserving first sighting sim_s)
"""
import ast
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, MagicMock, patch

import numpy as np

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure pymavlink can be imported or mocked if not in environment
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


class TestYoloModelDualClassMapping(unittest.TestCase):
    """1. YOLO model class 0/1 dual-class mapping & synonym coverage."""

    def test_sar_weights_file_and_classes(self):
        weights_path = PROJECT_ROOT / "yolo26s_sar_best.pt"
        self.assertTrue(weights_path.exists(), f"Weights file not found at {weights_path}")
        from ultralytics import YOLO
        model = YOLO(str(weights_path))
        self.assertEqual(model.names[0].lower(), 'person')
        self.assertEqual(model.names[1].lower(), 'person')

    def test_on_camera_image_synonym_filtering(self):
        """Verify on_camera_image accepts person synonyms and rejects vehicles."""
        allowed_names = ['person', '0', '1', 'pedestrian', 'people', 'group', 'groups', 'crowd', 'victim', 'victims']
        disallowed_names = ['car', 'bicycle', 'truck', 'van', 'dog', 'building', 'drone']

        for name in allowed_names:
            reporter = PersonReporter()
            reporter.position = (10.0, 20.0, 16.5, 5.0, 1000.0)
            mock_box = Mock()
            mock_box.cls = [Mock(item=Mock(return_value=0))]
            mock_box.conf = [Mock(item=Mock(return_value=0.85))]
            mock_box.xyxy = [Mock(tolist=Mock(return_value=[630.0, 470.0, 650.0, 490.0]))]

            mock_res = Mock()
            mock_res.plot = Mock(return_value=np.zeros((960, 1280, 3), dtype=np.uint8))
            mock_res.boxes = [mock_box]

            mock_model = Mock()
            mock_model.names = {0: name}
            mock_model.predict = Mock(return_value=[mock_res])
            reporter.yolo_model = mock_model

            mock_msg = Mock(
                width=1280,
                height=960,
                data=np.zeros((960, 1280, 3), dtype=np.uint8).tobytes(),
                encoding='bgr8',
                header=Mock()
            )
            with patch('time.monotonic', return_value=1000.05), patch('solution.report_person'):
                reporter.on_camera_image(mock_msg)

            self.assertEqual(
                reporter.det_used, 1,
                f"Class name '{name}' should be accepted by on_camera_image"
            )

        for name in disallowed_names:
            reporter = PersonReporter()
            reporter.position = (10.0, 20.0, 16.5, 5.0, 1000.0)
            mock_box = Mock()
            mock_box.cls = [Mock(item=Mock(return_value=5))]
            mock_box.conf = [Mock(item=Mock(return_value=0.85))]
            mock_box.xyxy = [Mock(tolist=Mock(return_value=[630.0, 470.0, 650.0, 490.0]))]

            mock_res = Mock()
            mock_res.plot = Mock(return_value=np.zeros((960, 1280, 3), dtype=np.uint8))
            mock_res.boxes = [mock_box]

            mock_model = Mock()
            mock_model.names = {5: name}
            mock_model.predict = Mock(return_value=[mock_res])
            reporter.yolo_model = mock_model

            mock_msg = Mock(
                width=1280,
                height=960,
                data=np.zeros((960, 1280, 3), dtype=np.uint8).tobytes(),
                encoding='bgr8',
                header=Mock()
            )
            with patch('time.monotonic', return_value=1000.05), patch('solution.report_person'):
                reporter.on_camera_image(mock_msg)

            self.assertEqual(
                reporter.det_used, 0,
                f"Disallowed class '{name}' must be filtered out"
            )


class TestMultiPathModelResolution(unittest.TestCase):
    """2. Multi-path weights resolution when invoked from external CWD."""

    def test_resolution_from_external_cwd(self):
        with tempfile.TemporaryDirectory() as external_dir:
            orig_cwd = os.getcwd()
            try:
                os.chdir(external_dir)
                script_dir = str(PROJECT_ROOT)
                weights_env = 'yolo26s_sar_best.pt'
                candidate_paths = [
                    weights_env,
                    os.path.join(script_dir, weights_env),
                    os.path.join('/student', weights_env),
                    os.path.join('/opt/kit', weights_env),
                    os.path.join(script_dir, 'yolo26s_sar_best.pt'),
                ]
                resolved = next((p for p in candidate_paths if os.path.isfile(p)), None)
                self.assertIsNotNone(resolved)
                self.assertTrue(os.path.isfile(resolved))
                self.assertEqual(os.path.abspath(resolved), str(PROJECT_ROOT / "yolo26s_sar_best.pt"))
            finally:
                os.chdir(orig_cwd)

    def test_person_reporter_instantiation_from_external_cwd(self):
        with tempfile.TemporaryDirectory() as external_dir:
            orig_cwd = os.getcwd()
            try:
                os.chdir(external_dir)
                reporter = PersonReporter()
                self.assertIsNotNone(reporter.yolo_model, "YOLO model must load successfully from external CWD")
            finally:
                os.chdir(orig_cwd)


class TestSpatialClusteringBoundary(unittest.TestCase):
    """3. Spatial clustering at 4.5m boundary across multiple headings."""

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

    def test_exact_4_49m_merge_across_8_headings(self):
        """Detections within 4.49m (<= 4.5m) merge into a single cluster across all 8 headings."""
        headings = [
            (1.0, 0.0),                  # North
            (-1.0, 0.0),                 # South
            (0.0, 1.0),                  # East
            (0.0, -1.0),                 # West
            (math.cos(math.pi / 4), math.sin(math.pi / 4)),   # NE
            (math.cos(3 * math.pi / 4), math.sin(3 * math.pi / 4)), # NW
            (-math.cos(math.pi / 4), math.sin(math.pi / 4)),  # SE
            (-math.cos(math.pi / 4), -math.sin(math.pi / 4)), # SW
        ]
        dist = 4.49

        for dn, de in headings:
            reporter = PersonReporter()
            base_n, base_e = 50.0, 50.0
            reporter._add_detection(base_n, base_e, 0.50, 1.0)
            self.assertEqual(len(reporter.clusters), 1)

            offset_n = base_n + dist * dn
            offset_e = base_e + dist * de
            reporter._add_detection(offset_n, offset_e, 0.55, 2.0)

            self.assertEqual(
                len(reporter.clusters), 1,
                f"Detection at distance {dist}m with heading ({dn:.2f}, {de:.2f}) must merge"
            )
            self.assertEqual(reporter.clusters[0].count, 2)
            self.assertAlmostEqual(reporter.clusters[0].north, (base_n + offset_n) / 2.0, places=5)
            self.assertAlmostEqual(reporter.clusters[0].east, (base_e + offset_e) / 2.0, places=5)
            self.assertAlmostEqual(reporter.clusters[0].max_conf, 0.55)

    def test_exact_4_51m_split_across_8_headings(self):
        """Detections at 4.51m (> 4.5m) must split into two distinct clusters across all 8 headings."""
        headings = [
            (1.0, 0.0),                  # North
            (-1.0, 0.0),                 # South
            (0.0, 1.0),                  # East
            (0.0, -1.0),                 # West
            (math.cos(math.pi / 4), math.sin(math.pi / 4)),   # NE
            (math.cos(3 * math.pi / 4), math.sin(3 * math.pi / 4)), # NW
            (-math.cos(math.pi / 4), math.sin(math.pi / 4)),  # SE
            (-math.cos(math.pi / 4), -math.sin(math.pi / 4)), # SW
        ]
        dist = 4.51

        for dn, de in headings:
            reporter = PersonReporter()
            base_n, base_e = 50.0, 50.0
            reporter._add_detection(base_n, base_e, 0.50, 1.0)
            self.assertEqual(len(reporter.clusters), 1)

            offset_n = base_n + dist * dn
            offset_e = base_e + dist * de
            reporter._add_detection(offset_n, offset_e, 0.55, 2.0)

            self.assertEqual(
                len(reporter.clusters), 2,
                f"Detection at distance {dist}m with heading ({dn:.2f}, {de:.2f}) must create a new cluster"
            )


class TestTwoTierConfirmationPolicy(unittest.TestCase):
    """4. Two-tier confirmation policy (conf >= 0.45 immediate, conf < 0.45 multi-hit)."""

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

    def test_immediate_high_confidence_emission(self):
        reporter = PersonReporter()
        reporter._add_detection(25.0, 35.0, 0.45, 10.0)
        self.assertEqual(reporter.reported_count, 1)
        self.assertTrue(reporter.clusters[0].reported)

        # Subsequent hit to same cluster updates centroid but does not re-report
        reporter._add_detection(26.0, 35.5, 0.90, 11.0)
        self.assertEqual(reporter.reported_count, 1)
        self.assertEqual(reporter.clusters[0].count, 2)
        self.assertEqual(reporter.clusters[0].max_conf, 0.90)

    def test_two_hit_confirmation_for_low_confidence(self):
        reporter = PersonReporter()
        # Hit 1: conf 0.40 (< 0.45)
        reporter._add_detection(25.0, 35.0, 0.40, 10.0)
        self.assertEqual(reporter.reported_count, 0)
        self.assertFalse(reporter.clusters[0].reported)

        # Hit 2: conf 0.38 (< 0.45) within 4.5m confirms
        reporter._add_detection(26.0, 35.0, 0.38, 12.0)
        self.assertEqual(reporter.reported_count, 1)
        self.assertTrue(reporter.clusters[0].reported)

        # Hit 3: should not report again
        reporter._add_detection(25.5, 35.0, 0.42, 14.0)
        self.assertEqual(reporter.reported_count, 1)


class TestPolymorphicRos2DetectionParsing(unittest.TestCase):
    """5. Polymorphic ROS2 detection parsing for Jazzy and Humble layouts."""

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

    def test_jazzy_point2d_position_layout(self):
        reporter = PersonReporter()
        reporter.position = (100.0, 200.0, 16.5, 5.0, 1000.0)

        det_msg = Mock()
        det = Mock()
        det.results = [Mock(hypothesis=Mock(class_id='person', score=0.88))]
        # Jazzy layout: bbox.center.position.x / y
        det.bbox = Mock(center=Mock(position=Mock(x=640.0, y=480.0)))
        det_msg.detections = [det]

        with patch('time.monotonic', return_value=1000.05):
            reporter.on_detections(det_msg)

        self.assertEqual(reporter.det_used, 1)
        self.assertEqual(reporter.reported_count, 1)
        self.assertAlmostEqual(reporter.clusters[0].north, 100.0)
        self.assertAlmostEqual(reporter.clusters[0].east, 200.0)

    def test_humble_direct_center_xy_layout(self):
        reporter = PersonReporter()
        reporter.position = (100.0, 200.0, 16.5, 5.0, 1000.0)

        det_msg = Mock()
        det = Mock()
        # Humble / legacy layout: r has score/id directly, center has x/y directly
        r = Mock(spec=['class_id', 'score'])
        r.class_id = 'group'
        r.score = 0.82
        det.results = [r]

        center = Mock(spec=['x', 'y'])
        center.x = 640.0
        center.y = 480.0
        det.bbox = Mock(spec=['center'])
        det.bbox.center = center
        det_msg.detections = [det]

        with patch('time.monotonic', return_value=1000.05):
            reporter.on_detections(det_msg)

        self.assertEqual(reporter.det_used, 1)
        self.assertEqual(reporter.reported_count, 1)
        self.assertAlmostEqual(reporter.clusters[0].north, 100.0)
        self.assertAlmostEqual(reporter.clusters[0].east, 200.0)


class TestFallbackVideoPassthrough(unittest.TestCase):
    """6. Fallback video passthrough when YOLO is None."""

    def test_passthrough_when_yolo_none(self):
        reporter = PersonReporter()
        reporter.yolo_model = None
        reporter.position = (10.0, 20.0, 16.5, 5.0, 1000.0)

        mock_pub1 = Mock()
        mock_pub2 = Mock()
        reporter.pub_annotated = mock_pub1
        reporter.pub_cam_annotated = mock_pub2

        raw_msg = Mock()
        with patch('time.monotonic', return_value=1000.05):
            reporter.on_camera_image(raw_msg)

        mock_pub1.publish.assert_called_once_with(raw_msg)
        mock_pub2.publish.assert_called_once_with(raw_msg)


class TestVictimReporterSchemaAndGpsMath(unittest.TestCase):
    """7. victim_report.py schema, sim_time_s alias, NumPy scalars, and bidirectional GPS."""

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

    def test_sim_time_s_alias(self):
        res = report_person(north_m=12.0, east_m=34.0, sim_time_s=42.5, confidence=0.88)
        self.assertEqual(res['sim_s'], 42.5)
        lines = self.report_file.read_text().strip().splitlines()
        self.assertEqual(len(lines), 1)
        saved = json.loads(lines[0])
        self.assertEqual(saved['sim_s'], 42.5)

    def test_numpy_scalar_serialization_safety(self):
        conf_np32 = np.float32(0.875)
        conf_np64 = np.float64(0.925)
        north_np = np.float64(15.2)

        # Must not raise TypeError: Object of type float32 is not JSON serializable
        res = report_person(north_m=north_np, east_m=20.0, confidence=conf_np32, sim_s=np.float32(10.0))
        self.assertAlmostEqual(res['confidence'], 0.875, places=3)

        lines = self.report_file.read_text().strip().splitlines()
        saved = json.loads(lines[-1])
        self.assertAlmostEqual(saved['confidence'], 0.875, places=3)

    def test_bidirectional_gps_conversion(self):
        # Home position conversion (0, 0) -> HOME_LAT, HOME_LON
        res_origin = report_person(north_m=0.0, east_m=0.0, sim_s=1.0)
        self.assertAlmostEqual(res_origin['lat'], HOME_LAT, places=6)
        self.assertAlmostEqual(res_origin['lon'], HOME_LON, places=6)

        # GPS -> NED conversion
        res_ned = report_person(lat=HOME_LAT, lon=HOME_LON, sim_s=2.0)
        self.assertAlmostEqual(res_ned['north_m'], 0.0, places=4)
        self.assertAlmostEqual(res_ned['east_m'], 0.0, places=4)

        # Non-zero offset round trip
        test_n, test_e = 75.5, -45.2
        res1 = report_person(north_m=test_n, east_m=test_e, sim_s=3.0)
        lat_conv, lon_conv = res1['lat'], res1['lon']

        res2 = report_person(lat=lat_conv, lon=lon_conv, sim_s=4.0)
        self.assertAlmostEqual(res2['north_m'], test_n, places=4)
        self.assertAlmostEqual(res2['east_m'], test_e, places=4)


class TestProactiveFlushTimestampFidelity(unittest.TestCase):
    """8. Proactive flush timestamp fidelity (preserving sighting sim_s)."""

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

    def test_flush_preserves_first_sim_s(self):
        reporter = PersonReporter()
        # Detection at sim_s = 23.4 with moderate confidence (unreported)
        reporter._add_detection(40.0, 60.0, 0.40, 23.4)
        self.assertEqual(reporter.reported_count, 0)
        self.assertEqual(reporter.clusters[0].first_sim_s, 23.4)

        # Trigger proactive flush at sim_s = 86.0
        reporter.flush_pending(sim_s=86.0)

        self.assertEqual(reporter.reported_count, 1)
        lines = self.report_file.read_text().strip().splitlines()
        self.assertEqual(len(lines), 1)
        saved = json.loads(lines[0])
        # Crucial: timestamp must be 23.4, not 86.0
        self.assertAlmostEqual(saved['sim_s'], 23.4, places=2)


if __name__ == '__main__':
    unittest.main()
