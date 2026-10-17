#!/usr/bin/env python3
"""
CHALLENGE_M1_EMPIRICS.PY
Adversarial empirical testing harness for Milestone 1:
1. Monte Carlo simulation of drone flight with pitch/roll attitude dynamics and ground plane error verification.
2. Missing / corrupted / abnormal /opt/kit/camera.json fallback behavior in both on_camera_image and on_detections.
3. High-frequency concurrency, race condition, and telemetry latency gate stress testing.
4. PersonCluster boundary stability (4.49m vs 4.51m), centroid chaining, and disk-report freezing audit.
"""

import math
import os
import sys
import time
import json
import random
import threading
from pathlib import Path
from unittest.mock import Mock, patch

# Ensure student-starter-kit is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import solution
from solution import PersonCluster, PersonReporter, DEDUP_RADIUS_M, ALTITUDE, SPEED


# ==============================================================================
# 1. MONTE CARLO SIMULATION: GROUND PLANE ERROR BOUND
# ==============================================================================

def run_monte_carlo_ground_plane(num_samples=10000, seed=42):
    """
    Monte Carlo simulation of drone flight at nominal speed 5.2 m/s at 16.5m altitude.
    Evaluates:
      1. Nominal Lawnmower Flight (yaw locked to 0.0 rad, as enforced by solution.py setpoint stream):
         - Flying North: forward pitch theta ~ -3.5 deg to -5.5 deg
         - Flying South: backward pitch theta ~ +3.5 deg to +5.5 deg
         - Flying East (track transition): right roll phi ~ +3.5 deg to +5.5 deg
         - Attitude jitter: pitch/roll noise ~ N(0, 1.2 deg)
         - Telemetry latency jitter: tau in [0.02s, 0.08s] (20Hz MAVLink stream)
         - Target pixels cx, cy uniformly sampled across 1280x960 image sensor
      2. Sensitivity Analysis:
         - Tilt sensitivity (1 deg to 10 deg)
         - Telemetry latency sensitivity (0.0s to 1.0s)
         - Yaw drift sensitivity (0 deg to 30 deg)
    """
    random.seed(seed)
    
    width = 1280.0
    height = 960.0
    hfov_rad = 1.04719755  # 60 degrees
    focal = width / (2.0 * math.tan(hfov_rad / 2.0))
    
    errors_nominal = []
    errors_with_yaw_drift = []
    projection_only_errors = []
    latency_errors = []
    
    for _ in range(num_samples):
        alt = random.uniform(15.5, 17.5)
        drone_n = random.uniform(15.0, 85.0)
        drone_e = random.uniform(15.0, 85.0)
        
        # Lawnmower flight legs (North sweeps 70%, East step 15%, South sweeps 15%)
        leg = random.choices(['NORTH', 'SOUTH', 'EAST'], weights=[0.45, 0.40, 0.15])[0]
        
        if leg == 'NORTH':
            # Drone moving North, yaw = 0, pitches down
            v_n = SPEED
            v_e = 0.0
            pitch_nominal = -math.radians(random.uniform(3.5, 5.2))
            roll_nominal = 0.0
        elif leg == 'SOUTH':
            # Drone moving South, yaw = 0, pitches up
            v_n = -SPEED
            v_e = 0.0
            pitch_nominal = math.radians(random.uniform(3.5, 5.2))
            roll_nominal = 0.0
        else: # EAST
            # Drone transitioning East, yaw = 0, rolls right
            v_n = 0.0
            v_e = SPEED
            pitch_nominal = 0.0
            roll_nominal = math.radians(random.uniform(3.5, 5.2))
            
        # Attitude oscillations / turbulence
        pitch_body = pitch_nominal + random.gauss(0, math.radians(1.0))
        roll_body = roll_nominal + random.gauss(0, math.radians(1.0))
        yaw_nominal = 0.0
        
        # Target pixel on sensor
        cx = random.uniform(40.0, width - 40.0)
        cy = random.uniform(40.0, height - 40.0)
        
        # Telemetry latency: 20Hz MAVLink stream has jitter 0.02s to 0.08s
        dt_lag = random.uniform(0.02, 0.08)
        true_drone_n = drone_n - v_n * dt_lag
        true_drone_e = drone_e - v_e * dt_lag
        
        # Ray in camera frame
        r_cam = [cx - width / 2.0, cy - height / 2.0, focal]
        norm = math.hypot(r_cam[0], r_cam[1], r_cam[2])
        r_cam = [r_cam[0]/norm, r_cam[1]/norm, r_cam[2]/norm]
        
        # Camera to body: X_cam -> Y_b, Y_cam -> -X_b, Z_cam -> Z_b
        r_body = [-r_cam[1], r_cam[0], r_cam[2]]
        
        # 1. NOMINAL CASE (yaw = 0)
        cr, sr = math.cos(roll_body), math.sin(roll_body)
        cp, sp = math.cos(pitch_body), math.sin(pitch_body)
        # R = Ry(pitch) * Rx(roll) when yaw = 0
        R_nom = [
            [cp, sp*sr, sp*cr],
            [0,  cr,    -sr],
            [-sp, cp*sr, cp*cr]
        ]
        r_ned = [
            R_nom[0][0]*r_body[0] + R_nom[0][1]*r_body[1] + R_nom[0][2]*r_body[2],
            R_nom[1][0]*r_body[0] + R_nom[1][1]*r_body[1] + R_nom[1][2]*r_body[2],
            R_nom[2][0]*r_body[0] + R_nom[2][1]*r_body[1] + R_nom[2][2]*r_body[2]
        ]
        
        if r_ned[2] > 0.05:
            lamb = alt / r_ned[2]
            n_true = true_drone_n + lamb * r_ned[0]
            e_true = true_drone_e + lamb * r_ned[1]
            
            # solution.py nadir projection
            n_est = drone_n - (cy - height / 2.0) * alt / focal
            e_est = drone_e + (cx - width / 2.0) * alt / focal
            
            err = math.hypot(n_est - n_true, e_est - e_true)
            errors_nominal.append(err)
            
            # Projection only (no lag)
            n_est_no_lag = true_drone_n - (cy - height / 2.0) * alt / focal
            e_est_no_lag = true_drone_e + (cx - width / 2.0) * alt / focal
            proj_err = math.hypot(n_est_no_lag - n_true, e_est_no_lag - e_true)
            projection_only_errors.append(proj_err)
            latency_errors.append(math.hypot(drone_n - true_drone_n, drone_e - true_drone_e))

        # 2. YAW DRIFT CASE (yaw drift ~ 10 deg)
        yaw_drift = math.radians(random.uniform(5.0, 15.0))
        cyaw, syaw = math.cos(yaw_drift), math.sin(yaw_drift)
        R_drift = [
            [cyaw*cp, cyaw*sp*sr - syaw*cr, cyaw*sp*cr + syaw*sr],
            [syaw*cp, syaw*sp*sr + cyaw*cr, syaw*sp*cr - cyaw*sr],
            [-sp,     cp*sr,                cp*cr]
        ]
        r_ned_drift = [
            R_drift[0][0]*r_body[0] + R_drift[0][1]*r_body[1] + R_drift[0][2]*r_body[2],
            R_drift[1][0]*r_body[0] + R_drift[1][1]*r_body[1] + R_drift[1][2]*r_body[2],
            R_drift[2][0]*r_body[0] + R_drift[2][1]*r_body[1] + R_drift[2][2]*r_body[2]
        ]
        if r_ned_drift[2] > 0.05:
            lamb_d = alt / r_ned_drift[2]
            n_true_d = true_drone_n + lamb_d * r_ned_drift[0]
            e_true_d = true_drone_e + lamb_d * r_ned_drift[1]
            err_d = math.hypot(n_est - n_true_d, e_est - e_true_d)
            errors_with_yaw_drift.append(err_d)

    errors_nominal.sort()
    n_count = len(errors_nominal)
    mean_err = sum(errors_nominal) / n_count
    std_err = math.sqrt(sum((x - mean_err)**2 for x in errors_nominal) / n_count)
    p95_err = errors_nominal[int(n_count * 0.95)]
    max_err = errors_nominal[-1]
    pass_rate = sum(1 for e in errors_nominal if e <= 2.5) / n_count * 100.0
    
    # Latency sensitivity analysis
    # If telemetry lag is tau, lag error = speed * tau.
    # When combined with mean projection error, find critical latency tau_crit where total error reaches 2.5m
    mean_proj = sum(projection_only_errors) / n_count
    mean_lag = sum(latency_errors) / n_count
    tau_crit = max(0.0, (2.5 - mean_proj) / SPEED)
    
    mean_yaw_drift_err = sum(errors_with_yaw_drift) / len(errors_with_yaw_drift)
    
    return {
        'num_samples': n_count,
        'mean_error': mean_err,
        'std_error': std_err,
        'p95_error': p95_err,
        'max_error': max_err,
        'pass_rate_2_5m': pass_rate,
        'mean_projection_error': mean_proj,
        'mean_latency_error': mean_lag,
        'critical_latency_tau_s': tau_crit,
        'mean_yaw_drift_error_10deg': mean_yaw_drift_err
    }


# ==============================================================================
# 2. CAMERA INTRINSICS INGESTION & FALLBACK RESILIENCE
# ==============================================================================

def test_camera_intrinsics_fallbacks():
    """
    Tests behavior of PersonReporter under various /opt/kit/camera.json conditions:
    1. File missing (FileNotFoundError)
    2. File empty or corrupted JSON syntax
    3. File containing non-dict JSON (null, array, scalar)
    4. Partial dict with missing keys
    5. Discrepancy between camera.json resolution and incoming ROS2 Image msg
    6. Non-nadir pitch_deg handling difference in on_camera_image vs on_detections
    """
    import io
    results = {}
    orig_open = open
    
    def make_open_mock(content_str=None, exc=None):
        def _mock_open(file, *args, **kwargs):
            if str(file) == '/opt/kit/camera.json':
                if exc:
                    raise exc
                return io.StringIO(content_str)
            return orig_open(file, *args, **kwargs)
        return _mock_open
    
    with patch('solution.report_person'), patch('victim_report.report_person'):
        # Test 2.1: Missing file fallback
        with patch('builtins.open', side_effect=make_open_mock(exc=FileNotFoundError())):
            with patch.dict('sys.modules', {'rclpy': Mock(), 'rclpy.context': Mock(), 'rclpy.executors': Mock()}):
                reporter = PersonReporter()
                results['missing_file_defaults'] = (
                    reporter.camera['hfov_rad'] == 1.04719755 and
                    reporter.camera['width'] == 1280 and
                    reporter.camera['height'] == 960 and
                    reporter.camera['pitch_deg'] == 90.0
                )
                
        # Test 2.2: Corrupted JSON syntax
        with patch('builtins.open', side_effect=make_open_mock(content_str="{ invalid json }")):
            with patch.dict('sys.modules', {'rclpy': Mock(), 'rclpy.context': Mock(), 'rclpy.executors': Mock()}):
                reporter = PersonReporter()
                results['corrupt_json_fallback'] = (
                    reporter.camera['hfov_rad'] == 1.04719755 and
                    reporter.camera['width'] == 1280
                )

        # Test 2.3: Non-dict JSON: null (None)
        with patch('builtins.open', side_effect=make_open_mock(content_str="null")):
            with patch.dict('sys.modules', {'rclpy': Mock(), 'rclpy.context': Mock(), 'rclpy.executors': Mock()}):
                null_handled = False
                try:
                    reporter = PersonReporter()
                    null_handled = True
                except TypeError:
                    null_handled = False
                results['null_json_handled'] = null_handled

        # Test 2.4: Partial JSON update
        partial_json = json.dumps({'hfov_rad': 1.5707963, 'pitch_deg': 85.0})
        with patch('builtins.open', side_effect=make_open_mock(content_str=partial_json)):
            with patch.dict('sys.modules', {'rclpy': Mock(), 'rclpy.context': Mock(), 'rclpy.executors': Mock()}):
                reporter = PersonReporter()
                results['partial_json_merge'] = (
                    abs(reporter.camera['hfov_rad'] - 1.5707963) < 1e-5 and
                    reporter.camera['width'] == 1280 and
                    reporter.camera['height'] == 960 and
                    abs(reporter.camera['pitch_deg'] - 85.0) < 1e-5
                )

        # Test 2.5: Image message resolution vs camera.json resolution in on_camera_image
        with patch('builtins.open', side_effect=make_open_mock(content_str=json.dumps({'width': 640, 'height': 480, 'hfov_rad': 1.04719755}))):
            mock_bridge = Mock()
            mock_bridge.imgmsg_to_cv2.return_value = Mock()
            mock_bridge.cv2_to_imgmsg.return_value = Mock()
            with patch.dict('sys.modules', {
                'rclpy': Mock(),
                'rclpy.context': Mock(),
                'rclpy.executors': Mock(),
                'cv_bridge': Mock(CvBridge=lambda: mock_bridge),
                'cv2': Mock(),
                'sensor_msgs': Mock(),
                'sensor_msgs.msg': Mock()
            }):
                reporter = PersonReporter()
                reporter.position = (10.0, 20.0, 16.5, 50.0, time.monotonic())
                
                # Mock YOLO box at center of 1280x960
                mock_box = Mock()
                mock_box.cls = [Mock(item=lambda: 0)]
                mock_box.conf = [Mock(item=lambda: 0.8)]
                mock_box.xyxy = [[630.0, 470.0, 650.0, 490.0]]  # center (640, 480)
                mock_res = Mock(boxes=[mock_box], plot=lambda: Mock())
                
                reporter.yolo_model = Mock()
                reporter.yolo_model.predict = lambda *a, **k: [mock_res]
                reporter.yolo_model.names = {0: 'person'}
                reporter.pub_annotated = Mock()
                
                # Incoming Image msg is 1280x960
                img_msg = Mock(width=1280, height=960)
                reporter.on_camera_image(img_msg)
                    
                if len(reporter.clusters) > 0:
                    c = reporter.clusters[0]
                    results['camera_image_uses_msg_dimensions'] = (
                        abs(c.north - 10.0) < 0.1 and abs(c.east - 20.0) < 0.1
                    )
                else:
                    results['camera_image_uses_msg_dimensions'] = False

        # Test 2.6: on_detections resolution dependency
        with patch('builtins.open', side_effect=make_open_mock(content_str=json.dumps({'width': 640, 'height': 480, 'hfov_rad': 1.04719755}))):
            with patch.dict('sys.modules', {'rclpy': Mock(), 'rclpy.context': Mock(), 'rclpy.executors': Mock()}):
                reporter = PersonReporter()
                reporter.position = (10.0, 20.0, 16.5, 50.0, time.monotonic())
                
                # External detection bbox from a 1280x960 detector placed at (640, 480)
                det = Mock()
                det.results = [Mock(hypothesis=Mock(class_id='person', score=0.8))]
                det.bbox = Mock(center=Mock(position=Mock(x=640.0, y=480.0)))
                det_msg = Mock(detections=[det])
                
                reporter.on_detections(det_msg)
                if len(reporter.clusters) > 0:
                    c = reporter.clusters[0]
                    # If detector sent pixel in 1280x960 coordinates, but camera.json was 640x480:
                    results['on_detections_mismatch_vulnerability'] = (
                        abs(c.north - 10.0) > 1.0 or abs(c.east - 20.0) > 1.0
                    )
                else:
                    results['on_detections_mismatch_vulnerability'] = False

    return results


# ==============================================================================
# 3. CONCURRENCY, RACE CONDITIONS & LATENCY GATE STRESS TEST
# ==============================================================================

def test_concurrency_and_latency_stress():
    """
    Stress tests PersonReporter under rapid concurrent telemetry and perception streams:
    1. Multi-threaded race condition between update_position (200Hz) and on_camera_image (50Hz)
    2. Latency freshness boundary: age = 0.99s vs 1.01s
    3. Spatial lag distortion: measure error when telemetry latency is 0.8s under speed 5.2 m/s
    4. Non-finite telemetry fuzzing (NaN, Inf, None) under continuous perception
    """
    results = {}
    
    reporter = PersonReporter()
    reporter.yolo_model = Mock()
    reporter.yolo_model.names = {0: 'person'}
    reporter.pub_annotated = Mock()
    
    # 3.1: Latency gate boundary test (0.99s vs 1.01s)
    # Test 0.99s age (must accept)
    reporter.clusters = []
    t_now = time.monotonic()
    reporter.position = (15.0, 15.0, 16.5, 10.0, t_now - 0.99)
    
    det = Mock()
    det.results = [Mock(hypothesis=Mock(class_id='person', score=0.85))]
    det.bbox = Mock(center=Mock(position=Mock(x=640.0, y=480.0)))
    reporter.on_detections(Mock(detections=[det]))
    accepted_0_99 = (len(reporter.clusters) == 1)
    
    # Test 1.01s age (must reject)
    reporter.clusters = []
    reporter.position = (15.0, 15.0, 16.5, 10.0, t_now - 1.01)
    reporter.on_detections(Mock(detections=[det]))
    rejected_1_01 = (len(reporter.clusters) == 0)
    
    results['latency_gate_0_99s_accepted'] = accepted_0_99
    results['latency_gate_1_01s_rejected'] = rejected_1_01
    
    # 3.2: Spatial error induced by allowable latency (< 1.0s)
    # If telemetry is 0.8s stale and drone travels at 5.2 m/s:
    speed = 5.2
    max_lag_err = speed * 0.99  # 5.148 meters!
    results['max_allowable_latency_spatial_error_m'] = max_lag_err
    results['exceeds_2_5m_at_stale_telemetry'] = (max_lag_err > 2.5)

    # 3.3: Multi-threaded race condition stress test
    # Simulate background telemetry thread writing position while main thread processes images
    stop_event = threading.Event()
    exceptions = []
    race_detected = False
    
    def telemetry_feeder():
        step = 0
        while not stop_event.is_set():
            step += 1
            n = 10.0 + (step % 50) * 0.5
            e = 20.0 + (step % 30) * 0.5
            h = 16.5
            msg = Mock(x=n, y=e, z=-h, time_boot_ms=step * 50)
            try:
                reporter.update_position(msg)
            except Exception as e:
                exceptions.append(('telemetry', e))
            time.sleep(0.002)  # ~500 Hz
            
    def perception_caller():
        for i in range(200):
            # Alternate on_detections
            det = Mock()
            det.results = [Mock(hypothesis=Mock(class_id='person', score=0.80))]
            det.bbox = Mock(center=Mock(position=Mock(x=640.0 + (i % 20), y=480.0)))
            try:
                reporter.on_detections(Mock(detections=[det]))
            except Exception as e:
                exceptions.append(('perception', e))
            time.sleep(0.003)
            
    t1 = threading.Thread(target=telemetry_feeder)
    t2 = threading.Thread(target=perception_caller)
    
    t1.start()
    t2.start()
    t2.join(timeout=3.0)
    stop_event.set()
    t1.join(timeout=1.0)
    
    results['concurrency_thread_exceptions'] = len(exceptions)
    results['concurrency_safe_cpython'] = (len(exceptions) == 0)

    # 3.4: Non-finite telemetry fuzzing
    fuzz_values = [
        (float('nan'), 10.0, -16.5),
        (10.0, float('inf'), -16.5),
        (-float('inf'), 10.0, -16.5),
        (10.0, 10.0, float('nan'))
    ]
    fuzz_passed = True
    reporter.position = (10.0, 10.0, 16.5, 1.0, time.monotonic())
    for (fx, fy, fz) in fuzz_values:
        bad_msg = Mock(x=fx, y=fy, z=fz, time_boot_ms=1000)
        reporter.update_position(bad_msg)
        # Position should NOT have been updated with non-finite values
        if not (math.isfinite(reporter.position[0]) and math.isfinite(reporter.position[1]) and math.isfinite(reporter.position[2])):
            fuzz_passed = False
            break
            
    results['non_finite_telemetry_rejected'] = fuzz_passed

    return results


# ==============================================================================
# 4. PERSONCLUSTER DEDUPLICATION BOUNDARY & STABILITY TEST
# ==============================================================================

def test_person_cluster_stability():
    """
    Empirically tests PersonCluster deduplication at boundary distances:
    1. Distance 4.49m (must merge into 1 cluster)
    2. Distance 4.51m (must separate into 2 clusters)
    3. Micro-boundary at 4.500000000000001m vs 4.499999999999999m
    4. Centroid chaining / spatial drift audit across multiple sequential detections
    5. Disk report freezing: check whether initial report coordinates freeze at first hit
    """
    results = {}
    
    # 4.1 Boundary: 4.49m vs 4.51m
    # Test 4.49m
    reporter_449 = PersonReporter()
    reporter_449.position = (0.0, 0.0, 16.5, 10.0, time.monotonic())
    c1 = PersonCluster(10.0, 10.0, 0.9, 10.0)
    reporter_449.clusters.append(c1)
    
    # Add second detection at 4.49m east: (10.0, 14.49)
    # Using on_detections logic:
    n2, e2 = 10.0, 14.49
    matched_449 = False
    for c in reporter_449.clusters:
        if math.hypot(n2 - c.north, e2 - c.east) <= DEDUP_RADIUS_M:
            c.add(n2, e2, 0.85)
            matched_449 = True
            break
            
    # Test 4.51m
    reporter_451 = PersonReporter()
    reporter_451.position = (0.0, 0.0, 16.5, 10.0, time.monotonic())
    c1_451 = PersonCluster(10.0, 10.0, 0.9, 10.0)
    reporter_451.clusters.append(c1_451)
    
    n3, e3 = 10.0, 14.51
    matched_451 = False
    for c in reporter_451.clusters:
        if math.hypot(n3 - c.north, e3 - c.east) <= DEDUP_RADIUS_M:
            c.add(n3, e3, 0.85)
            matched_451 = True
            break
            
    results['boundary_4_49m_merged'] = matched_449
    results['boundary_4_49m_cluster_count'] = len(reporter_449.clusters)
    results['boundary_4_49m_centroid_east'] = reporter_449.clusters[0].east
    
    results['boundary_4_51m_merged'] = matched_451
    
    # 4.2 Precision boundary: 4.4999999m vs 4.5000001m
    dist_sub = 4.4999999
    dist_sup = 4.5000001
    results['boundary_epsilon_below_merged'] = (math.hypot(0, dist_sub) <= DEDUP_RADIUS_M)
    results['boundary_epsilon_above_separated'] = (math.hypot(0, dist_sup) > DEDUP_RADIUS_M)

    # 4.3 Centroid Chaining / Cluster Drift Simulation
    # Sequential hits spaced by 3.0m in a line:
    # Points: x = 0, 3.0, 5.5, 7.5, 9.2
    # Span is 9.2 meters (more than double DEDUP_RADIUS_M = 4.5m)
    chain_cluster = PersonCluster(0.0, 0.0, 0.9, 1.0)
    chained_points = [3.0, 5.5, 7.5, 9.2]
    chained_absorbed = []
    
    for pt in chained_points:
        dist_to_centroid = math.hypot(0.0 - chain_cluster.north, pt - chain_cluster.east)
        if dist_to_centroid <= DEDUP_RADIUS_M:
            chain_cluster.add(0.0, pt, 0.9)
            chained_absorbed.append((pt, dist_to_centroid, chain_cluster.east))
            
    results['centroid_chaining_span_m'] = 9.2
    results['centroid_chaining_absorbed_count'] = len(chained_absorbed)
    results['centroid_chaining_final_centroid_east'] = chain_cluster.east
    # Did it absorb points farther than 4.5m from the original point (0, 0)?
    results['centroid_chaining_absorbed_distant_points'] = any(pt > 4.5 for pt, _, _ in chained_absorbed)

    # 4.4 Report Freezing on Disk Analysis
    # If the first hit has conf >= 0.45:
    # It triggers _report immediately at hit 1 coordinates.
    # Subsequent hits shift the in-memory centroid, but reports.jsonl is NEVER updated!
    with patch('solution.report_person') as mock_report:
        reporter_freeze = PersonReporter()
        reporter_freeze.position = (10.0, 20.0, 16.5, 10.0, time.monotonic())
        
        # Hit 1: noisy position (+2.0m north) with conf=0.85
        det1 = Mock()
        det1.results = [Mock(hypothesis=Mock(class_id='person', score=0.85))]
        det1.bbox = Mock(center=Mock(position=Mock(x=640.0, y=480.0)))  # maps to (10.0, 20.0)
        reporter_freeze.position = (10.0, 20.0, 16.5, 10.0, time.monotonic())
        reporter_freeze.on_detections(Mock(detections=[det1]))
        
        if mock_report.call_args is not None:
            kwargs = mock_report.call_args[1] if len(mock_report.call_args) > 1 else mock_report.call_args.kwargs
            first_call_coords = (kwargs.get('north_m'), kwargs.get('east_m'))
        else:
            first_call_coords = None
        
        # Hit 2: true position (12.0, 20.0) with conf=0.85
        # shifts centroid to (11.0, 20.0)
        det2 = Mock()
        det2.results = [Mock(hypothesis=Mock(class_id='person', score=0.85))]
        # Offset in y: +2m north
        # focal = 1280 / (2 * tan(hfov/2)) = 1108.51
        # cy offset: -(cy - 480) * 16.5 / 1108.51 = 2.0 => cy = 480 - 2.0 * 1108.51 / 16.5 = 345.635
        det2.bbox = Mock(center=Mock(position=Mock(x=640.0, y=345.635)))
        reporter_freeze.position = (10.0, 20.0, 16.5, 10.0, time.monotonic())
        reporter_freeze.on_detections(Mock(detections=[det2]))
        
        call_count = mock_report.call_count
        in_memory_centroid = (reporter_freeze.clusters[0].north, reporter_freeze.clusters[0].east) if reporter_freeze.clusters else None
        
        results['disk_report_call_count'] = call_count
        results['disk_reported_coords'] = first_call_coords
        results['in_memory_refined_centroid'] = in_memory_centroid
        results['disk_report_frozen_at_first_hit'] = (
            call_count == 1 and 
            first_call_coords is not None and 
            in_memory_centroid is not None and 
            abs(in_memory_centroid[0] - first_call_coords[0]) > 0.5
        )

    return results


# ==============================================================================
# MAIN EXECUTION & SUMMARY OUTPUT
# ==============================================================================

def main():
    print("=" * 80)
    print("ADVERSARIAL EMPIRICAL AUDIT: MILESTONE 1 (solution.py)")
    print("=" * 80)
    
    # 1. Monte Carlo Ground Plane
    print("\n[TEST 1] Monte Carlo Ground Plane Projection Error (10,000 runs)...")
    mc_res = run_monte_carlo_ground_plane(num_samples=10000)
    print(f"  Samples:                 {mc_res['num_samples']}")
    print(f"  Mean Error:              {mc_res['mean_error']:.3f} m  (Requirement: <= 2.5m)")
    print(f"  Std Deviation:           {mc_res['std_error']:.3f} m")
    print(f"  95th Percentile Error:   {mc_res['p95_error']:.3f} m")
    print(f"  Maximum Error:           {mc_res['max_error']:.3f} m")
    print(f"  Pass Rate (<= 2.5m):     {mc_res['pass_rate_2_5m']:.2f} %")
    print(f"  Decomposition:")
    print(f"    - Mean Projection (Attitude): {mc_res['mean_projection_error']:.3f} m")
    print(f"    - Critical Latency Tau:       {mc_res['critical_latency_tau_s']:.3f} s")
    print(f"    - Mean Yaw Drift Error (10°): {mc_res['mean_yaw_drift_error_10deg']:.3f} m")
    
    # 2. Camera Intrinsics
    print("\n[TEST 2] Camera Intrinsics & Fallback Ingestion Audit...")
    cam_res = test_camera_intrinsics_fallbacks()
    for k, v in cam_res.items():
        print(f"  - {k}: {v}")
        
    # 3. Concurrency & Latency Stress
    print("\n[TEST 3] Concurrency, Race Condition & Latency Gate Stress...")
    conc_res = test_concurrency_and_latency_stress()
    for k, v in conc_res.items():
        print(f"  - {k}: {v}")
        
    # 4. PersonCluster Stability
    print("\n[TEST 4] PersonCluster Stability & Boundary Deduplication Audit...")
    clust_res = test_person_cluster_stability()
    for k, v in clust_res.items():
        print(f"  - {k}: {v}")
        
    print("\n" + "=" * 80)
    print("EMPIRICAL TEST RUN COMPLETE")
    print("=" * 80)
    
    output_summary = {
        'monte_carlo': mc_res,
        'camera_intrinsics': cam_res,
        'concurrency_latency': conc_res,
        'clustering_stability': clust_res
    }
    
    # Dump JSON results for reporting
    with open(PROJECT_ROOT / 'challenge_m1_results.json', 'w') as f:
        json.dump(output_summary, f, indent=2)
    print(f"\n[OUTPUT] Detailed results written to {PROJECT_ROOT / 'challenge_m1_results.json'}")
    sys.stdout.flush()
    os._exit(0)

if __name__ == '__main__':
    main()
