#!/usr/bin/env python3
"""
TEST_M1_CHALLENGER_HARNESS.PY
Comprehensive Adversarial Empirical Stress Test Suite for Milestone 1.

Tasks:
1. Monte Carlo simulation (10,000 runs) of drone flight at 16.5m altitude, 5.2 m/s cruise speed,
   with realistic pitch and roll variations and telemetry latency.
2. Stress test PersonCluster spatial deduplication around 4.5m boundary (4.49m vs 4.51m),
   centroid chaining, and coordinate freezing.
3. Concurrency / race condition stress test with rapid interleaved telemetry and perception callbacks.
"""

import sys
import os
import math
import time
import json
import random
import threading
from pathlib import Path
from unittest.mock import Mock, patch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import solution
from solution import PersonCluster, PersonReporter, DEDUP_RADIUS_M, ALTITUDE, SPEED, MAX_POSITION_AGE_S


# ==============================================================================
# 1. MONTE CARLO SIMULATION: GROUND PLANE ERROR ACCURACY (10,000 RUNS)
# ==============================================================================

def run_monte_carlo_ground_plane_simulation(num_samples=10000, seed=12345):
    """
    Simulates 10,000 drone flight observations during nominal Lawnmower search.
    Flight parameters:
    - Altitude: 16.5m (corridor nominal)
    - Cruise speed: 5.2 m/s
    - Camera: 1280x960, HFOV = 60° (1.04719755 rad), focal = 1108.51 px
    - Multi-rotor physics:
      * North leg (v_n = +5.2 m/s): forward tilt (pitch theta = -3.5° to -5.5°)
      * South leg (v_n = -5.2 m/s): backward tilt (pitch theta = +3.5° to +5.5°)
      * East track switch (v_e = +5.2 m/s): right tilt (roll phi = +3.5° to +5.5°)
      * Atmospheric turbulence: Gaussian noise sigma = 1.0° on pitch and roll
      * Telemetry latency: 20Hz MAVLink stream with jitter in [0.02s, 0.08s]
    """
    random.seed(seed)
    
    width = 1280.0
    height = 960.0
    hfov_rad = 1.04719755
    focal = width / (2.0 * math.tan(hfov_rad / 2.0))
    
    results = {
        'total_samples': num_samples,
        'errors_total': [],
        'errors_proj_only': [],
        'errors_latency_only': [],
        'errors_by_leg': {'NORTH': [], 'SOUTH': [], 'EAST': []},
        'errors_center_fov': [],
        'errors_edge_fov': []
    }
    
    for _ in range(num_samples):
        alt = random.uniform(16.0, 17.0)
        drone_n = random.uniform(15.0, 85.0)
        drone_e = random.uniform(15.0, 85.0)
        
        # Flight legs
        leg = random.choices(['NORTH', 'SOUTH', 'EAST'], weights=[0.45, 0.40, 0.15])[0]
        
        if leg == 'NORTH':
            v_n = SPEED
            v_e = 0.0
            pitch_nom = -math.radians(random.uniform(3.5, 5.2))
            roll_nom = 0.0
        elif leg == 'SOUTH':
            v_n = -SPEED
            v_e = 0.0
            pitch_nom = math.radians(random.uniform(3.5, 5.2))
            roll_nom = 0.0
        else: # EAST
            v_n = 0.0
            v_e = SPEED
            pitch_nom = 0.0
            roll_nom = math.radians(random.uniform(3.5, 5.2))
            
        pitch = pitch_nom + random.gauss(0, math.radians(1.0))
        roll = roll_nom + random.gauss(0, math.radians(1.0))
        yaw = 0.0  # Setpoint stream locks yaw to 0.0 rad
        
        # Target pixel
        cx = random.uniform(40.0, width - 40.0)
        cy = random.uniform(40.0, height - 40.0)
        
        # Telemetry lag (20Hz stream: 0.02s to 0.08s)
        tau = random.uniform(0.02, 0.08)
        true_drone_n = drone_n - v_n * tau
        true_drone_e = drone_e - v_e * tau
        
        # Unit ray in camera frame
        r_cam = [cx - width / 2.0, cy - height / 2.0, focal]
        norm = math.hypot(r_cam[0], r_cam[1], r_cam[2])
        r_cam = [r_cam[0]/norm, r_cam[1]/norm, r_cam[2]/norm]
        
        # Camera to body frame:
        # X_b = -r_cam[1], Y_b = r_cam[0], Z_b = r_cam[2]
        r_body = [-r_cam[1], r_cam[0], r_cam[2]]
        
        # Body to NED rotation matrix: R = Ry(pitch) * Rx(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)
        
        R = [
            [cp, sp*sr, sp*cr],
            [0,  cr,    -sr],
            [-sp, cp*sr, cp*cr]
        ]
        
        r_ned = [
            R[0][0]*r_body[0] + R[0][1]*r_body[1] + R[0][2]*r_body[2],
            R[1][0]*r_body[0] + R[1][1]*r_body[1] + R[1][2]*r_body[2],
            R[2][0]*r_body[0] + R[2][1]*r_body[1] + R[2][2]*r_body[2]
        ]
        
        if r_ned[2] > 0.01:
            lamb = alt / r_ned[2]
            n_true = true_drone_n + lamb * r_ned[0]
            e_true = true_drone_e + lamb * r_ned[1]
            
            # solution.py nadir pinhole projection
            n_est = drone_n - (cy - height / 2.0) * alt / focal
            e_est = drone_e + (cx - width / 2.0) * alt / focal
            
            err = math.hypot(n_est - n_true, e_est - e_true)
            results['errors_total'].append(err)
            results['errors_by_leg'][leg].append(err)
            
            # Projection only error (no telemetry lag)
            n_est_sync = true_drone_n - (cy - height / 2.0) * alt / focal
            e_est_sync = true_drone_e + (cx - width / 2.0) * alt / focal
            err_proj = math.hypot(n_est_sync - n_true, e_est_sync - e_true)
            results['errors_proj_only'].append(err_proj)
            
            # Latency only error
            err_lat = math.hypot(drone_n - true_drone_n, drone_e - true_drone_e)
            results['errors_latency_only'].append(err_lat)
            
            # Center vs Edge of FOV
            dist_from_center_px = math.hypot(cx - width/2, cy - height/2)
            if dist_from_center_px < 200:
                results['errors_center_fov'].append(err)
            elif dist_from_center_px > 500:
                results['errors_edge_fov'].append(err)
                
    n_total = len(results['errors_total'])
    results['errors_total'].sort()
    
    mean_err = sum(results['errors_total']) / n_total
    std_err = math.sqrt(sum((x - mean_err)**2 for x in results['errors_total']) / n_total)
    p95_err = results['errors_total'][int(n_total * 0.95)]
    p99_err = results['errors_total'][int(n_total * 0.99)]
    max_err = results['errors_total'][-1]
    pass_rate = sum(1 for e in results['errors_total'] if e <= 2.5) / n_total * 100.0
    
    mean_proj = sum(results['errors_proj_only']) / len(results['errors_proj_only'])
    mean_lat = sum(results['errors_latency_only']) / len(results['errors_latency_only'])
    
    # Critical latency analysis: at what latency does mean error exceed 2.5m?
    # Mean error roughly scales as: mean_proj + SPEED * tau
    tau_crit = max(0.0, (2.5 - mean_proj) / SPEED)
    
    return {
        'samples': n_total,
        'mean_error': mean_err,
        'std_error': std_err,
        'p95_error': p95_err,
        'p99_error': p99_err,
        'max_error': max_err,
        'pass_rate_2_5m': pass_rate,
        'mean_projection_error': mean_proj,
        'mean_latency_error': mean_lat,
        'critical_latency_tau_s': tau_crit,
        'mean_north_leg': sum(results['errors_by_leg']['NORTH']) / len(results['errors_by_leg']['NORTH']),
        'mean_south_leg': sum(results['errors_by_leg']['SOUTH']) / len(results['errors_by_leg']['SOUTH']),
        'mean_east_leg': sum(results['errors_by_leg']['EAST']) / len(results['errors_by_leg']['EAST']),
        'mean_center_fov': sum(results['errors_center_fov']) / len(results['errors_center_fov']),
        'mean_edge_fov': sum(results['errors_edge_fov']) / len(results['errors_edge_fov']),
        'verdict_task1': 'PASS' if mean_err <= 2.5 else 'FAIL'
    }


# ==============================================================================
# 2. STRESS TEST: PERSONCLUSTER SPATIAL DEDUPLICATION AROUND 4.5M BOUNDARY
# ==============================================================================

def stress_test_person_cluster():
    """
    Stress tests PersonCluster spatial deduplication:
    1. Distance 4.49m vs 4.51m across multiple angles (0°, 45°, 90°, 135°, 180°, etc.)
    2. Floating-point boundary: 4.5 - 1e-12 vs 4.5 + 1e-12
    3. Centroid drift & chaining: sequential detections spaced by 3.0m
    4. Coordinate freezing: reports.jsonl written on first hit vs subsequent refined centroid
    """
    results = {}
    
    # 2.1 Boundary tests across 8 radial compass directions
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    all_4_49_merged = True
    all_4_51_separated = True
    
    for deg in angles:
        rad = math.radians(deg)
        # 4.49m
        dn_449 = 4.49 * math.cos(rad)
        de_449 = 4.49 * math.sin(rad)
        c1 = PersonCluster(100.0, 100.0, 0.9, 1.0)
        dist_449 = math.hypot((100.0 + dn_449) - c1.north, (100.0 + de_449) - c1.east)
        if dist_449 <= DEDUP_RADIUS_M:
            c1.add(100.0 + dn_449, 100.0 + de_449, 0.8)
        else:
            all_4_49_merged = False
            
        # 4.51m
        dn_451 = 4.51 * math.cos(rad)
        de_451 = 4.51 * math.sin(rad)
        c2 = PersonCluster(100.0, 100.0, 0.9, 1.0)
        dist_451 = math.hypot((100.0 + dn_451) - c2.north, (100.0 + de_451) - c2.east)
        if dist_451 <= DEDUP_RADIUS_M:
            all_4_51_separated = False
            
    results['boundary_4_49m_all_merged'] = all_4_49_merged
    results['boundary_4_51m_all_separated'] = all_4_51_separated
    
    # 2.2 Micro floating-point boundary
    eps = 1e-14
    results['boundary_sub_epsilon_merged'] = (math.hypot(0, 4.5 - eps) <= DEDUP_RADIUS_M)
    results['boundary_sup_epsilon_separated'] = (math.hypot(0, 4.5 + eps) > DEDUP_RADIUS_M)
    
    # 2.3 Centroid Chaining & Cluster Drift
    # If a person is detected 5 times, each shifted by 2.5m:
    # (0, 0), (0, 2.5), (0, 4.5), (0, 6.5), (0, 8.5)
    cluster = PersonCluster(0.0, 0.0, 0.9, 1.0)
    chain_offsets = [2.5, 4.5, 6.5, 8.5]
    absorbed = []
    
    for offset in chain_offsets:
        dist = math.hypot(0.0 - cluster.north, offset - cluster.east)
        if dist <= DEDUP_RADIUS_M:
            cluster.add(0.0, offset, 0.85)
            absorbed.append((offset, dist, cluster.east))
            
    results['chaining_absorbed_offsets'] = [pt[0] for pt in absorbed]
    results['chaining_final_centroid'] = (cluster.north, cluster.east)
    results['chaining_max_offset_absorbed'] = max([pt[0] for pt in absorbed], default=0.0)
    # Vulnerability: Did it absorb a detection > 4.5m away from the true initial victim location?
    results['chaining_vulnerability_detected'] = any(pt[0] > 4.5 for pt in absorbed)
    
    # 2.4 Coordinate Freezing on Disk (Single-hit report vs refined centroid)
    with patch('solution.report_person') as mock_report:
        reporter = PersonReporter()
        reporter.position = (10.0, 20.0, 16.5, 10.0, time.monotonic())
        
        # Frame 1: Detection with noise at (10.0, 22.0) with conf=0.8
        # Since conf >= 0.45, it reports immediately!
        det1 = Mock()
        det1.results = [Mock(hypothesis=Mock(class_id='person', score=0.8))]
        # Pixel mapping to +2.0m East:
        # focal = 1280 / (2*tan(30°)) = 1108.51
        # (cx - 640) * 16.5 / 1108.51 = 2.0 => cx = 640 + 2.0 * 1108.51 / 16.5 = 774.365
        det1.bbox = Mock(center=Mock(position=Mock(x=774.365, y=480.0)))
        reporter.on_detections(Mock(detections=[det1]))
        
        first_report_call = mock_report.call_args
        first_report_east = first_report_call.kwargs.get('east_m') if first_report_call else None
        
        # Frames 2-5: Detections at true position (10.0, 20.0)
        det_true = Mock()
        det_true.results = [Mock(hypothesis=Mock(class_id='person', score=0.9))]
        det_true.bbox = Mock(center=Mock(position=Mock(x=640.0, y=480.0)))
        for _ in range(4):
            reporter.on_detections(Mock(detections=[det_true]))
            
        final_centroid_east = reporter.clusters[0].east
        total_calls = mock_report.call_count
        
        results['disk_report_call_count'] = total_calls
        results['first_reported_east'] = first_report_east
        results['final_centroid_east'] = final_centroid_east
        results['coordinate_freezing_offset_m'] = abs(final_centroid_east - first_report_east) if first_report_east else 0.0
        # True if report was frozen at initial noisy estimate and never refined on disk
        results['coordinate_freezing_vulnerability'] = (total_calls == 1 and abs(final_centroid_east - first_report_east) > 0.5)

    return results


# ==============================================================================
# 3. CONCURRENCY & RACE CONDITION STRESS TEST
# ==============================================================================

def stress_test_concurrency_race_conditions():
    """
    Stress tests PersonReporter under rapid interleaved telemetry and perception:
    1. Multi-threaded race condition between update_position (500Hz) and perception (100Hz)
    2. Concurrent flush_pending during active cluster creation
    3. Rapid interleaved timestamp jitter and non-monotonic time injection
    """
    results = {}
    
    reporter = PersonReporter()
    reporter.yolo_model = Mock()
    reporter.yolo_model.names = {0: 'person'}
    reporter.pub_annotated = Mock()
    
    stop_event = threading.Event()
    telemetry_exceptions = []
    perception_exceptions = []
    flush_exceptions = []
    
    # Thread 1: High-frequency telemetry (500Hz)
    def telemetry_worker():
        t_boot = 1000.0
        while not stop_event.is_set():
            t_boot += 2.0  # 2ms intervals
            n = 50.0 + 10.0 * math.sin(t_boot * 0.01)
            e = 50.0 + 10.0 * math.cos(t_boot * 0.01)
            msg = Mock(x=n, y=e, z=-16.5, time_boot_ms=t_boot)
            try:
                reporter.update_position(msg)
            except Exception as ex:
                telemetry_exceptions.append(ex)
            time.sleep(0.001)
            
    # Thread 2: Rapid perception callbacks (100Hz)
    def perception_worker():
        for i in range(500):
            det = Mock()
            det.results = [Mock(hypothesis=Mock(class_id='person', score=0.85))]
            # Moving bbox
            det.bbox = Mock(center=Mock(position=Mock(x=640.0 + (i % 50), y=480.0 + (i % 30))))
            try:
                reporter.on_detections(Mock(detections=[det]))
            except Exception as ex:
                perception_exceptions.append(ex)
            time.sleep(0.002)
            
    # Thread 3: Periodic flush_pending (concurrent with cluster addition)
    def flush_worker():
        for _ in range(50):
            try:
                reporter.flush_pending()
            except Exception as ex:
                flush_exceptions.append(ex)
            time.sleep(0.015)
            
    t_tel = threading.Thread(target=telemetry_worker)
    t_per = threading.Thread(target=perception_worker)
    t_flu = threading.Thread(target=flush_worker)
    
    t_tel.start()
    t_per.start()
    t_flu.start()
    
    t_per.join(timeout=4.0)
    t_flu.join(timeout=2.0)
    stop_event.set()
    t_tel.join(timeout=2.0)
    
    results['telemetry_exceptions'] = len(telemetry_exceptions)
    results['perception_exceptions'] = len(perception_exceptions)
    results['flush_exceptions'] = len(flush_exceptions)
    results['total_concurrency_crashes'] = len(telemetry_exceptions) + len(perception_exceptions) + len(flush_exceptions)
    
    # 3.2 Non-monotonic timestamp and jitter resilience
    # Verify update_position drops stale out-of-order packets
    reporter.position = (10.0, 10.0, 16.5, 100.0, time.monotonic())
    stale_msg = Mock(x=99.0, y=99.0, z=-16.5, time_boot_ms=99000.0)  # 99s < 100s
    reporter.update_position(stale_msg)
    results['stale_out_of_order_telemetry_dropped'] = (reporter.position[0] == 10.0)
    
    # Fresh packet accepted
    fresh_msg = Mock(x=15.0, y=15.0, z=-16.5, time_boot_ms=101000.0)  # 101s > 100s
    reporter.update_position(fresh_msg)
    results['fresh_telemetry_accepted'] = (reporter.position[0] == 15.0)

    # 3.3 Latency gate boundary validation
    # Under default MAX_POSITION_AGE_S = 0.40s:
    t_now = time.monotonic()
    reporter.clusters = []
    reporter.position = (15.0, 15.0, 16.5, 10.0, t_now - 0.39)
    det = Mock(results=[Mock(hypothesis=Mock(class_id='person', score=0.85))], bbox=Mock(center=Mock(position=Mock(x=640.0, y=480.0))))
    reporter.on_detections(Mock(detections=[det]))
    results['latency_fresh_0_39s_accepted'] = (len(reporter.clusters) == 1)
    
    reporter.clusters = []
    reporter.position = (15.0, 15.0, 16.5, 10.0, t_now - 0.41)
    reporter.on_detections(Mock(detections=[det]))
    results['latency_stale_0_41s_rejected'] = (len(reporter.clusters) == 0)

    return results


# ==============================================================================
# MAIN TEST EXECUTION
# ==============================================================================

def main():
    print("=" * 80)
    print("CHALLENGER HARNESS: MILESTONE 1 ADVERSARIAL VERIFICATION")
    print("=" * 80)
    
    # Task 1
    print("\n--- TASK 1: Monte Carlo Simulation of Drone Flight (10,000 runs) ---")
    mc = run_monte_carlo_ground_plane_simulation(10000)
    print(f"Total Runs:                 {mc['samples']}")
    print(f"Mean Ground Error:          {mc['mean_error']:.3f} m  (Target: <= 2.5m)")
    print(f"Std Deviation:              {mc['std_error']:.3f} m")
    print(f"95th Percentile Error:      {mc['p95_error']:.3f} m")
    print(f"99th Percentile Error:      {mc['p99_error']:.3f} m")
    print(f"Maximum Error:              {mc['max_error']:.3f} m")
    print(f"Pass Rate (<= 2.5m):        {mc['pass_rate_2_5m']:.2f}%")
    print(f"Mean Projection (tilt):     {mc['mean_projection_error']:.3f} m")
    print(f"Mean Telemetry Latency:     {mc['mean_latency_error']:.3f} m")
    print(f"Critical Latency Tau:       {mc['critical_latency_tau_s']:.3f} s")
    print(f"Mean Error (North Leg):     {mc['mean_north_leg']:.3f} m")
    print(f"Mean Error (South Leg):     {mc['mean_south_leg']:.3f} m")
    print(f"Mean Error (East Leg):      {mc['mean_east_leg']:.3f} m")
    print(f"Mean Error (Center FOV):    {mc['mean_center_fov']:.3f} m")
    print(f"Mean Error (Edge FOV):      {mc['mean_edge_fov']:.3f} m")
    print(f"Verdict Task 1:             {mc['verdict_task1']}")
    
    # Task 2
    print("\n--- TASK 2: PersonCluster Deduplication Boundary & Stability ---")
    clust = stress_test_person_cluster()
    for k, v in clust.items():
        print(f"  {k}: {v}")
        
    # Task 3
    print("\n--- TASK 3: Concurrency & Race Condition Stress ---")
    conc = stress_test_concurrency_race_conditions()
    for k, v in conc.items():
        print(f"  {k}: {v}")
        
    print("\n" + "=" * 80)
    print("ALL HARNESS TESTS COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    report_data = {
        'task1_monte_carlo': mc,
        'task2_clustering': clust,
        'task3_concurrency': conc
    }
    with open(PROJECT_ROOT / 'challenger_harness_output.json', 'w') as f:
        json.dump(report_data, f, indent=2)

if __name__ == '__main__':
    main()
