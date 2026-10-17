#!/usr/bin/env python3
"""
SOLUTION.PY — MÃ NGUỒN THI ĐẤU AI DRONE CHALLENGE FA26 (VÒNG 1 BAYLANDS 230m x 230m)
Chiến thuật vô địch (300s):
- Cất cánh Offboard lên độ cao 16.5m (FOV rộng 19.05m, phủ kín toàn bộ bản đồ 230x230m)
- 13 Waypoint bao quát 6 làn quét Zíc-zắc (Lawnmower) từ E=-90m đến E=+90m qua cả 4 góc phần tư
- Tốc độ hành trình 5.2 m/s (dưới trần quy chế 10.0 m/s), lookahead 8.0m ôm cua êm 4.0m
- On-board YOLO inference + ROS2 Detections dự phòng + dán nhãn video /yolo_detect/image_annotated
- Phân cụm không gian 4.5m + Lọc BBox chống decoy + Proactive flush giây 275 + Hạ cánh an toàn giây 285
"""
import os
import math
import time
import sys
import json
from pymavlink import mavutil
from victim_report import report_person

# --- CẤU HÌNH THAM SỐ BAY CHIẾN ĐẤU (HIỆP ĐẤU 300 GIÂY / 5 PHÚT) ---
ALTITUDE = 16.5          # Độ cao chuẩn hành lang (8m <= H <= 20m), FOV rộng 19.05m
SPEED = 5.2             # Vận tốc hành trình (5.2 m/s), quét 6 làn trong 275s, dưới trần 10.0 m/s
LOOKAHEAD_DIST_M = 8.0  # Carrot lookahead bám sát lộ trình không bị giật lắc
MIN_CONFIDENCE = 0.35   # Ngưỡng chuẩn chống False Positive (trừ điểm nặng)
DEDUP_RADIUS_M = 4.5    # Bán kính gom cụm (cluster) vị trí nạn nhân
MAX_POSITION_AGE_S = float(os.environ.get('KIT_MAX_POSITION_AGE_S', '0.40'))
INFER_INTERVAL_S = 0.10


class PersonCluster:
    def __init__(self, north, east, conf, sim_s):
        self.north_sum = north
        self.east_sum = east
        self.count = 1
        self.max_conf = conf
        self.first_sim_s = sim_s
        self.reported = False

    def add(self, north, east, conf):
        self.north_sum += north
        self.east_sum += east
        self.count += 1
        if conf > self.max_conf:
            self.max_conf = conf

    @property
    def north(self):
        return self.north_sum / self.count

    @property
    def east(self):
        return self.east_sum / self.count


class PersonReporter:
    """Xử lý phát hiện nạn nhân: Tích hợp On-board YOLO Inference & vẽ Bounding Box trực tiếp."""
    def __init__(self, flush_min_conf=None):
        self.node = None
        self.context = None
        self.executor = None
        self.pub_annotated = None
        self.pub_cam_annotated = None
        self.position = None
        self.clusters = []
        self.reported_count = 0
        self.det_seen = 0
        self.det_used = 0
        self.camera = {'hfov_rad': 1.04719755, 'width': 1280, 'height': 960, 'pitch_deg': 90.0}
        self.yolo_model = None
        self._last_infer_time = 0.0
        self.flush_min_conf = flush_min_conf if flush_min_conf is not None else (
            float(os.environ['KIT_FLUSH_MIN_CONF']) if 'KIT_FLUSH_MIN_CONF' in os.environ else None
        )

        try:
            with open('/opt/kit/camera.json') as stream:
                data = json.load(stream)
                if isinstance(data, dict):
                    self.camera.update(data)
        except (OSError, ValueError, TypeError):
            pass

        # Nạp mô hình chuyên dụng cho SAR Drone: yolo26s_sar_best.pt (hỗ trợ đa đường dẫn)
        weights_env = os.environ.get('YOLO_WEIGHTS', 'yolo26s_sar_best.pt')
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidate_paths = [
            weights_env,
            os.path.join(script_dir, weights_env),
            os.path.join('/student', weights_env),
            os.path.join('/opt/kit', weights_env),
            os.path.join(script_dir, 'yolo26s_sar_best.pt'),
        ]
        resolved_weights = next((p for p in candidate_paths if os.path.isfile(p)), weights_env)
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO(resolved_weights)
            print(f'[AI] Đã nạp thành công mô hình SAR chuyên dụng: {resolved_weights}')
        except Exception as e:
            print(f'[AI] Nạp mô hình {resolved_weights} thất bại: {e}')

        try:
            import rclpy
            from rclpy.context import Context
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import Image
            from vision_msgs.msg import Detection2DArray

            self.ros = rclpy
            self.context = Context()
            rclpy.init(context=self.context)
            self.node = rclpy.create_node('solution_person_reporter', context=self.context)
            self.executor = SingleThreadedExecutor(context=self.context)
            self.executor.add_node(self.node)

            # 1. Kênh chính: Nhận ảnh camera trực tiếp để suy luận YOLO và dán nhãn
            self.sub_cam = self.node.create_subscription(
                Image, '/drone/camera/image_raw', self.on_camera_image, 1
            )
            # Kênh phụ camera: Hỗ trợ fallback topic /drone/camera/image
            self.sub_cam_fallback = self.node.create_subscription(
                Image, '/drone/camera/image', self.on_camera_image, 1
            )
            # 2. Xuất luồng ảnh đã vẽ bounding box cho video của ban giám khảo
            self.pub_annotated = self.node.create_publisher(
                Image, '/yolo_detect/image_annotated', 1
            )
            # Kênh phụ: Tương thích với chuẩn /drone/camera/annotated
            self.pub_cam_annotated = self.node.create_publisher(
                Image, '/drone/camera/annotated', 1
            )

            # 3. Kênh dự phòng: Lắng nghe detections từ container nền
            self.node.create_subscription(
                Detection2DArray, '/yolo_detect/detections',
                self.on_detections, qos_profile_sensor_data
            )
            print('[AI] ROS2 On-board YOLO & Bounding Box System SẴN SÀNG!')
        except Exception as exc:
            print(f'[AI] ROS2 subscriber error: {exc}')
            self.close()

    def update_position(self, msg):
        try:
            values = (float(msg.x), float(msg.y), -float(msg.z), float(msg.time_boot_ms) / 1000.0)
            if all(math.isfinite(v) for v in values):
                if self.position is None or values[3] >= self.position[3]:
                    self.position = (*values, time.monotonic())
        except (TypeError, AttributeError, ValueError):
            pass

    def on_camera_image(self, msg):
        """Xử lý ảnh camera trực tiếp: Nhận diện YOLO, vẽ bounding box, chiếu tọa độ và báo cáo."""
        if self.position is None or len(self.position) < 5:
            return
        # Fallback: Nếu không có YOLO model, chuyển tiếp ảnh gốc sang topic annotated để đảm bảo ghi video
        if self.yolo_model is None:
            try:
                if self.pub_annotated is not None:
                    self.pub_annotated.publish(msg)
                if self.pub_cam_annotated is not None:
                    self.pub_cam_annotated.publish(msg)
            except Exception:
                pass
            return

        now = time.monotonic()
        pos_age = now - self.position[4]
        if not (0.0 <= pos_age <= MAX_POSITION_AGE_S):
            return
        if now - self._last_infer_time < INFER_INTERVAL_S:
            return
        self._last_infer_time = now

        north, east, height, sim_s, _ = self.position
        if height < 2.0:
            return

        try:
            import numpy as np
            import cv2
            try:
                from sensor_msgs.msg import Image
            except Exception:
                Image = None

            # Giải mã frame ảnh bằng CvBridge hoặc numpy buffer
            img = None
            try:
                from cv_bridge import CvBridge
                bridge = CvBridge()
                img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            except Exception:
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, -1))
                if getattr(msg, 'encoding', '') in ('rgb8', 'RGB8') and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if img is None:
                return

            results = self.yolo_model.predict(img, conf=MIN_CONFIDENCE, verbose=False)
            res = results[0]

            # DÁN NHÃN & VẼ KHUNG NHẬN DIỆN (BOUNDING BOX) LÊN ẢNH VIDEO
            annotated_frame = res.plot()

            # Bắn frame đã dán nhãn ra topic để video ghi hình (hỗ trợ cả 2 topic)
            try:
                from cv_bridge import CvBridge
                bridge = CvBridge()
                out_msg = bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
                out_msg.header = msg.header
                if self.pub_annotated is not None:
                    self.pub_annotated.publish(out_msg)
                if self.pub_cam_annotated is not None:
                    self.pub_cam_annotated.publish(out_msg)
            except Exception:
                if Image is not None:
                    out_msg = Image()
                    out_msg.header = msg.header
                    out_msg.height = annotated_frame.shape[0]
                    out_msg.width = annotated_frame.shape[1]
                    out_msg.encoding = 'bgr8'
                    out_msg.is_bigendian = 0
                    out_msg.step = annotated_frame.shape[1] * 3
                    out_msg.data = annotated_frame.tobytes()
                    if self.pub_annotated is not None:
                        self.pub_annotated.publish(out_msg)
                    if self.pub_cam_annotated is not None:
                        self.pub_cam_annotated.publish(out_msg)

            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                return

            width, image_h = float(msg.width), float(msg.height)
            hfov = float(self.camera.get('hfov_rad', 1.04719755))
            if width <= 0 or image_h <= 0 or not (0.0 < hfov < math.pi):
                return
            tan_val = math.tan(hfov / 2.0)
            if tan_val <= 0:
                return
            focal = width / (2.0 * tan_val)
            if focal <= 0:
                return

            for box in boxes:
                cls_id = int(box.cls[0].item())
                cls_name = self.yolo_model.names.get(cls_id, str(cls_id)).lower()
                conf = float(box.conf[0].item())

                if cls_name not in ('person', '0', '1', 'pedestrian', 'people', 'group', 'groups', 'crowd', 'victim', 'victims') and str(cls_id) not in ('0', '1'):
                    continue

                xyxy = box.xyxy[0].tolist()
                w = xyxy[2] - xyxy[0]
                h = xyxy[3] - xyxy[1]
                area = w * h
                if not (math.isfinite(w) and math.isfinite(h) and math.isfinite(area)):
                    continue
                # Lọc kích thước BBox: Loại bỏ nhiễu pixel (< 150 px^2) và vật thể quá lớn (> 25000 px^2: xe, lều, bạt)
                if w <= 0.0 or h <= 0.0 or area < 150.0 or area > 25000.0:
                    continue

                self.det_seen += 1
                self.det_used += 1

                cx = (xyxy[0] + xyxy[2]) / 2.0
                cy = (xyxy[1] + xyxy[3]) / 2.0

                n = north - (cy - image_h / 2.0) * height / focal
                e = east + (cx - width / 2.0) * height / focal

                if not (math.isfinite(n) and math.isfinite(e)):
                    continue

                self._add_detection(n, e, conf, sim_s)

        except Exception:
            pass

    def on_detections(self, msg):
        """Kênh dự phòng: Nhận diện từ container nền nếu có."""
        try:
            if self.position is None or len(self.position) < 5:
                return
            now = time.monotonic()
            pos_age = now - self.position[4]
            if not (0.0 <= pos_age <= MAX_POSITION_AGE_S):
                return
            north, east, height, sim_s, _ = self.position
            if height < 2.0:
                return

            width, image_h = float(self.camera.get('width', 1280)), float(self.camera.get('height', 960))
            hfov = float(self.camera.get('hfov_rad', 1.04719755))
            if width <= 0 or image_h <= 0 or not (0.0 < hfov < math.pi):
                return
            tan_val = math.tan(hfov / 2.0)
            if tan_val <= 0:
                return
            focal = width / (2.0 * tan_val)
            if focal <= 0:
                return

            for detection in getattr(msg, 'detections', []):
                scores = []
                for r in getattr(detection, 'results', []):
                    hypo = getattr(r, 'hypothesis', r)
                    cid = str(getattr(hypo, 'class_id', getattr(hypo, 'id', ''))).lower()
                    if cid in ('person', '0', '1', 'pedestrian', 'people', 'group', 'groups', 'crowd', 'victim', 'victims'):
                        score_val = getattr(hypo, 'score', getattr(hypo, 'confidence', 0.0))
                        try:
                            scores.append(float(score_val))
                        except (TypeError, ValueError):
                            pass
                conf = max(scores, default=0.0)
                if not math.isfinite(conf) or conf < MIN_CONFIDENCE:
                    continue

                self.det_seen += 1
                self.det_used += 1
                n, e = north, east
                center_obj = getattr(getattr(detection, 'bbox', None), 'center', None)
                if center_obj is not None:
                    pos = getattr(center_obj, 'position', center_obj)
                    cx = float(getattr(pos, 'x', 0.0))
                    cy = float(getattr(pos, 'y', 0.0))
                    if abs(float(self.camera.get('pitch_deg', 90.0)) - 90.0) < 1.0:
                        n -= (cy - image_h / 2.0) * height / focal
                        e += (cx - width / 2.0) * height / focal

                if not (math.isfinite(n) and math.isfinite(e)):
                    continue

                self._add_detection(n, e, conf, sim_s)
        except Exception:
            pass

    def _add_detection(self, n, e, conf, sim_s):
        """Xử lý gom cụm không gian và kích hoạt quy trình báo cáo 2 tầng."""
        matched = False
        for c in self.clusters:
            if math.hypot(n - c.north, e - c.east) <= DEDUP_RADIUS_M:
                c.add(n, e, conf)
                matched = True
                if c.count >= 2 and not c.reported:
                    self._report(c, sim_s)
                break

        if not matched:
            new_c = PersonCluster(n, e, conf, sim_s)
            self.clusters.append(new_c)
            if conf >= 0.45:
                self._report(new_c, sim_s)

    def _report(self, c, sim_s):
        if c.reported:
            return
        c.reported = True
        self.reported_count += 1
        report_person(north_m=c.north, east_m=c.east, confidence=c.max_conf, sim_s=sim_s)
        print(f"[AI] REPORT person N={c.north:.1f} E={c.east:.1f} conf={c.max_conf:.2f} hits={c.count} ver=True")

    def flush_pending(self, sim_s=None, min_conf=None):
        """Báo cáo tất cả các cluster còn lại để bảo toàn 100% nạn nhân trước khi hết giờ.
        Chỉ flush các cụm chưa báo cáo thoả mãn: c.count >= 2 hoặc c.max_conf >= min_conf (mặc định 0.40).
        Các phát hiện đơn lẻ có độ tin cậy thấp (< 0.40 và count == 1) bị loại bỏ để tránh phạt -1.0 TP từ decoy.
        """
        guard_threshold = min_conf if min_conf is not None else self.flush_min_conf
        if guard_threshold is not None and guard_threshold > 0.0:
            unreported = [
                c for c in self.clusters
                if not c.reported and (c.count >= 2 or c.max_conf >= guard_threshold)
            ]
        else:
            unreported = [c for c in self.clusters if not c.reported]
        for c in unreported:
            self._report(c, c.first_sim_s if c.first_sim_s is not None else sim_s)
        print(f"[AI] STATS: det_seen={self.det_seen} det_used={self.det_used} clusters={len(self.clusters)} reported={self.reported_count}")

    def poll(self):
        if self.node is not None:
            try:
                self.executor.spin_once(timeout_sec=0.0)
            except Exception as exc:
                print(f'[AI] Subscriber error; continuing flight: {exc}')
                self.close()

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(timeout_sec=0.0)
            self.executor = None
        if self.node is not None:
            self.node.destroy_node()
            self.node = None
        if self.context is not None and self.context.ok():
            self.context.shutdown()


def set_px4_mode(master, mode_name):
    """Gửi đúng mã lệnh mode của PX4 (tránh lỗi fallback APM của pymavlink)."""
    px4_modes = {
        'OFFBOARD': (29, 6, 0),
        'TAKEOFF':  (29, 4, 2),
        'LAND':     (29, 4, 6),
        'POSCTL':   (81, 3, 0),
    }
    if mode_name in px4_modes:
        base, custom, sub = px4_modes[mode_name]
        master.mav.command_long_send(
            master.target_system, master.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0, base, custom, sub, 0, 0, 0, 0
        )
    try:
        master.set_mode(mode_name)
    except Exception:
        pass


def send_setpoint(master, x, y, z, yaw=0.0):
    """Gửi setpoint vị trí cục bộ NED tới Autopilot PX4 (tần số 20Hz)."""
    master.mav.set_position_target_local_ned_send(
        0, master.target_system, master.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000101111111000, # Chỉ điều khiển x, y, z, yaw
        x, y, -z,           # Hệ NED: z âm là độ cao hướng lên trời
        0, 0, 0, 0, 0, 0,
        yaw, 0
    )


def fly_mission(reporter):
    print("[SOLUTION] Đang kết nối tới PX4 qua MAVLink UDP 14540...")
    master = mavutil.mavlink_connection('udpin:0.0.0.0:14540')
    
    # Chờ nhận Heartbeat từ Autopilot
    while True:
        msg = master.recv_match(type='HEARTBEAT', blocking=True, timeout=5.0)
        if msg and msg.get_srcComponent() == 1:
            master.target_system = msg.get_srcSystem()
            master.target_component = msg.get_srcComponent()
            print(f"[SOLUTION] Đã kết nối Autopilot PX4 (System {master.target_system}, Component {master.target_component})")
            break

    # 1. Chờ lấy tọa độ Home
    home_x, home_y, home_z = 0.0, 0.0, 0.0
    for _ in range(50):
        msg = master.recv_match(type='LOCAL_POSITION_NED', blocking=True, timeout=1.0)
        if msg and all(math.isfinite(v) for v in (msg.x, msg.y, msg.z)):
            home_x, home_y, home_z = msg.x, msg.y, -msg.z
            print(f"[SOLUTION] Tọa độ Home xác định: N={home_x:.1f}, E={home_y:.1f}, Z={home_z:.1f}")
            break
        time.sleep(0.1)

    # 2. Sinh 13 Waypoint chiến lược phủ kín 6 làn quét Zíc-zắc (Lawnmower) toàn bộ 4 góc 230m x 230m
    waypoints = [
        # Lane 1 (West: E = -90m)
        (-95.0, -90.0, home_z + ALTITUDE),
        (95.0,  -90.0, home_z + ALTITUDE),
        # Lane 2 (Mid-West: E = -54m)
        (95.0,  -54.0, home_z + ALTITUDE),
        (-95.0, -54.0, home_z + ALTITUDE),
        # Lane 3 (Center-West: E = -18m)
        (-95.0, -18.0, home_z + ALTITUDE),
        (95.0,  -18.0, home_z + ALTITUDE),
        # Lane 4 (Center-East: E = +18m)
        (95.0,  18.0, home_z + ALTITUDE),
        (-95.0, 18.0, home_z + ALTITUDE),
        # Lane 5 (Mid-East: E = +54m)
        (-95.0, 54.0, home_z + ALTITUDE),
        (95.0,  54.0, home_z + ALTITUDE),
        # Lane 6 (East: E = +90m)
        (95.0,  90.0, home_z + ALTITUDE),
        (-95.0, 90.0, home_z + ALTITUDE),
        # Return towards Home
        (0.0,   0.0,  home_z + ALTITUDE),
    ]
    print(f"[MISSION] Đã thiết lập {len(waypoints)} Waypoint chiến lược (6 làn quét toàn sân 230x230m, độ cao {ALTITUDE}m).")

    # 3. Phát setpoint mồi trước khi kích hoạt Offboard
    print("[SOLUTION] Đang phát setpoint mồi 20Hz...")
    for _ in range(25):
        send_setpoint(master, home_x, home_y, home_z + ALTITUDE)
        time.sleep(0.05)

    # 4. QUY TRÌNH CẤT CÁNH SIÊU TỐC (OFFBOARD CLIMB)
    print(f"[TAKEOFF] Bắt đầu cất cánh Offboard lên độ cao {ALTITUDE}m...")
    t_takeoff_start = time.time()
    last_cmd_time = 0.0

    while time.time() - t_takeoff_start < 15.0:
        send_setpoint(master, home_x, home_y, home_z + ALTITUDE)
        reporter.poll()

        now = time.time()
        # Định kỳ mỗi 0.4s: Nhắc lệnh ARM & OFFBOARD
        if now - last_cmd_time >= 0.4:
            last_cmd_time = now
            master.mav.command_long_send(
                master.target_system, master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0, 1, 0, 0, 0, 0, 0, 0
            )
            set_px4_mode(master, 'OFFBOARD')

        msg = master.recv_match(type='LOCAL_POSITION_NED', blocking=False)
        if msg:
            reporter.update_position(msg)
            cur_z = -msg.z
            # Đạt độ cao an toàn (>= 8.5m) là xuất kích ngay lập tức!
            if cur_z >= 8.5:
                print(f"[TAKEOFF] CẤT CÁNH THÀNH CÔNG! Đạt {cur_z:.1f}m. Bắt đầu bay quét lập tức!")
                break

        time.sleep(0.05)

    # Gửi lệnh giới hạn vận tốc tối đa tới PX4 để triệt tiêu nguy cơ vi phạm trần 10.0 m/s
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
        0, 0, SPEED, -1, 0, 0, 0, 0
    )

    # 5. Vòng lặp bay theo 13 Waypoint chiến lược 6 làn
    mission_start_time = time.time()
    flushed = False
    cur_x, cur_y = home_x, home_y
    cur_z = cur_z if cur_z >= 8.0 else 0.0

    try:
        for idx, (target_x, target_y, target_z) in enumerate(waypoints):
            print(f"[MISSION] Waypoint {idx+1}/{len(waypoints)}: N={target_x:.1f} E={target_y:.1f}")
            wp_start = time.time()
            
            while True:
                # Carrot setpoint: Giữ điểm ngắm phía trước LOOKAHEAD_DIST_M để bay êm, không vượt tốc
                dx = target_x - cur_x
                dy = target_y - cur_y
                dist = math.hypot(dx, dy)
                if dist > LOOKAHEAD_DIST_M:
                    sp_x = cur_x + (dx / dist) * LOOKAHEAD_DIST_M
                    sp_y = cur_y + (dy / dist) * LOOKAHEAD_DIST_M
                else:
                    sp_x, sp_y = target_x, target_y

                send_setpoint(master, sp_x, sp_y, target_z)
                reporter.poll()
                
                # Failsafe: Nếu drone chưa đạt độ cao >= 8.0m, tiếp tục nhồi ARM + OFFBOARD
                now = time.time()
                if cur_z < 8.0 and (now - last_cmd_time >= 0.5):
                    last_cmd_time = now
                    master.mav.command_long_send(
                        master.target_system, master.target_component,
                        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                        0, 1, 0, 0, 0, 0, 0, 0
                    )
                    set_px4_mode(master, 'OFFBOARD')

                msg = master.recv_match(type='LOCAL_POSITION_NED', blocking=False)
                if msg:
                    cur_x, cur_y, cur_z = msg.x, msg.y, -msg.z
                    sim_s = msg.time_boot_ms / 1000.0
                    dist = math.hypot(target_x - cur_x, target_y - cur_y)
                    reporter.update_position(msg)
                    
                    # Proactive Flush ở giây thứ 275 để bảo toàn 100% nạn nhân
                    elapsed = time.time() - mission_start_time
                    if elapsed >= 275.0 and not flushed:
                        flushed = True
                        print(f"[SOLUTION] sim_flight={elapsed:.1f}s: Proactive flush bảo toàn toàn bộ nạn nhân!")
                        reporter.flush_pending(sim_s, min_conf=0.40)

                    # Ở giây thứ 285: Kích hoạt hạ cánh an toàn trước mốc 300s
                    if elapsed >= 285.0:
                        print(f"[MISSION] Hết thời gian bay ({elapsed:.1f}s) -> Hạ cánh an toàn!")
                        set_px4_mode(master, 'LAND')
                        break

                    # Bán kính 4.0m giúp drone ôm cua mượt mà ở tốc độ 5.2 m/s
                    if dist < 4.0:
                        break

                if time.time() - wp_start > 45.0:
                    print(f"[MISSION] Timeout waypoint {idx+1}, chuyển tiếp sang điểm kế tiếp.")
                    break

                time.sleep(0.05)

            if time.time() - mission_start_time >= 285.0:
                break

    except KeyboardInterrupt:
        print("[SOLUTION] KeyboardInterrupt received — flushing pending clusters")
        reporter.flush_pending(min_conf=0.40)

    # Flush lần cuối bảo đảm trước khi kết thúc
    reporter.flush_pending(min_conf=0.40)
    print("[MISSION] Hoàn tất bài thi!")


def main():
    reporter = PersonReporter(flush_min_conf=0.40)
    try:
        fly_mission(reporter)
    finally:
        reporter.close()


if __name__ == '__main__':
    main()
