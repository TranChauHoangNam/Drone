#!/usr/bin/env python3
"""
Node ROS 2 Nhận Diện Cứu Hộ Cứu Nạn (SAR) bằng YOLOv8.
Tự động ưu tiên tải model đã huấn luyện: yolov8_sar_best.pt (nếu có), fallback về yolov8n.pt.
Tích hợp:
- Tự động lưu 1 FRAME OUTPUT duy nhất đạt chuẩn cứu hộ: output_victim.jpg
- Vẽ giao diện Tactical HUD (Tọa độ, Bounding box kép, Confidence)
- Hỗ trợ phát hiện người ở cả cự ly 30m và 12m (nằm, đứng, núp cây)
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO


class DroneSARYOLODetector(Node):
    def __init__(self):
        super().__init__('drone_sar_yolo_detector')

        # 1. Ưu tiên tìm model đã train riêng cho Drone
        workspace_dir = Path(__file__).parent
        custom_model = workspace_dir / 'yolov8_sar_best.pt'
        base_model = workspace_dir / 'yolov8n.pt'

        if custom_model.exists():
            model_path = str(custom_model)
            self.get_logger().info(f'🚀 [SAR AI] ĐÃ TÌM THẤY MODEL ĐÃ TRAIN: {model_path}')
        elif base_model.exists():
            model_path = str(base_model)
            self.get_logger().info(f'ℹ️ [SAR AI] Sử dụng model nền: {model_path}')
        else:
            model_path = 'yolov8n.pt'

        self.model = YOLO(model_path)
        self.bridge = CvBridge()
        self.last_log_time = 0.0

        # Cờ lưu 1 frame output duy nhất
        self.best_confidence = 0.0
        self.output_saved = False
        self.output_path = workspace_dir / 'output_victim.jpg'
        self.start_time = time.time()

        # Đăng ký nhận ảnh từ camera drone
        self.sub_image = self.create_subscription(
            Image,
            '/drone/camera/image_raw',
            self.image_callback,
            10
        )

        # Xuất luồng ảnh đã gán nhãn
        self.pub_annotated = self.create_publisher(
            Image,
            '/yolo_detect/image_annotated',
            10
        )

        self.get_logger().info('✅ Hệ thống SAR AI sẵn sàng! Đang lắng nghe: /drone/camera/image_raw')
        self.get_logger().info(f'📸 File output sẽ được lưu tại: {self.output_path}')

    def draw_tactical_hud(self, img, box_info, best_conf):
        """Vẽ lớp phủ thông số quân sự / cứu hộ chuyên nghiệp lên ảnh output."""
        h, w, _ = img.shape
        hud = img.copy()

        # 1. Băng thông số trên đỉnh (Top Banner)
        cv2.rectangle(hud, (0, 0), (w, 50), (20, 20, 20), -1)
        elapsed = int(time.time() - self.start_time)
        time_str = f"T+{elapsed//60:02d}:{elapsed%60:02d}"
        utc_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")

        cv2.putText(hud, "SAR UAV TACTICAL MISSION | ALT: ~12M | RANGE: <50M", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(hud, f"TIME: {time_str} | UTC: {utc_str}", (15, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # 2. Băng thông số dưới đáy (Bottom Banner)
        cv2.rectangle(hud, (0, h - 55), (w, h), (20, 20, 20), -1)
        cv2.putText(hud, "STATUS: TARGET LOCATED & VERIFIED | ACTION: PAYLOAD READY", (15, h - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.putText(hud, f"CONFIDENCE: {best_conf:.1%} | OUTPUT: {self.output_path.name}", (15, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # Hòa trộn mờ nhẹ 85%
        cv2.addWeighted(hud, 0.85, img, 0.15, 0, img)

        # 3. Vẽ Bounding box kép màu xanh Neon quanh nạn nhân
        if box_info:
            x1, y1, x2, y2 = box_info
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (0, 255, 255), 1)

            # Tâm ngắm (Crosshair)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.drawMarker(img, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

            label = f"VICTIM LOCATED [{best_conf:.1%}]"
            cv2.putText(img, label, (x1, max(25, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 4. Cắt góc zoom Picture-in-Picture (PiP) ở góc trên bên phải
            try:
                crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                if crop.size > 0:
                    pip_size = (160, 160)
                    pip_crop = cv2.resize(crop, pip_size)
                    cv2.rectangle(img, (w - 170, 60), (w - 8, 222), (0, 255, 255), 2)
                    img[62:222, w - 168:w - 8] = pip_crop
                    cv2.putText(img, "ZOOM PiP", (w - 160, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            except Exception:
                pass

        return img

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Lỗi giải mã ảnh: {e}')
            return

        # Dự đoán bằng YOLO (chạy nhanh với conf=0.30 để bắt nhạy)
        results = self.model.predict(cv_image, conf=0.30, verbose=False)
        annotated_frame = results[0].plot()

        boxes = results[0].boxes
        current_best_conf = 0.0
        best_box_coords = None

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                cls_name = self.model.names[cls_id].lower()
                conf = float(box.conf[0].item())

                # Lọc: Chỉ quan tâm nếu là người (person / pedestrian / people)
                if 'person' in cls_name or 'pedestrian' in cls_name or 'people' in cls_name:
                    if conf > current_best_conf:
                        current_best_conf = conf
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        best_box_coords = (x1, y1, x2, y2)

        # In log định kỳ
        now = time.time()
        if now - self.last_log_time >= 1.5:
            if current_best_conf > 0:
                self.get_logger().info(f"🎯 [SAR PHÁT HIỆN]: Nạn nhân ({current_best_conf:.1%})")
            self.last_log_time = now

        # LOGIC XUẤT 1 FRAME DUY NHẤT ĐẠT CHUẨN:
        # Nếu phát hiện người với độ tin cậy cao hơn kỷ lục cũ và >= 0.50
        if current_best_conf >= 0.50 and current_best_conf > self.best_confidence:
            self.best_confidence = current_best_conf
            hud_frame = self.draw_tactical_hud(cv_image.copy(), best_box_coords, current_best_conf)

            # Ghi ra file output_victim.jpg
            cv2.imwrite(str(self.output_path), hud_frame)
            self.output_saved = True
            self.get_logger().info(
                f"📸 [XUẤT 1 FRAME THÀNH CÔNG] Đã lưu {self.output_path.name} (Conf: {current_best_conf:.1%})"
            )

        # Hiển thị cửa sổ FPV
        cv2.imshow("Drone FPV Camera - YOLOv8 AI Detection", annotated_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            rclpy.shutdown()
            sys.exit(0)

        # Bắn ra topic ROS 2
        try:
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
            annotated_msg.header = msg.header
            self.pub_annotated.publish(annotated_msg)
        except Exception:
            pass


def main():
    rclpy.init()
    node = DroneSARYOLODetector()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        cv2.destroyAllWindows()
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
