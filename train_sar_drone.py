#!/usr/bin/env python3
"""
Script Huấn Luyện (Fine-tuning) Model YOLOv8 cho Drone Tìm Kiếm Cứu Nạn (SAR).
Sử dụng Dataset VisDrone chuyên cho góc nhìn trên cao (Aerial View).
Tối ưu hóa chạy trên GPU NVIDIA RTX 5070 Ti.
Tích hợp các kỹ thuật đặc trị:
- Người nằm / xoay đa hướng (degrees=180, flipud=0.5)
- Núp cây / che khuất 1 phần (erasing=0.4, mosaic=1.0, mixup=0.15)
"""

import os
import shutil
import sys
from pathlib import Path

import torch
from ultralytics import YOLO


def main():
    print("==================================================================")
    print("   KHỞI ĐỘNG HUẤN LUYỆN YOLOv8 - DRONE TÌM KIẾM CỨU NẠN (SAR)")
    print("==================================================================")

    # 1. Kiểm tra GPU CUDA
    if not torch.cuda.is_available():
        print("[CẢNH BÁO] Không tìm thấy GPU CUDA! Script sẽ chạy trên CPU (rất chậm).")
        device = 'cpu'
    else:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[GPU OK] Thiết bị phát hiện: {gpu_name} ({vram_gb:.1f} GB VRAM)")
        device = 0

    workspace_dir = Path.home() / "Workspace/260727-drone-setup"
    runs_dir = workspace_dir / "runs_sar"
    model_base = "yolo26s.pt"

    # 2. Tải mô hình nền YOLO26 Small
    print(f"[1/3] Đang tải mô hình nền: {model_base} (YOLO 2026 Small)...")
    model = YOLO(model_base)

    # 3. Tiến hành huấn luyện
    print("[2/3] Bắt đầu quá trình Huấn luyện trên tập dữ liệu VisDrone...")
    print("      (Nếu lần đầu chạy, hệ thống sẽ tự động tải dataset VisDrone ~2GB)")
    print("      Các chiêu thức đặc trị ca khó đã được kích hoạt: xoay 180°, xóa che 40%...")

    try:
        results = model.train(
            data="VisDrone.yaml",        # Dataset chuẩn cho Drone
            epochs=70,                   # 70 vòng học sâu (tối đa hóa độ nhạy và mAP)
            patience=20,                 # Tự động dừng sớm nếu 20 vòng không tăng thêm điểm
            imgsz=640,                   # Kích thước khung hình
            batch=16,                    # Tối ưu cho 16GB VRAM
            device=device,               # Ép chạy trên RTX 5070 Ti
            workers=4,
            
            # --- CÁC THAM SỐ ĐẶC TRỊ CA NÚP CÂY & NẰM BẤT TỈNH ---
            degrees=180.0,               # Xoay ảnh 180 độ (nhận diện mọi hướng nằm)
            flipud=0.5,                  # Lật trên dưới (người nằm lộn ngược)
            fliplr=0.5,                  # Lật trái phải
            hsv_h=0.015,                 # Biến thiên màu sắc
            hsv_v=0.4,                   # Biến thiên độ sáng tối (bóng râm)
            mosaic=1.0,                  # Ghép 4 góc nhìn
            mixup=0.15,                  # Hòa trộn ảnh (người chìm vào cỏ)
            erasing=0.4,                 # Xóa che 40% thân người (đặc trị NÚP CÂY/CHE KHUẤT)
            
            project=str(runs_dir),
            name="sar_visdrone_model",
            exist_ok=True,
            save=True,
            plots=True
        )

        # 4. Lưu lại model xịn nhất vào tất cả các thư mục dự án
        best_pt_src = runs_dir / "sar_visdrone_model/weights/best.pt"
        target_locations = [
            workspace_dir / "yolo26s_sar_best.pt",
            workspace_dir / "student-starter-kit/yolo26s_sar_best.pt",
            Path.home() / "Desktop/Drone/yolo26s_sar_best.pt",
            Path.home() / "Desktop/Drone/student-starter-kit/yolo26s_sar_best.pt",
            workspace_dir / "student-starter-kit/yolov8_sar_best.pt",
            Path.home() / "Desktop/Drone/student-starter-kit/yolov8_sar_best.pt",
        ]

        if best_pt_src.exists():
            for target_model in target_locations:
                target_model.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(best_pt_src, target_model)
                print(f"[XUẤT MODEL] Đã cập nhật: {target_model}")
            print("==================================================================")
            print("   HUẤN LUYỆN TỪ CON SỐ 0 HOÀN TẤT THÀNH CÔNG RỰC RỠ!")
            print(f"   Model tối ưu đã được lưu tại: {best_pt_src}")
            print("==================================================================")
        else:
            print(f"[XONG] Kết quả huấn luyện được lưu trong: {runs_dir}")

    except Exception as e:
        print(f"[LỖI TRONG QUÁ TRÌNH HUẤN LUYỆN]: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
