# HƯỚNG DẪN LÀM BÀI VÀ NỘP BÀI — AI DRONE CHALLENGE FA26 (VÒNG 1)

Chào mừng các bạn sinh viên tham gia Vòng 1 — Thử thách Tìm kiếm & Cứu nạn Baylands ($100\text{ m} \times 100\text{ m}$)!

---

## 1. Cấu trúc thư mục Starter Kit

```
student-starter-kit/
├── solution.py          # File mã nguồn chính sinh viên cần phát triển (Bay & Báo cáo)
├── victim_report.py     # Thư viện chuẩn do BTC cung cấp để ghi nhận tọa độ nạn nhân
├── pack_submission.py   # Script kiểm tra hợp lệ và đóng gói bài nộp
└── README.md            # Tài liệu hướng dẫn này
```

---

## 2. Sinh viên cần làm gì trong code?

Mở file `solution.py`, sinh viên tập trung vào 3 khu vực:

1. **Điều chỉnh chiến thuật bay:**
   - Thay đổi độ cao `ALTITUDE` (Phải nằm trong hành lang an toàn: $8\text{m} \le H \le 20\text{m}$).
   - Thay đổi vận tốc `SPEED` (Khuyến nghị $3.0 - 4.5\text{ m/s}$ để drone không bị nghiêng lật ở góc cua).
   - Thay đổi khoảng cách làn `LANE_SPACING` để đảm bảo camera phủ kín mặt đất (gối biên $\ge 20\%$).

2. **Tích hợp Model AI YOLO:**
   - Bài mẫu đã có `PersonReporter` lắng nghe `/yolo_detect/detections` (`vision_msgs/Detection2DArray`), nhận nhãn `person` với confidence ≥ 0.35.
   - Với camera pitch 90°, yaw 0, baseline chiếu tâm bbox xuống mặt đất từ `/opt/kit/camera.json`; nếu thiếu thông số camera thì dùng vị trí drone gần đúng. Muốn tăng độ chính xác cần bù attitude, độ cao địa hình và đồng bộ timestamp ảnh.
   - Chỉ dùng vị trí MAVLink mới trong 1 giây, bỏ qua detections khi độ cao dưới 2 m và lọc trùng trong bán kính 6 m. Report ghi vào `$KIT_REPORT_PATH` kèm thời gian mô phỏng.
   - Khi thiếu ROS2, chương trình in cảnh báo và tiếp tục bay; không tự tạo report giả. Chạy trong môi trường đã source `/opt/kit/kit-env.sh` để bật subscriber.
   - **Custom model (`.pt` / `.onnx`, tổng gói nộp ≤ 30 MB):** Copy file weights vào thư mục bài làm (cùng cấp `solution.py`) **trước** khi chạy `pack_submission.py`. Container chấm chạy `--network none` — không tải model từ Internet lúc thi. Khi pack, khai báo đường dẫn qua manifest env, ví dụ:
     ```bash
     python3 pack_submission.py --team-id T05 --env YOLO_WEIGHTS=my_person.pt --env KIT_YOLO=1
     ```
     Hoặc trong `solution.py` đọc `os.environ.get('YOLO_WEIGHTS', 'yolov8n.pt')` nếu đội tự chạy inference (node YOLO nền của kit vẫn dùng biến `YOLO_WEIGHTS` / `KIT_YOLO` từ manifest khi chấm).
   - Có thể mở rộng `PersonReporter` hoặc thay pipeline nhận diện; điểm AI vẫn tính từ `report_person` và Ground Truth, không từ video.

3. **Gửi báo cáo khi phát hiện nạn nhân:**
   - Khi phát hiện có người với độ tin cậy cao, gọi hàm:
     ```python
     from victim_report import report_person

     report_person(
         north_m=cur_north,
         east_m=cur_east,
         confidence=float(conf),
         sim_s=current_sim_time
     )
     # Hoặc báo cáo theo tọa độ GPS:
     # report_person(lat=cur_lat, lon=cur_lon, confidence=float(conf), sim_s=current_sim_time)
     ```

### 2.1. Cấu trúc dữ liệu báo cáo & Cơ chế ghi nhận
- Thư viện tự động ghi từng dòng JSONL vào `/student/output/reports.jsonl`:
  ```json
  {"north_m": 92.85, "east_m": 11.42, "confidence": 0.57, "sim_s": 24.3, "frame_id": null}
  ```
  - `north_m`, `east_m`: Tọa độ mặt đất tính bằng mét so với điểm cất cánh của drone (khớp hệ tọa độ NED của PX4).
  - `confidence`: Điểm tin cậy từ model AI (từ `0.0` đến `1.0`).
  - `sim_s`: Thời gian mô phỏng (giây). Dùng để Grader loại bỏ các báo cáo gửi muộn (late reports) sau khi hết giờ thi.
- **Cơ chế streaming liên tục:** Báo cáo được flush trực tiếp vào file ngay khi phát hiện, **không cần đợi drone bay hết hành trình**.

### 2.2. Xử lý trùng lặp & Báo sai (False Positive)
- **1 vật thể bị gửi 2 lần lệch nhau một chút:**
  - Grader xét khoảng cách Euclidean $\le 10\text{m}$.
  - Lần gửi đầu tiên khớp thành công tính là **True Positive (TP — Có điểm)**.
  - Lần gửi tiếp theo vào cùng nạn nhân đó được coi là **Duplicate (Trùng lặp)**: không cộng thêm điểm và **không bị phạt**.
- **Gửi đúng vị trí nhưng sai vật thể (ví dụ báo nhầm xe thành người):**
  - Ground Truth chỉ có tọa độ người bị nạn. Nếu ở đó chỉ có xe hoặc đất trống (không có người trong bán kính $10\text{m}$), Grader tính là **False Positive (FP — Báo rác)**.
  - **Hình phạt:** Mỗi điểm FP sẽ bị **trừ $0.5$ lần điểm đúng** ở phần AI và làm giảm điểm Precision ở phần Tọa độ. Đội thi cần chủ động lọc trùng và kiểm soát ngưỡng tin cậy (`confidence`) trước khi gửi.

### 2.3. Cơ chế cô lập bài thi giữa các đội
Khi nhiều đội cùng chạy test trên server, dữ liệu được bảo vệ tuyệt đối:
- **Thư mục riêng:** Mỗi đội chạy trong workspace riêng `grading/runs/<team>/workspace:/student`.
- **Mạng riêng:** Gán `ROS_DOMAIN_ID = 20 + slot_idx` và phân vùng Gazebo độc lập, không bị lẫn topic ROS hay MAVLink.
- **Telemetry & Video riêng:** Ghi thẳng vào thư mục artifact riêng của từng đội.

### 2.4. Định dạng Ground Truth & Quy định Điểm cất cánh cố định
- **Dữ liệu Ground Truth:** BTC lưu danh sách 150 nạn nhân theo định dạng JSON với tọa độ `(north_m, east_m)` tính bằng mét so với bãi cất cánh. Hệ thống chấm hỗ trợ đối chiếu cả tọa độ mét `(north_m, east_m)`, tọa độ GPS `(lat, lon)` và tọa độ Gazebo `(x, y)`.
- **Điểm cất cánh cố định (Spawn Point):** Để đảm bảo công bằng cho cả 35 đội thi, mọi drone đều cất cánh từ điểm xuất phát cố định $(X=0, Y=0, Z=0)$ trong Gazebo (Home GPS `lat=37.412173, lon=-121.998878`). Sinh viên **không được thay đổi vị trí spawn**.
- **Chiến thuật bay:** Đội thi có thể lập trình cho drone cất cánh rồi bay thẳng với tốc độ cao ($5 - 6\text{ m/s}$) đến bất kỳ phân vùng nào trên bản đồ rồi mới bắt đầu hạ thấp tìm kiếm. Tọa độ báo cáo `(north_m, east_m)` vẫn luôn được tính từ điểm cất cánh $(0, 0)$ của drone (lấy từ `LOCAL_POSITION_NED` `msg.x, msg.y`).

---

## 3. Các Giới Hạn Cần Tuân Thủ (Vi phạm sẽ bị hủy bài)

1. **Dung lượng bài nộp:** Toàn bộ code + model trọng số `.pt` phải **$\le 30\text{ MB}$** (Khuyên dùng YOLOv8n dung lượng ~6.2MB).
2. **Không có Internet / Mạng nội bộ lúc thi:** Container chạy chế độ `--network none`. Tuyệt đối không viết lệnh `pip install` hay `git clone` trong lúc chạy bài. Mọi thư viện chuẩn (`torch`, `ultralytics`, `opencv`, `numpy`, `pymavlink`) đã được cài sẵn trong image `drone-kit:v2`.
3. **Giới hạn thời gian:** Tối đa **8 phút mô phỏng (Sim Time)** hoặc **10 phút thời gian thực (Wall Time)**. Quá thời gian này hệ thống sẽ tự ngắt bài thi.

---

## 4. Cách đóng gói bài nộp

Khi đã hoàn thiện code và model, mở terminal tại thư mục này và chạy:

```bash
python3 pack_submission.py --team-id <MÃ_ĐỘI> --entry solution.py
```
*(Ví dụ đội T05: `python3 pack_submission.py --team-id T05 --entry solution.py`)*

**Video chỉ có HUD, không vẽ khung YOLO** (quan sát địa hình / đường bay rõ hơn):

```bash
python3 pack_submission.py --team-id T05 --no-bbox --upload http://<IP_SERVER_LAB>:8765
```

Cờ `--no-bbox` ghi `KIT_DRAW_BBOX=0` vào `manifest.json`; viewer và file `video.mp4` vẫn có dải OSD (`H`, `v`, `pitch`, `N`, `E`, `boxes`, `wall`) nhưng không overlay bounding box trên ảnh.

Script sẽ tự động:
- Kiểm tra cú pháp toàn bộ file Python.
- Kiểm tra kích thước gói nộp $\le 30\text{ MB}$.
- Tạo ra 2 file nộp bài:
  - `submission_T05.tar.gz`
  - `submission_T05.sha256`

---

## 5. Nộp bài và Nhận kết quả tự động

1. **Nộp bài:**
   - **Cách 1 (CLI):** `python3 pack_submission.py --team-id T05 --upload http://<IP_SERVER_LAB>:8765`
   - **Cách 2 (Web Portal):** Truy cập web chấm thi của Lab AI, nhập mã đội và upload file `submission_T05.tar.gz`.
   - Khi upload được chấp nhận, server trả **Team Token** và CLI tự lưu vào `.drone_team_token`. Không gửi file này cho đội khác và không nộp kèm trong bài thi.

### Chốt phiên bản sau khi khóa upload

Sau khi BTC khóa cổng nộp bài, mỗi đội có 10 phút để chọn một phiên bản đã nộp:

```bash
python3 pack_submission.py --team-id T05 --select 1
# hoặc chọn bằng một phần hash / tên file archive:
python3 pack_submission.py --team-id T05 --select 20261016T014005Z_8cc8f110.tar.gz
```

CLI tự đọc `.drone_team_token` và URL server đã lưu từ lần upload trước. Nếu chạy trên máy khác, dùng:

```bash
python3 pack_submission.py --team-id T05 --select 1 --upload http://<IP_SERVER_LAB>:8765 --team-token tok_xxx
```

Hết 10 phút, server trả `423 Locked`; đội chưa chọn sẽ được BTC chạy fallback tự động sang bản có điểm Public cao nhất.

2. **Nhận kết quả (Sau 5 – 7 phút):**
   Hệ thống chạy bài nộp trong sandbox mô phỏng và trả về:
   - **`score.json`:** Bảng điểm chi tiết:
     - Điểm AI (Tối đa 50 điểm)
     - Điểm Tọa độ (Tối đa 30 điểm)
     - Điểm An toàn bay (Tối đa 20 điểm)
   - **`console.log`:** Nhật ký in `stdout/stderr` của code để đội kiểm tra lỗi.
   - **`video.mp4`:** Video góc nhìn từ camera drone kèm HUD thời gian thực (mặc định có bounding box YOLO; dùng `--no-bbox` khi pack nếu muốn ẩn khung).

### Xem điểm, log và video bài thi

Không cần đóng gói lại để xem kết quả của đội:

```bash
python3 pack_submission.py --team-id T05 --status http://127.0.0.1:8765
# Có thể thay URL bằng domain HTTPS Cloudflare do BTC cung cấp.
```

Lệnh in tổng điểm /100, điểm AI, tọa độ, an toàn, TP/FP/duplicates,
sim time, wall time, 20 dòng console cuối và liên kết video. Khi chưa chấm,
trạng thái là `pending`, các điểm chưa có hiển thị `—`.
Mở `<URL>/dashboard` để xem bảng xếp hạng và hoạt động cập nhật mỗi 4 giây.

Mỗi đội có **100 lượt nộp được chấp nhận mỗi ngày (UTC+7)**. Bài bị từ chối
không trừ lượt. Khi hết lượt, server trả HTTP 429; liên hệ BTC để cấp thêm.

### Tần suất upload và chế độ BTC

Mỗi đội được gửi tối đa **1 lần trong 60 giây** và **2 lần trong 300 giây**
(cửa sổ trượt, tính cả upload có gói không hợp lệ; request bị rate limit không
kéo dài thời gian chờ). HTTP 429 trả JSON `status: rejected` cùng số giây cần đợi.
Quota 100 bài accepted/ngày vẫn áp dụng riêng.

BTC sửa `grading/rate_limit_config.json` trên server; thay đổi có hiệu lực ở
request kế tiếp, không cần restart. `window_1m_max`, `window_5m_max` là số nguyên
dương; file thiếu/sai dùng mặc định 1 và 2, không bật bypass bằng key. Lịch sử
upload lưu trong `grading/rate_limit.db`, dùng chung cho các thread và sau restart.

```bash
python3 pack_submission.py --team-id T05 --upload http://<IP_SERVER_LAB>:8765 --admin-key <KEY_BTC>
```

Cờ trên gửi header `X-Admin-Key`; key đúng cấu hình hoặc peer trực tiếp là
`127.0.0.1`/`::1` sẽ bỏ giới hạn tần suất, vẫn kiểm tra gói, deadline và quota ngày.
Không chia sẻ key BTC cho sinh viên. Không dựa vào `X-Forwarded-For` để bypass.
Nếu reverse proxy chạy cùng máy và kết nối receiver bằng loopback, các upload qua
proxy cũng được bypass theo quy tắc peer trực tiếp; BTC cần chọn cấu hình mạng phù hợp.

### Chạy với môi trường native drone-setup

Tương thích giao thức với PX4 SITL v1.17.0 và ROS 2 Humble theo
`../260727-drone-setup/AGENT_SETUP.md`. Khởi động SITL theo `RUN.md`, source
`env.sh` và kích hoạt Python environment đã cài `pymavlink`, rồi chạy:

```bash
cd /path/to/student-starter-kit
python3 solution.py
```

`solution.py` nhận MAVLink bằng `udpin:0.0.0.0:14540`, cùng cổng với script bay
native `fly_around_30s.py`. Chỉ một chương trình điều khiển dùng cổng này tại một
thời điểm. Mã mẫu dùng MAVLink trực tiếp, không yêu cầu ROS node để điều khiển bay.

`report_person(...)` ưu tiên `KIT_REPORT_PATH` nếu được đặt. Nếu có mount
`/student`, mặc định ghi `/student/output/reports.jsonl`; trên native không có
`/student`, ghi `output/reports.jsonl` tương đối với thư mục đang chạy.

Dashboard hiển thị Slot 1/2, pha khởi động, bay với thời gian mô phỏng thực từ
telemetry, pha tính điểm/video và tên đội đang chờ. Bài nộp mới có toast và pulse.
Console mới có timestamp UTC từng dòng; console cũ chỉ có mốc bắt đầu lần chạy
(hoặc mốc sửa file), ghi rõ không biết thời điểm thực của từng dòng.
