# Height Measurement System (BTL_Nhung)

Hệ thống đo chiều cao bằng Computer Vision với kiến trúc `Mobile App (Expo) + Backend/API + AI Core`.

## 1. Mô Tả Dự Án

Dự án xây dựng giải pháp đo chiều cao tự động từ ảnh camera điện thoại, dùng thảm tham chiếu để quy đổi từ pixel sang cm.

Mục tiêu chính:
- Đo chiều cao nhanh bằng điện thoại iPhone (Expo app).
- Kiểm tra chất lượng ảnh và tư thế trước khi trả kết quả.
- Lưu lịch sử đo để theo dõi tăng trưởng theo thời gian.
- Mở rộng dần sang user/family management, admin dashboard và advice.

Giá trị MVP:
- Chạy demo end-to-end: chụp ảnh -> gửi backend -> AI xử lý -> trả `height_cm` + `confidence` + cảnh báo.

## 2. Kiến Trúc Tổng Quan

```text
iPhone Camera (Expo)
-> Capture 3-5 frames
-> FastAPI Backend (/measure)
-> AI Core Pipeline:
   Person Detection (YOLO)
   + Reference Mat Detection (HSV/YOLO custom)
   + Pose Estimation (MediaPipe)
   + Height Measurement
   + Quality Check
-> JSON Result
-> Mobile App hiển thị kết quả + lưu lịch sử
```

## 3. Tính Năng Hiện Có (Baseline)

- API backend FastAPI nhận ảnh đo chiều cao.
- Pipeline AI xử lý ảnh với:
  - detect người
  - detect thảm tham chiếu
  - pose estimation
  - ước lượng chiều cao
  - quality checks
- Mobile app Expo:
  - preview camera
  - chụp ảnh
  - gửi ảnh lên backend
  - hiển thị kết quả đo
- Lưu dữ liệu đo cơ bản và ảnh debug phục vụ đánh giá sai số.

## 4. WBS Dự Án

## 4.1 AI Core

### 4.1.1 Person Detection
- Dùng YOLO pretrained để detect `person`.
- Validate số người trong ảnh:
  - `0 người` -> lỗi.
  - `1 người` -> hợp lệ.
  - `>1 người` -> lỗi.
- Export `bbox` người.

### 4.1.2 Reference Mat Detection
- MVP: detect thảm bằng HSV + rectangle detection.
- Giai đoạn sau: train YOLO custom class `reference_mat`.
- Input kích thước thảm:
  - `100x100 cm`
  - `50x100 cm`
  - `custom size`
- Export `bbox/corners` của thảm.

### 4.1.3 Pose Estimation
- Dùng MediaPipe Pose.
- Detect keypoints:
  - `head/nose`
  - `shoulder`
  - `hip`
  - `knee`
  - `ankle`
  - `heel`
- Validate visibility của keypoints.

### 4.1.4 Height Measurement
- Tính tỷ lệ `pixel/cm` từ reference mat.
- Đo khi người đứng thẳng trước camera.
- Multi-segment placeholder:
  - `head -> nose`
  - `nose -> hip`
  - `hip -> knee`
  - `knee -> ankle`
  - `ankle -> heel`
- Output:
  - `height_cm`
  - `confidence`
  - `warning_message`

### 4.1.5 Quality Check
- Ảnh có đúng `1 người`.
- Có đúng `1 thảm`.
- Thấy đủ toàn thân.
- Ảnh không quá mờ.
- Tư thế đứng hợp lệ.
- Camera không nghiêng quá nhiều.

### 4.1.6 Debug & Evaluation
- Lưu ảnh debug có bbox/keypoints.
- Lưu JSON kết quả.
- Test sai số với chiều cao thật.
- Tính `MAE` / `error %`.

## 4.2 Mobile App

### 4.2.1 Camera Module
- Kết nối camera iPhone qua Expo.
- Preview realtime.
- Chụp `3-5` frame.
- Resize/compress ảnh.
- Chọn frame tốt nhất hoặc gửi từng frame.

### 4.2.2 AI Core Integration
- Gửi ảnh sang AI Core API.
- Nhận kết quả:
  - chiều cao
  - trạng thái đo
  - cảnh báo
- Hiển thị kết quả cho user.

### 4.2.3 Measurement Flow
- Chọn người cần đo.
- Chọn loại thảm/kích thước thảm.
- Bấm `Bắt đầu đo`.
- Chụp frame.
- Trả kết quả.
- Lưu lịch sử đo.

### 4.2.4 User App Functions
- Đăng ký / đăng nhập.
- Hồ sơ cá nhân.
- Quản lý thành viên gia đình.
- Thêm trẻ/người thân.
- Xem lịch sử đo.
- Xem biểu đồ tăng trưởng.
- Nhận lời khuyên cải thiện chiều cao.

### 4.2.5 Admin Functions
- Quản lý user.
- Quản lý hồ sơ đo.
- Xem thống kê hệ thống.
- Quản lý nội dung lời khuyên.
- Quản lý lỗi đo / feedback.

## 4.3 Backend / API

### 4.3.1 Auth API
- Login/register.
- JWT/session.
- Role: `admin/user`.

### 4.3.2 User API
- CRUD user profile.
- CRUD family members.
- Lưu thông tin:
  - tên
  - tuổi/ngày sinh
  - giới tính
  - chiều cao hiện tại
  - cân nặng (nếu có)

### 4.3.3 Measurement API
- Upload image/frame.
- Gọi AI Core.
- Lưu kết quả đo.
- Trả kết quả về app.

### 4.3.4 Advice API
- Rule-based advice theo:
  - tuổi
  - giới tính
  - chiều cao
  - lịch sử tăng trưởng
- Placeholder AI advice cho giai đoạn sau.

### 4.3.5 History API
- Lịch sử đo.
- Biểu đồ chiều cao theo thời gian.
- Export báo cáo.

## 4.4 Database

### 4.4.1 Tables / Collections
- `users`
- `family_members`
- `measurements`
- `measurement_images`
- `advice_articles`
- `admin_logs`

### 4.4.2 Measurement Record (Sample)

```json
{
  "user_id": "...",
  "member_id": "...",
  "height_cm": 170.5,
  "confidence": 0.87,
  "mat_size": "100x100",
  "image_url": "...",
  "created_at": "..."
}
```

## 5. Cấu Trúc Dự Án

```text
height_measurement_system/
├── ai_core/
│   └── ai-service/          # AI service tối giản
│       ├── checkpoints/
│       │   ├── person_detection/   # yolov8n.pt
│       │   ├── mat_detection/      # best.pt, last.pt
│       │   └── pose_mediapipe/     # ghi chú/runtime assets cho MediaPipe Pose
│       └── tools/                  # train/eval/prepare scripts
├── app/                     # FastAPI backend + measurement pipeline
│   ├── api.py
│   ├── pipeline.py
│   ├── measurement_core.py
│   ├── pose_estimator.py
│   ├── pose_validator.py
│   ├── calibration.py
│   ├── height_estimator.py
│   ├── measurement_storage.py
│   └── main.py
├── mobile-app/              # Expo React Native app
│   ├── App.js
│   ├── package.json
│   └── app.json
├── future_modules/          # Module planned cho phase sau
├── requirements.txt         # Python dependencies
└── README.md
```

### 5.1 Checkpoint Paths (Current)

- Person detection: `ai_core/ai-service/checkpoints/person_detection/yolov8n.pt`
- Mat detection: `ai_core/ai-service/checkpoints/mat_detection/best.pt`
- Mat detection (last): `ai_core/ai-service/checkpoints/mat_detection/last.pt`
- Pose: `ai_core/ai-service/checkpoints/pose_mediapipe/` (MediaPipe không dùng `.pt` local trong project này)

## 6. Yêu Cầu Môi Trường

- Python `3.10+` (khuyến nghị 3.10 hoặc 3.11)
- Node.js `18+`
- npm `9+`
- iPhone có cài Expo Go (nếu test mobile thật)
- Máy tính và điện thoại cùng mạng LAN khi test API local

## 7. Hướng Dẫn Chạy Dự Án

## 7.1 Chạy Backend (FastAPI)

1. Cài dependencies Python:

```bash
pip install -r requirements.txt
```

2. Chạy API server:

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

3. Kiểm tra API:
- Swagger UI: `http://localhost:8000/docs`
- Health endpoint (nếu có): `http://localhost:8000/health`

## 7.2 Chạy Mobile App (Expo)

1. Vào thư mục mobile app:

```bash
cd mobile-app
npm install
npx expo start
```

2. Mở Expo Go trên iPhone và quét QR code.

3. Cấu hình backend URL trong app:
- Không dùng `localhost` trên iPhone.
- Dùng IP LAN của máy chạy backend, ví dụ: `http://192.168.1.100:8000`.

## 7.3 Flow Đo Cơ Bản

1. Mở app, chọn/nhập thông tin đo.
2. Chụp `3-5` frame.
3. App gửi ảnh sang API backend.
4. AI Core xử lý và trả kết quả.
5. App hiển thị:
- chiều cao ước lượng
- confidence
- cảnh báo nếu ảnh/tư thế không hợp lệ

## 8. Milestone 7 Ngày MVP

| Ngày | Mục tiêu |
|---|---|
| 1 | Setup AI Core + YOLO detect person |
| 2 | Detect reference mat bằng HSV |
| 3 | Tích hợp MediaPipe Pose |
| 4 | Tính chiều cao cơ bản |
| 5 | Expo app chụp 3-5 frame gửi AI Core |
| 6 | Lưu lịch sử đo + UI kết quả |
| 7 | Test sai số + fix lỗi + demo |

## 9. Priority Hiện Tại

Thực hiện theo thứ tự:
1. AI Core detect person
2. AI Core detect mat
3. Pose estimation
4. Height measurement
5. Expo capture image
6. Send image to AI Core
7. Save history
8. User/family/admin/advice

## 10. Ghi Chú Triển Khai

- Nên thêm `.gitignore` để loại `__pycache__/`, `node_modules/`, ảnh output/debug và file lớn khỏi git.
- Với production, nên tách:
  - AI inference service
  - API business service
  - storage (DB + object storage)
- Nên chuẩn hóa format JSON response để mobile xử lý ổn định.

