# height_measurement_system

MVP hiện dùng kiến trúc `iPhone + Expo Go + FastAPI` với baseline `thảm xanh HSV + MediaPipe Pose + multi-segment`, rồi mở rộng dần sang pose correction, temporal filter, và hybrid ML correction.

## Baseline pipeline

```text
iPhone Camera
-> Expo App
-> Camera preview
-> Capture frame
-> POST /measure-height
-> Detect green mat
-> MediaPipe keypoints
-> Multi-segment height
-> Quality check
-> Hybrid correction placeholder
-> Return result to app
```

## Current structure

```text
height_measurement_system/
├── app/
│   ├── api.py
│   ├── camera_service.py
│   ├── pose_estimator.py
│   ├── pose_validator.py
│   ├── calibration.py
│   ├── height_estimator.py
│   ├── measurement_storage.py
│   ├── pipeline.py
│   └── main.py
├── future_modules/
│   ├── multi_segment_estimator.py
│   ├── sitting_pose_estimator.py
│   ├── leaning_pose_correction.py
│   ├── depth_estimation.py
│   └── temporal_filter.py
├── mobile-app/
│   ├── App.js
│   ├── app.json
│   ├── babel.config.js
│   └── package.json
├── requirements.txt
└── README.md
```

## Stack

- Frontend: React Native + Expo
- Camera: `expo-camera`
- Backend: FastAPI
- AI xử lý: OpenCV + MediaPipe
- Demo app: Expo Go bằng QR code

## Current baseline

### Stage 1 - Rule-based / Explainable

- Camera image
- Detect green mat
- MediaPipe keypoints
- Multi-segment height
- `height_raw`

### Stage 2 - Quality check

- Mat quality
- Pose visibility
- Body tilt
- Head confidence
- Full-body visibility

### Stage 3 - ML correction model

Input:
- `height_raw`
- segment lengths `h1-h5`
- pose angles
- keypoint visibility
- mat detection quality
- image crop or skeleton image

Output:
- `height_correction`
- `confidence_score`

Final:
- `height_final = height_raw + height_correction`

## Backend modules

- `app/api.py`: API `/health` và `/measure` nhận ảnh từ iPhone app.
- `app/pipeline.py`: pipeline xử lý frame dùng lại cho API và CLI.
- `app/measurement_core.py`: green-mat HSV detection và multi-segment height.
- `app/hybrid_correction_model.py`: placeholder cho hybrid correction model.
- `app/pose_estimator.py`: chạy MediaPipe Pose và trích xuất landmark theo pixel.
- `app/pose_validator.py`: chỉ cho phép đo khi người đứng thẳng, đủ khung hình.
- `app/calibration.py`: đổi pixel sang cm từ chiều cao pixel của thảm xanh.
- `app/measurement_storage.py`: lưu ảnh chụp và lịch sử đo vào `data/`.
- `app/main.py`: chế độ test cục bộ bằng camera trực tiếp trên máy phát triển.

## Mobile app

- `mobile-app/App.js`: preview camera, đổi camera trước/sau, chụp ảnh, gửi backend, hiển thị kết quả.
- Backend URL mặc định đang là `http://192.168.1.100:8000`.
- Trên iPhone thật, phải đổi URL này thành IP LAN thực của máy chạy backend Python.

## Run

1. Chạy backend Python:

```bash
pip install -r requirements.txt
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

2. Chạy Expo app:

```bash
cd mobile-app
npm install
npx expo start
```

3. Mở Expo Go trên iPhone và quét QR code.

4. Trong app, nhập:
- `Backend URL`: ví dụ `http://192.168.1.100:8000`
- `Reference cm`: chiều cao thật của thảm xanh
- `Reference px`: có thể để trống, backend sẽ tự detect thảm xanh bằng HSV

## Baseline rule

- Nếu đứng thẳng -> đo.
- Nếu nghiêng/cúi/ngồi -> cảnh báo.
- Sau này -> thay cảnh báo bằng module ước lượng nâng cao.

## 7-day plan

- Day 1: dựng Expo app, quét QR trên iPhone, mở camera iPhone.
- Day 2: gửi ảnh từ app sang backend Python.
- Day 3: tích hợp OpenCV + MediaPipe Pose.
- Day 4: kiểm tra tư thế đứng thẳng.
- Day 5: calibration bằng vật tham chiếu.
- Day 6: tính chiều cao và hoàn thiện UI kết quả.
- Day 7: test sai số, chỉnh tham số, chuẩn bị demo.

## Planned placeholders

- `leaning_pose_correction.py`: hiệu chỉnh khi người nghiêng nhẹ.
- `multi_segment_estimator.py`: cộng các đoạn cơ thể khi người cúi hoặc ngồi.
- `depth_estimation.py`: hỗ trợ khi camera không cố định hoặc cần hiệu chỉnh phối cảnh.
- `temporal_filter.py`: làm mượt kết quả bằng moving average hoặc Kalman filter.
- `app/hybrid_correction_model.py`: về sau dùng MobileNet hoặc model tương tự để học correction/error hoặc đo trực tiếp khi bỏ thảm.

## Notes

- MVP hiện dùng iPhone camera là nhanh nhất để demo trong 7 ngày.
- Giai đoạn sau có thể thay đầu vào bằng IP camera, camera ngoài, hoặc RTSP stream.
- Baseline hiện tại: nếu không detect được thảm xanh hoặc pose không đủ tốt thì backend sẽ trả cảnh báo thay vì cố đo.
- Hybrid future: có thể học `error = manual_height - rule_based_height` để suy ra `height_final`.
# BTL_Nhung
