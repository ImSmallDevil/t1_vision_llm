# .\.venv-gpu\Scripts\python.exe ".\[ANSWER]_04_DL_01_ObjectDetect.py"

from ultralytics import YOLO
import cv2
import time


model = YOLO("src/models/YOLO/yolo11n.pt")
model.to("cuda")

camera_id = 1  # 기본 웹캠. 다른 카메라는 1, 2 등으로 변경
cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    raise RuntimeError(f"카메라를 열 수 없습니다: camera_id={camera_id}")

displayed_fps = 0.0

while True:
    start_time = time.perf_counter()
    ret, frame = cap.read()

    if not ret:
        print("카메라 프레임을 읽지 못했습니다.")
        break

    # Q 또는 ESC 키로 종료
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), 27):
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


    results = model.predict(
        source=frame,   # source image
        conf=0.25,      # Confidence Threshold
        iou=0.5,        # IoU Threshold
        verbose=False,  # no output prints
        classes=None,   # selected class
    )

    output_frame = results[0].plot()

    elapsed_time = time.perf_counter() - start_time
    current_fps = 1.0 / elapsed_time

    if displayed_fps == 0:
        displayed_fps = current_fps
    else:
        displayed_fps = 0.9 * displayed_fps + 0.1 * current_fps

    cv2.putText(output_frame, f"FPS: {displayed_fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("YOLO Object Detection with FPS", output_frame)

cap.release()
cv2.destroyAllWindows()


# =======================================================================
# from ultralytics import YOLO
# import cv2
# import time


# model = YOLO("src/models/YOLO/yolo11n_int8.engine")

# pipeline = (
#     "nvarguscamerasrc sensor-id=0 ! "
#     "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
#     "nvvidconv ! "
#     "video/x-raw, format=BGRx ! "
#     "videoconvert ! "
#     "video/x-raw, format=BGR ! "
#     "queue leaky=downstream max-size-buffers=1 ! "
#     "appsink drop=true max-buffers=1 sync=false"
# )

# cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

# displayed_fps = 0.0

# while True:
#     start_time = time.perf_counter()

#     ret, frame = cap.read()

#     if not ret:
#         break
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

#     results = model.predict(
#         source=frame,   # source image
#         conf=0.25,      # Confidence Threshold
#         iou=0.5,        # IoU Threshold
#         verbose=False,  # no output prints
#         classes=None,   # selected class
#     )

#     output_frame = results[0].plot()

#     elapsed_time = time.perf_counter() - start_time
#     current_fps = 1.0 / elapsed_time

#     if displayed_fps == 0:
#         displayed_fps = current_fps
#     else:
#         displayed_fps = 0.9 * displayed_fps + 0.1 * current_fps

#     cv2.putText(output_frame, f"FPS: {displayed_fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

#     cv2.imshow("YOLO Object Detection with FPS", output_frame)

# cap.release()
# cv2.destroyAllWindows()