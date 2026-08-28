# .\.venv-gpu\Scripts\python.exe ".\[ANSWER]_04_DL_01_ObjectDetect.py"

from ultralytics import YOLO
import cv2


model = YOLO("src/models/YOLO/yolo11n.pt")
model.to("cuda")

camera_id = 0  # 기본 웹캠. 다른 카메라는 1, 2 등으로 변경
cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    raise RuntimeError(f"카메라를 열 수 없습니다: camera_id={camera_id}")

while True:
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

    cv2.imshow("YOLO Object Detection", output_frame)

cap.release()
cv2.destroyAllWindows()