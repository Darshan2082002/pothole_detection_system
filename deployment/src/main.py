import cv2
import time
from .detector import PotholeDetector
from .camera import Camera
from .utils import draw_boxes

def main():
    print("Starting application...")

    detector = PotholeDetector()
    print("Model loaded")

    camera = Camera()
    print("Camera initialized")

    prev_time = 0
    prev_time = 0

    while True:
        frame = camera.read()
        if frame is None:
            break

        boxes = detector.detect(frame)
        frame = draw_boxes(frame, boxes)

        # FPS calculation
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time else 0
        prev_time = current_time

        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Pothole Detection", frame)

        if cv2.waitKey(25) & 0xFF == 27:
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()