from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "best.onnx"

# Model parameters
IMG_SIZE = 640
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45

# Class names
CLASS_NAMES = ["pothole"]

# Camera source (0 = default webcam)
CAMERA_SOURCE = 1
#CAMERA_SOURCE = "test_video.mp4"