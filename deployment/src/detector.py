import cv2
import numpy as np
import onnxruntime as ort
from .config import MODEL_PATH, IMG_SIZE, CONF_THRESHOLD

class PotholeDetector:
    def __init__(self):
        # Initialize inference session on CPU
        self.session = ort.InferenceSession(
            str(MODEL_PATH),
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, frame):
        """
        Prepares the raw OpenCV frame for the ONNX model.
        """
        # 1. Resize to model's expected input (e.g., 640x640)
        img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        # 2. Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 3. Normalize pixel values to [0, 1]
        img = img.astype(np.float32) / 255.0
        # 4. Change data layout from HWC to CHW
        img = np.transpose(img, (2, 0, 1))
        # 5. Add batch dimension: (1, 3, 640, 640)
        img = np.expand_dims(img, axis=0)
        return img

    def postprocess(self, frame, outputs):
        """
        Converts raw model output into clean bounding boxes.
        """
        # YOLO outputs are usually (1, 5, 8400) or (1, 8400, 5)
        # We ensure it is (8400, 5) where 5 is [x, y, w, h, conf]
        preds = outputs[0]
        if preds.shape[1] < preds.shape[2]:
            preds = np.transpose(preds, (0, 2, 1))
        preds = preds[0]

        orig_h, orig_w = frame.shape[:2]
        boxes = []
        confidences = []

        # Calculate scaling factors
        x_factor = orig_w / IMG_SIZE
        y_factor = orig_h / IMG_SIZE

        for pred in preds:
            conf = pred[4] # The 5th element is confidence

            # Filter out low-confidence "noise"
            if conf >= CONF_THRESHOLD:
                x_center, y_center, w, h = pred[0], pred[1], pred[2], pred[3]

                # Convert center-based coords to top-left corner coords
                # and scale back to the original image resolution
                left = int((x_center - w / 2) * x_factor)
                top = int((y_center - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                boxes.append([left, top, width, height])
                confidences.append(float(conf))

        # NON-MAXIMUM SUPPRESSION (NMS)
        # This is the most important part to fix your "Red Screen"
        # It removes overlapping boxes. 0.45 is a balanced IoU threshold.
        indices = cv2.dnn.NMSBoxes(
            boxes, 
            confidences, 
            CONF_THRESHOLD, 
            0.45
        )

        final_boxes = []
        if len(indices) > 0:
            # Flatten indices for the loop
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                # Return format: (x1, y1, x2, y2, confidence)
                final_boxes.append((x, y, x + w, y + h, confidences[i]))

        return final_boxes

    def detect(self, frame):
        """
        Full detection pipeline.
        """
        input_tensor = self.preprocess(frame)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        return self.postprocess(frame, outputs)