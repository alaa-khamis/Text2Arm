from ultralytics import YOLO
import cv2
import torch

class YOLOv8Detector:
    def __init__(self, model_path="yolov8x.pt"):
        self.model = YOLO(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.model.eval()

    def detect_objects(self, rgb_image, target_objects=None):
        # Convert RGB to BGR for OpenCV compatibility
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        # Detect objects
        if target_objects:
            # Mapping class names to indices
            class_name_to_index = {v: k for k, v in self.model.model.names.items()}
            target_class_indices = [class_name_to_index[obj] for obj in target_objects]
            results = self.model.predict(source=bgr_image, classes=target_class_indices, save=False, verbose=False, device=self.device)
        else:
            # Detect all objects
            results = self.model.predict(source=bgr_image, classes=None, save=False, verbose=False, device=self.device)

        # Annotate the image with detection results
        annotated_image = results[0].plot()

        # Convert back to RGB
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        return annotated_image, results
