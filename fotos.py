import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
import cv2

# Especificar la configuración del detector de objetos
options = vision.ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path="Models/efficientdet_lite0 (1).tflite"),
    max_results=5,
    score_threshold=0.2,
    running_mode=vision.RunningMode.IMAGE)
detector = vision.ObjectDetector.create_from_options(options)

# Leer la imagen de entrada
image = cv2.imread("data/miku.jfif")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_rgb = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

# Detectar objetos sobre la imagen
detection_result = detector.detect(image_rgb)
#print(detection_result)

for detection in detection_result.detections:
    # Bounding box
    bbox = detection.bounding_box
    bbox_x, bbox_y, bbox_w, bbox_h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height

    # Score y Category name
    category = detection.categories[0]
    score = category.score * 100
    category_name = category.category_name

    cv2.rectangle(image, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y - 30), (100, 255, 0), -1)
    cv2.rectangle(image, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), (100, 255, 0), 2)
    cv2.putText(image, f"{category_name}: {score:.2f}%", (bbox_x + 5, bbox_y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2)

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()