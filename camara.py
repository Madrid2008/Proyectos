import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
import cv2
import time
detection_result_list = []
def detection_callback(result, output_image, timestamp_ms):
     detection_result_list.append(result)
options = vision.ObjectDetectorOptions(
     base_options=BaseOptions(model_asset_path="Models/efficientdet_lite0.tflite"),
     max_results=5,
     score_threshold=0.15,
     running_mode=vision.RunningMode.LIVE_STREAM, 
     result_callback=detection_callback)
detector = vision.ObjectDetector.create_from_options(options)
cap = cv2.VideoCapture(0)  # Intentamos sin especificar el backend
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara")
    exit()
while True:
     ret, frame = cap.read()
     if ret == False:
          break
     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
     frame_rgb = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
     detection_result = detector.detect_async(frame_rgb, time.time_ns() // 1_000_000)
     if detection_result_list:
          for detection in detection_result_list[0].detections:   
               bbox = detection.bounding_box
               bbox_x, bbox_y, bbox_w, bbox_h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
               category = detection.categories[0]
               score = category.score * 100
               category_name = category.category_name
               cv2.rectangle(frame, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y - 30), (100, 255, 0), -1)
               cv2.rectangle(frame, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), (100, 255, 0), 2)
               cv2.putText(frame, f"{category_name} {score:.2f}%", (bbox_x + 5, bbox_y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                         0.6, (255, 255, 255), 2)
          cv2.imshow('Video', frame)
          detection_result_list.clear()
     if cv2.waitKey(1) & 0xFF == 27:
          break
cap.release()
cv2.destroyAllWindows()