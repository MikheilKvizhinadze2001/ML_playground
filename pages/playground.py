import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

model = YOLO("yolov8n.pt")

def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)

    return results


def predict_and_detect(chosen_model, img, classes=[], conf=0.5):
    results = predict(chosen_model, img, classes, conf=conf)

    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), 2)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
    return img, results


# read the image
image = cv2.imread("pages/boxes.png")

result_img, results = predict_and_detect(model, image, classes=[], conf=0.5)

# Display the output image
st.image(result_img, use_column_width=True)

# Display the bounding boxes and annotations
for result in results:
    for box in result.boxes:
        st.write(f"Object: {result.names[int(box.cls[0])]}")
        st.write(f"Confidence: {box.conf.item():.2f}")
        st.write(f"Bounding Box: ({int(box.xyxy[0][0])}, {int(box.xyxy[0][1])}) - ({int(box.xyxy[0][2])}, {int(box.xyxy[0][3])})")