import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict
import time

model = YOLO("yolov8n.pt")

# Function to annotate video
@st.cache_data
def annotate_video(video_path):
    track_history = defaultdict(lambda: [])

    names = model.model.names
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"
    # Get video information
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    # Define the codec and create VideoWriter object
    result = cv2.VideoWriter("object_tracking.mp4",
                        cv2.VideoWriter_fourcc(*'XVID'),
                        fps,
                        (w, h))

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model.track(frame, persist=True, verbose=False)
            boxes = results[0].boxes.xyxy.cpu()

            if results[0].boxes.id is not None:

                # Extract prediction results
                clss = results[0].boxes.cls.cpu().tolist()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                confs = results[0].boxes.conf.float().cpu().tolist()

                # Annotator Init
                annotator = Annotator(frame, line_width=2)

                for box, cls, track_id in zip(boxes, clss, track_ids):
                    annotator.box_label(box, color=colors(int(cls), True), label=names[int(cls)])

                    # Store tracking history
                    track = track_history[track_id]
                    track.append((int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)))
                    if len(track) > 30:
                        track.pop(0)

                    # Plot tracks
                    points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.circle(frame, (track[-1]), 7, colors(int(cls), True), -1)
                    cv2.polylines(frame, [points], isClosed=False, color=colors(int(cls), True), thickness=2)

            result.write(frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    result.release()
    cap.release()
    cv2.destroyAllWindows()
    return "object_tracking.mp4"  # Return the file path

@st.cache_data
def real_time_object_detection(_model, _video_placeholder):  # Add video_placeholder parameter
    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        raise Exception("Cannot open camera")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        # Perform object detection with your model
        results = model(frame, augment=False)

        # Draw bounding boxes and labels (modify as needed)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{result.names[int(box.cls)]}: {box.conf.item():.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

        fps = 1 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        cv2.imshow("Real-Time Object Detection", frame)
        _video_placeholder.image(frame, channels="BGR")  # Update Streamlit placeholder

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()