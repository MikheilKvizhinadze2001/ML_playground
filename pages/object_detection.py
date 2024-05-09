import streamlit as st
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
import requests
import time
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tempfile
from video_preprocessing import annotate_video, real_time_object_detection
import subprocess
import os
import tempfile
import base64
import cv2

# message if box is not found
no_box_found_message = '''
                        Model couldn't find any objects in the image 🥺. Your model is like a digital
                     ‘I Spy’ game. It’s trained to find certain objects in images. If it can’t
                     find anything, it might be because the objects are too small, the image is
                     too complex, or the model hasn’t been trained on these objects. So, it’s like
                     asking it to find a cat in a dog picture.
                     It won’t work because it’s only been trained to find cats! 😄
                        '''

def predict_and_detect(chosen_model, img, classes=[], conf=0.5):
    results = chosen_model.predict(img, conf=conf)
    if len(results[0].boxes) == 0:
        st.info(no_box_found_message)
        st.write("Current Image:")
        return img
# Create a loading message
loading_message = st.empty()

# Create a loading timer
start_time = time.time()

# Check if the model is already loaded
if "processor" not in st.session_state or "model" not in st.session_state:
    try:
        # Load the model and cache it to avoid loading it at each run
        loading_message.write("Loading models...")
        if "yolo_model" not in st.session_state:
            st.session_state.yolo_model = YOLO("yolov8n.pt")
        
        @st.cache_data(show_spinner=False)
        def load_model():
            processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
            model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
            return processor, model

        processor, model = load_model()
        st.session_state.processor = processor
        st.session_state.model = model

        # Display a success message when the model is loaded
        loading_message.write("Model loaded successfully!")
        end_time = time.time()
        loading_time = end_time - start_time
        st.write(f"Loading time: {loading_time:.2f} seconds")

    except Exception as e:
        # Display an error message if something goes wrong
        loading_message.write("Error loading model!")
        st.error(e)
else:
    loading_message.write("Model Loading!")


st.header("Welcome 😊! This is an object detection app using various models")

st.write("Think of object detection as a process of finding and classifying objects in an image. It is a computer vision task that deals with detecting instances of semantic objects of a certain class in digital images and videos. Object detection algorithms typically use machine learning or deep learning to detect an object in an image.")
st.write("Like this 👇")
st.image("pages/boxes.png")
st.write("To see how well the model can detect objects in an image, you can upload an image below.          Upload an image and the model will detect objects in it and put bounding boxes around them, bam.")

# Define allowed image types
ALLOWED_IMAGE_TYPES = ["jpg", "jpeg", "png"]  

# Create a file uploader widget
uploaded_file = st.file_uploader("Choose an image file", type=ALLOWED_IMAGE_TYPES) 

st.write("Now select the color for the bounding boxes and annotations, default is red.")


# Add a color picker widget
selected_color = st.color_picker("Select bounding box color", "#FF0000")  # Default to red
select_color_annotation = st.color_picker("Select annotation color", "#FF0000")  # Default to red

selected_color_bgr = tuple(int(selected_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
select_color_annotation_bgr = tuple(int(select_color_annotation.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
# Check if an image file was uploaded


st.write("Now, select the model which would predict the objects in the image and draw bounding boxes around them. You can choose between YOLO and DETR.")
st.write("""
        YOLO (You Only Look Once) is like a quick glance at a photo. It scans the entire image in one go and identifies objects. It’s super fast and can detect objects in real-time, which makes it great for applications like self-driving cars. However, it might miss some smaller or overlapping objects because of its speed.

On the other hand, DETR (Detection Transformer) is like a careful study of a painting. It uses transformers (the same technology behind many language models) to understand the ‘context’ of different parts of the image. This means it’s really good at handling complex scenes with lots of objects or where objects overlap. However, it’s a bit slower than YOLO.

So, if you need speed, go for YOLO. If you need to handle complex scenes, go for DETR. Either way, you’re in for an exciting journey into the world of object detection! 😄
         """)
# Create a radio button to choose between models
model_choice = st.radio("Choose a model", ["YOLO", "DETR"])

if uploaded_file is not None and st.button("Process Image"):
    file_details = uploaded_file.name.split(".")  # Split filename to get extension
    file_extension = file_details[-1].lower()   # Get extension in lowercase

    if file_extension in ALLOWED_IMAGE_TYPES:
        # Open the uploaded image file
        image = Image.open(uploaded_file)
        if model_choice == "DETR":
            # Preprocess the image and perform object detection
            inputs = st.session_state.processor(images=image, return_tensors="pt")
            outputs = st.session_state.model(**inputs)
            # Convert outputs to COCO API and filter detections with score > 0.9
            target_sizes = torch.tensor([image.size[::-1]])
            results = st.session_state.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
                    # Check if there are any detections
            if len(results["scores"]) == 0:
                st.info(no_box_found_message)
                st.write("Current Image: 👇")
            else:
                # Draw bounding boxes and annotations on the image
                draw = ImageDraw.Draw(image)
                for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                    box = [round(i, 2) for i in box.tolist()]
                    draw.rectangle(box, outline=selected_color)
                    draw.text((box[0], box[1]), f"{st.session_state.model.config.id2label[label.item()]} ({round(score.item(), 3)})", fill=select_color_annotation)

                # Display the image with bounding boxes and annotations
                st.success('Success!', icon="🎉")
                st.write(f"Here are {model_choice}'s predictions 👇")
                st.write('Decimal numbers in the annotations represent the confidence score of the model. The higher the score, the more confident the model is in its prediction.')
                st.image(image, use_column_width=True)
                st.write("But wait, how does the object detection work and what is DETR? 🤔")
        else:
            st.session_state.yolo_model = YOLO("yolov8n.pt")

            def predict_and_detect(chosen_model, img, classes=[], conf=0.5):
                results = chosen_model.predict(img, conf=conf)
                if len(results[0].boxes) == 0:
                    st.info(no_box_found_message)
                    st.write("Current Image:")
                    return img

                # Enhanced annotation and bounding box display
                for result in results:
                    for box in result.boxes:
                        label = f"{result.names[int(box.cls[0])]} {box.conf.item():.2f}"
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                        text_w, text_h = text_size

                        # Dynamic text placement
                        if box.xyxy[0][1] > text_h + 10:  # Enough space above
                            text_y = int(box.xyxy[0][1]) - 10
                        else:  # Place text below (inside the box)
                            text_y = int(box.xyxy[0][1]) + text_h + 5

                        cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                                    (int(box.xyxy[0][2]), int(box.xyxy[0][3])), selected_color_bgr, 2)
                        cv2.putText(img, label, (int(box.xyxy[0][0]), text_y),
                                    cv2.FONT_HERSHEY_PLAIN, 1, select_color_annotation_bgr, 1)

                # Potential image expansion
                padding = 20  # Adjust as needed
                h, w = img.shape[:2]
                need_top_padding = any(box.xyxy[0][1] - text_h < padding for box in result.boxes)
                # Add more conditions for other sides if needed (e.g., need_bottom_padding)

                if need_top_padding:
                    img = cv2.copyMakeBorder(img, padding, 0, 0, 0, cv2.BORDER_CONSTANT)

                st.success("Bam!")
                st.write(f"Here are {model_choice}'s predictions")
                st.write("Decimal numbers in the annotations represent the confidence score of the model. The higher the score, the more confident the model is in its prediction.")
                st.write("But wait, how does object detection work and what is YOLO? 🤔")
                return img

            image = np.array(image)  # Convert PIL Image to numpy array
            result_img = predict_and_detect(st.session_state.yolo_model, image, classes=[], conf=0.5)

            # Display the output image
            st.image(result_img, use_column_width=True)
            

    


    else:
        st.error(f"Invalid file type. Please upload a supported image format: {ALLOWED_IMAGE_TYPES}")
        


_to_be_continued = '''
                this is a placeholder for the continuation of the object detection explanation
                    '''



st.title("Object Tracking in Videos")
st.write("""
        Object tracking in videos is like a digital spotlight that follows a specific object as it moves through the video frames.
        It processes the video frame by frame, identifies the object, and then ‘tracks’ it throughout the video. The result is a new
        video where the tracked object is highlighted in each frame. It’s a fun and useful way to analyze videos! 🎥😄
        """)
st.write("To see how well the model can track objects in a video, you can upload a video below. Upload a video and the model will track objects in it and put bounding boxes around them, bam 😁")



# Create a file uploader widget
# Allow users to upload a video file
video_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

# Check if a video file was uploaded
if video_file is not None and st.button("Process Video"):
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(video_file.read())

    processed_video_path = annotate_video(st.session_state.yolo_model,tfile.name)

    # Open the processed video file in binary mode and read its content
    with open(processed_video_path, 'rb') as f:
        video_content = f.read()


    # Encode the video data to Base64
    encoded_video = base64.b64encode(video_content)

    # Decode Base64 to binary
    decoded_video = base64.b64decode(encoded_video)

    # Save the binary data to an .mp4 file
    output_file_path = "processed_video.mp4"
    with open(output_file_path, "wb") as output_file:
        output_file.write(decoded_video)

    # Display a download button for the saved .mp4 file
    st.download_button(label="Download video", data=decoded_video, file_name="processed_video.mp4", mime="video/mp4")



st.write("Below, you can toggle the button to start the real-time detection using your webcam. The model will detect objects in real-time and put bounding boxes around them 📷")



# Create a button to start the webcam
real_time_detection = st.checkbox("Start Real-Time Detection")

video_placeholder = st.empty()
run_detection = True  

while run_detection: 
    with video_placeholder.container():
        if real_time_detection:
            real_time_object_detection(st.session_state.yolo_model, video_placeholder)
        else:
            video_placeholder.empty()  
            run_detection = False


