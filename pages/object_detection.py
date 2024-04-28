import streamlit as st
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image, ImageDraw
import requests
import time
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Create a loading message
loading_message = st.empty()

# Create a progress bar
progress_bar = st.progress(0)

# Create a loading timer
start_time = time.time()

# Check if the model is already loaded
if "processor" not in st.session_state or "model" not in st.session_state:
    try:
        # Load the model and cache it to avoid loading it at each run
        @st.cache_data(show_spinner=False)
        def load_model():
            processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
            model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
            return processor, model

        # Update the progress bar and loading timer as the model loads
        for i in range(100):
            progress_bar.progress(i+1)
            loading_message.write(f"Loading model... {i+1}% complete")
            time.sleep(0.1)  # Simulate loading time
            if i == 99:
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
    loading_message.write("Model load")


st.header("Welcome 😊! This is an object detection app using DETR (DEtection TRansformer) model")

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
# Check if an image file was uploaded

if uploaded_file is not None and st.button("Process Image"):
    file_details = uploaded_file.name.split(".")  # Split filename to get extension
    file_extension = file_details[-1].lower()   # Get extension in lowercase

    if file_extension in ALLOWED_IMAGE_TYPES:
        # Open the uploaded image file
        image = Image.open(uploaded_file)

        # Preprocess the image and perform object detection
        inputs = st.session_state.processor(images=image, return_tensors="pt")
        outputs = st.session_state.model(**inputs)
        # Convert outputs to COCO API and filter detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        results = st.session_state.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
                # Check if there are any detections
        if len(results["scores"]) == 0:
            st.info("""
                    Model couldn't find any objects in the image 🥺. Your model is like a digital
                     ‘I Spy’ game. It’s trained to find certain objects in images. If it can’t
                     find anything, it might be because the objects are too small, the image is
                     too complex, or the model hasn’t been trained on these objects. So, it’s like
                     asking it to find a cat in a dog picture.
                     It won’t work because it’s only been trained to find cats! 😄
                    """)
        else:
            # Draw bounding boxes and annotations on the image
            draw = ImageDraw.Draw(image)
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                draw.rectangle(box, outline=selected_color)
                draw.text((box[0], box[1]), f"{st.session_state.model.config.id2label[label.item()]} ({round(score.item(), 3)})", fill=select_color_annotation)

            # Display the image with bounding boxes and annotations
            st.success('Success!', icon="🎉")
            st.image(image, use_column_width=True)

    else:
        st.error(f"Invalid file type. Please upload a supported image format: {ALLOWED_IMAGE_TYPES}")

        

st.write("But wait, how does object detection work and what is DETR? 🤔")


