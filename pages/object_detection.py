import streamlit as st
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image, ImageDraw
import requests



# load the model and cache it to avoid loading it at each run
@st.cache_data()
def load_model():
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    return processor, model

processor, model = load_model()



st.header("Welcome 😊! This is an object detection app using DETR (DEtection TRansformer) model")

st.write("Think of object detection as a process of finding and classifying objects in an image. It is a computer vision task that deals with detecting instances of semantic objects of a certain class in digital images and videos. Object detection algorithms typically use machine learning or deep learning to detect an object in an image.")
st.image("pages/boxes.png")
st.write("Upload an image and the model will detect objects in it and put bounding boxes around them.")
st.write("Allowed image formats: jpg, jpeg, png")

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
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # Convert outputs to COCO API and filter detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        # Draw bounding boxes and annotations on the image
        draw = ImageDraw.Draw(image)
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            draw.rectangle(box, outline=selected_color)
            draw.text((box[0], box[1]), f"{model.config.id2label[label.item()]} ({round(score.item(), 3)})", fill=select_color_annotation)

        # Display the image with bounding boxes and annotations
        st.success('Success!', icon="🎉")
        st.image(image, use_column_width=True)


    else:
        st.error(f"Invalid file type. Please upload a supported image format: {ALLOWED_IMAGE_TYPES}")

st.write("But wait, how does object detection work and what is DETR? 🤔")
