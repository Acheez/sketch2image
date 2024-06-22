import streamlit as st
import tensorflow as tf
import numpy as np
import os
from models import get_model  # Ensure you have a function to load the sketch model
from config import CHECKPOINT_DIR  # Add SKETCH_CHECKPOINT_DIR to your config
from utils import load_image, generate_images, generate_sketches  # Ensure you have a function to generate sketches
from PIL import Image

TEMP_DIR = "temp_images"

if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# Load models
try:
    model = get_model(CHECKPOINT_DIR)
    print("Model loaded successfully")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Sidebar
st.sidebar.title("Image to Sketch and Sketch to Image")
st.sidebar.write("Upload an image and the model will generate a corresponding sketch or image.")

# Main title and description
st.title("Image to Sketch and Sketch to Image Generator")
st.write("Upload your image below, and the machine learning model will generate a new sketch or image based on your upload. You can compare the original image with the generated output.")

# Create two tabs for different functionalities
tab1, tab2 = st.tabs(["Sketch to Image", "Image to Sketch"])

with tab1:
    st.header("Sketch to Image")
    uploaded_sketch_file = st.file_uploader("Choose a sketch image...", type=["jpg", "jpeg", "png"], key="sketch")

    if uploaded_sketch_file is not None:
        # Display progress spinner
        with st.spinner('Processing your sketch image...'):
            # Save uploaded file to temporary directory
            file_path = os.path.join(TEMP_DIR, uploaded_sketch_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_sketch_file.getbuffer())

            # Load and preprocess the image
            image = load_image(file_path)
            image = np.expand_dims(image, axis=0)

            # Generate image using the model
            prediction = generate_images(model.generator, image)
            generated_image = (prediction + 1) / 2.0  # Rescale to [0, 1]
            image = (image[0] + 1) / 2.0

            # Display images side by side
            col1, col2 = st.columns(2)

            with col1:
                st.header("Original Sketch")
                st.image(image, use_column_width=True)

            with col2:
                st.header("Generated Image")
                st.image(generated_image.numpy(), use_column_width=True)

            # Clean up
            os.remove(file_path)

with tab2:
    st.header("Image to Sketch")
    uploaded_real_file = st.file_uploader("Choose a real image...", type=["jpg", "jpeg", "png"], key="real")

    if uploaded_real_file is not None:
        # Display progress spinner
        with st.spinner('Processing your real image...'):
            # Save uploaded file to temporary directory
            file_path = os.path.join(TEMP_DIR, uploaded_real_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_real_file.getbuffer())

            # Load and preprocess the image
            image = load_image(file_path)
            image = np.expand_dims(image, axis=0)

            # Generate sketch using the model
            prediction = generate_sketches(file_path)  # Ensure you have a generate_sketches function
            # generated_sketch = (prediction + 1) / 2.0  # Rescale to [0, 1]
            generated_sketch=np.clip(prediction/ 255.0,0,1)
            image = (image[0] + 1) / 2.0

            # Display images side by side
            col1, col2 = st.columns(2)

            with col1:
                st.header("Original Image")
                st.image(image, use_column_width=True)

            with col2:
                st.header("Generated Sketch")
                st.image(generated_sketch, use_column_width=True)

            # Clean up
            os.remove(file_path)
