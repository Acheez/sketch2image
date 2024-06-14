import streamlit as st
import tensorflow as tf
import numpy as np
import os
from models import get_model
from config import CHECKPOINT_DIR
from utils import load_image, generate_images
from PIL import Image

TEMP_DIR = "temp_images"

if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

try:
    model = get_model(CHECKPOINT_DIR)
    print("Model loaded successfully")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Sidebar
st.sidebar.title("Sketch to Image")
st.sidebar.write("Upload a sketch image and the model will generate a corresponding image.")

# Main title and description
st.title("Sketch to Image Generator")
st.write("Upload your sketch below, and the machine learning model will generate a new image based on your sketch. You can compare the original sketch with the generated image.")

# File uploader
uploaded_file = st.file_uploader("Choose a sketch image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display progress spinner
    with st.spinner('Processing your image...'):
        # Save uploaded file to temporary directory
        file_path = os.path.join(TEMP_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

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
