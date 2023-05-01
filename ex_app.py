import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from utility import preprocess_image
import os
import time

# Create header
st.title("Diabetic Retinopathy Detection App")

# Create sidebar
st.sidebar.title("Diabetic Retinopathy Detection App")
st.sidebar.markdown("Upload a fundus image and the app will predict whether there is diabetic retinopathy present.")

# Create file uploader
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["png"])

# Load the image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded fundus image.', use_column_width=True)
    st.write(np.asarray(image).shape)
# Preprocess the image
if uploaded_file is not None:
    # Save the uploaded image to a temporary directory
    with open(os.path.join(os.curdir, 'temp', uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    # Display the uploaded image
    #st.image(uploaded_file, caption='Uploaded fundus image', use_column_width=True)
    # Preprocess the uploaded image
    preprocessed_image = preprocess_image(os.path.join('temp', uploaded_file.name))
    print(preprocess_image)

