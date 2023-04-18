import streamlit as st
from PIL import Image
from clearml import Model
from keras.models import load_model
from utility import preprocess_image, OptimizedRounder
import numpy as np

st.set_page_config(
    page_title="Diabetic Retinopathy Detection App",
    page_icon=":eye:",
    layout="centered"
)
st.markdown(
    """
    <style>
    .main-container {
        padding-top: 0rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Set the title of the app
st.title('Diabetic Retinopathy Detection')


def load_keras_model(model_id="48a76b1277154398991d1d079db968ae"):
     model_path = Model(model_id="48a76b1277154398991d1d079db968ae").get_local_copy()
     model = load_model(model_path)
     return model

def process_image(img_path):
     img = preprocess_image(img_path)
     img = np.expand_dims(img, axis=0)
     return img

def predict(img_path):
     model = load_keras_model(model_id="48a76b1277154398991d1d079db968ae")
     data = process_image(img_path)
     pred = model.predict(data)
     return pred
     


with st.container():
        # display the resized image in Streamlit
        image = Image.open("resources/about_dr.jpeg")
        st.image(image, caption='About Diabetic Retinopathy', width=700)
        

# Create sidebar
st.sidebar.image('resources/JKVISION1.png', width=200)
st.sidebar.title("Diabetic Retinopathy Detection App")
st.sidebar.markdown("Upload a fundus image and the app will tell you at which stage of Diabetic Retinopathy the Patient is.")

# Create file uploader
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["png","jpg"]) 
import tempfile
import os

# create a temporary directory
temp_dir = tempfile.TemporaryDirectory()
file_path = ''
# save the uploaded file to the temporary directory
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    #image = image.resize(image.width/2,image.height/2)
    st.header('You Uploaded')
    with st.container():
        # display the resized image in Streamlit
        st.image(image, caption='Uploaded Image (Resized)', width=350)

    #st.image(image, caption='Uploaded fundus image.', use_column_width=True,width=100)
    file_path = os.path.join(temp_dir.name, uploaded_file.name)
    print(f'file_path {file_path}')
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

button = st.button("Detect 👁️  Diabetic Retinopathy")
button_css = """
    <style>
        div.stButton > button:first-child {
            background-color: Green;
            border: 10px;
            color: white;
            padding: 15px 32px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 32px;
            margin: 4px 2px;
            cursor: pointer;
        }
    </style>
"""
st.markdown(button_css, unsafe_allow_html=True)

with st.spinner('Detecting Retinopathy.........'):
    if button:
        # Check if image uploaded
        if uploaded_file is not None:
            # Make prediction
            prediction = predict(file_path)
            print(f'prediction = {prediction}')
            
            coef = [0.5, 1.5, 2.5, 3.5]

            if prediction < coef[0]:
                prediction = 0
            elif prediction >= coef[0] and prediction < coef[1]:
                prediction = 1
            elif prediction >= coef[1] and prediction < coef[2]:
                prediction = 2
            elif prediction >= coef[2] and prediction < coef[3]:
                prediction = 3
            else:
                prediction = 4


            if prediction == 0:
                st.success('Congratulations! No diabetic retinopathy detected in the image.')
            elif prediction == 1:
                st.warning('Careful, The Patient is in the First Stage(Mild) of DR.')
            elif prediction == 2:
                st.warning('Careful, The Patient is in the 2nd Stage(Moderate) of DR.')
            elif prediction == 3:
                st.error('Careful, The Patient is in the 3rd Stage(Severe) of DR.')
            elif prediction == 4:
                st.error('Careful, The Patient is in the 4th Stage(Proliferative) of DR.')
            # Display prediction
            # Your code here
        else:
            st.warning("Please upload an image first.")

col1,col2 =st.columns(2)
with col1:
        cmatrix = Image.open("resources/confusion_matrix_Binary.png")
        st.image(cmatrix, caption='Binary Confusion Matrix', width=350)
with col2:
        roc = Image.open("resources/ROC_AUC.png")
        st.image(roc, caption='ROC-AUC Curve', width=350,)