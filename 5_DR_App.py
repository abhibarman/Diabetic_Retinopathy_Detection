import streamlit as st
from PIL import Image
from clearml import Model
from keras.models import load_model
from utility import preprocess_image, OptimizedRounder
import numpy as np
from keras.utils import load_img, img_to_array
from keras.applications.efficientnet_v2 import preprocess_input
import shap
from lime import lime_image
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from utility import mapping

st.set_page_config(
    page_title="Diabetic Retinopathy Detection App",
    page_icon=":eye:",
    layout="centered"
)
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)
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
#show metrics 
show_metrics_on_load = True

def load_keras_model(model_id="48a76b1277154398991d1d079db968ae"):
     model_path = Model(model_id="48a76b1277154398991d1d079db968ae").get_local_copy()
     print(f'model_path {model_path}')
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

def explain_prediction(file_path):
     name = file_path.split('/')[-1].split('.')[0]
     model_path = Model(model_id="48a76b1277154398991d1d079db968ae").get_local_copy()
     model = load_model(model_path)
     image_path = file_path
     x_val = np.empty((1, 224, 224, 3), dtype=np.uint8)
     img = load_img(image_path, target_size=(224, 224))
     img_array = img_to_array(img)
     img_array = preprocess_input(img_array)

     prediction = model.predict(x_val)
     print(f'Model Prediction: {prediction}')

     optR = OptimizedRounder()
     coefficients = [0.49964604, 1.55479703, 2.4369177,  3.26701671] # Learnt while training the efficient net model.
     print(f'Coefficients: {coefficients}')
     prediction = optR.predict(prediction, coefficients)
     #print('Optirmized Prediction :{y_val_pred}')
     print(prediction)

     x_val[0,:,:,:] = img_array

     explainer = lime_image.LimeImageExplainer()
     explanation = explainer.explain_instance(x_val[0], model.predict, top_labels=1, hide_color=0, num_samples=1000)
     temp, mask = explanation.get_image_and_mask(0, positive_only=False, num_features=10, hide_rest=False)
     fig, axe = plt.subplots(figsize=(7, 3.5))
     plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
     #axe.text(180, 190, 'Predicted:'+mapping.get(int(prediction[0].item())), bbox=dict(facecolor='red', alpha=0.0))
     plt.show()
     name += '_lime.png'
     fig.savefig(name)
   

     #
     mapping = {0:'No DR',1:'Mild',2:'Moderate',3:'Severe',4:'Proliferative DR'} 
     #masker = shap.maskers.Image("inpaint_telea", demo_x_val[0].shape)
     #explainer = shap.Explainer(model, masker, output_names=list(mapping.keys()))
     #shap_values = explainer(demo_x_val[0:], max_evals=500, batch_size=50, outputs=shap.Explanation.argsort.flip[:1])
     #
     masker = shap.maskers.Image('inpaint_telea',x_val[0].shape)
     explainer = shap.Explainer(model, masker,output_names=list(mapping.keys()))

     # here we use 500 evaluations of the underlying model to estimate the SHAP values
     shap_values = explainer(x_val, max_evals=1000, batch_size=50, outputs=shap.Explanation.argsort.flip[:1])
     #shap.image_plot(shap_values, x_val)
     shap.image_plot(shap_values[0])
     st.pyplot()
     #print(shap_values.shape)
     #return shap_values

     
     return name

     


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
button_clicked = False
with st.spinner('Detecting Retinopathy.........'):
    show_metrics_on_load = False
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
        
        if uploaded_file is not None:
             button_clicked = True


if uploaded_file is not None:
     if button_clicked:
          with st.spinner('Generating Explanation.........'):          
            print(f'file_path {file_path}')
            img_path = explain_prediction(file_path)
            image = Image.open(img_path)
            st.image(image, caption='Lime Explanation', width=500)
          



st.write("<div style='height: 50px;'></div>", unsafe_allow_html=True)
col1,col2 =st.columns(2)
if show_metrics_on_load:     
    with col1:
            cmatrix = Image.open("resources/confusion_matrix_Binary.png")
            st.image(cmatrix, caption='Binary Confusion Matrix', width=350)
    with col2:
            roc = Image.open("resources/ROC_AUC.png")
            st.image(roc, caption='ROC-AUC Curve', width=350,)