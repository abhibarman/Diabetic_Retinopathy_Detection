from clearml import Model, Logger
import pandas as pd
from keras.models import load_model
import numpy as np
from utility import preprocess_image, OptimizedRounder
from clearml import Task, TaskTypes
from PIL import Image
import os
import shap
from lime import lime_image
import matplotlib.pyplot as plt
from keras.utils import img_to_array,load_img
from skimage.segmentation import mark_boundaries


task = Task.init(project_name='Diabetic_Retinopathy_Detection', 
                 task_name='Retinopathy Inference', 
                 task_type=TaskTypes.inference,
                 reuse_last_task_id=False
                 )

# Create an input model using the ClearML ID of a model already registered in the ClearML platform
model_path = Model(model_id="48a76b1277154398991d1d079db968ae").get_local_copy()
model = load_model(model_path)
print(model.summary())

data = pd.read_csv('inference_Data/names.csv')
y_true = data.diagnosis.values.tolist()
ids = data.id_code.values.tolist()

x_val = np.empty((data.shape[0], 224, 224, 3), dtype=np.uint8)
for i,file in enumerate(data.id_code.values):    
    # add and upload Image (stored as .png file)
    im = Image.open(os.path.join('inference_Data',file))
    #task.upload_artifact('Input Image', im)

    Logger.current_logger().report_image("image", "image PIL", iteration=0, image=im)

    x_val[i,:,:,:] = preprocess_image('inference_Data/'+file)

print(x_val.shape)
print(x_val[:1:3].shape)
print(x_val[1:3].shape)
print(x_val[0].shape)
y_pred = model.predict(x_val)
print(f'Model Prediction: {y_pred}')

optR = OptimizedRounder()
optR.fit(y_pred, y_true)
coefficients = optR.coefficients()
print(f'Coefficients: {coefficients}')
y_val_pred = optR.predict(y_pred, coefficients)
y_val_pred  = np.squeeze(y_val_pred)

df = pd.DataFrame({'Actual':y_true,'Predicted':y_val_pred, 'Id_Code':ids})
print(df)
df.to_csv('model_inference.csv', index=False)


masker = shap.maskers.Image('inpaint_telea',x_val[0].shape)
explainer = shap.Explainer(model, masker)
# here we use 500 evaluations of the underlying model to estimate the SHAP values
shap_values = explainer(x_val, max_evals=500)
shap.image_plot(shap_values)

#LIME 

# Create a LIME explainer
explainer = lime_image.LimeImageExplainer()
for i in range(x_val.shape[0]):
    # Explain the predictions made by the model
    explanation = explainer.explain_instance(x_val[i], model.predict, top_labels=1, hide_color=0, num_samples=1000)
    temp, mask = explanation.get_image_and_mask(0, positive_only=False, num_features=10, hide_rest=False)
    fig, axe = plt.subplots(figsize=(7, 3.5))
    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    if y_true[i]:
        axe.text(180, 180, 'Actual:'+y_true[i], bbox=dict(facecolor='red', alpha=0.0))
    axe.text(180, 190, 'Predicted:'+y_val_pred[i], bbox=dict(facecolor='red', alpha=0.0))
    plt.show()
