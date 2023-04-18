from clearml import Model, Logger
import pandas as pd
from keras.models import load_model
import numpy as np
from utility import preprocess_image, OptimizedRounder
from clearml import Task, TaskTypes
from PIL import Image
import os

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


