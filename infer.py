from clearml import Model
import pandas as pd
from keras.models import load_model
import numpy as np
from utility import preprocess_image, OptimizedRounder

# LOAD MODEL
model_path = Model(model_id="48a76b1277154398991d1d079db968ae").get_local_copy()
model = load_model(model_path)
print(model.summary())

mapping = {0:'No DR',1:'Mild',2:'Moderate',3:'Severe',4:'Proliferative DR'}

#LOAD & PREPARE DATA
data = pd.read_csv('jarvis_data/names.txt')
data['diagnosis'] = data.diagnosis.astype(float)
y_true = data.diagnosis.values.tolist()
x_val = np.empty((data.shape[0], 224, 224, 3), dtype=np.uint8)
counter = 0
multiplier = 0
incr = 1
for i,file in enumerate(data.id_code.values): 
    if i ==0:
        multiplier = 5 
    if i !=0 and i % multiplier ==0:
        counter += 1 
        incr += 1
        multiplier = multiplier + 5 * incr 
    file_path = 'jarvis_data/Train/'+ mapping.get(counter)+'/'+file
    x_val[i,:,:,:] = preprocess_image(file_path)
    

#MAKE PREDICTION
print(x_val.shape)
y_pred = model.predict(x_val)
print(f'Model Prediction: {y_pred}')

optR = OptimizedRounder()
optR.fit(y_pred, y_true)
coefficients = optR.coefficients()
print(f'Coefficients: {coefficients}')
y_val_pred = optR.predict(y_pred, coefficients)
y_val_pred  = np.squeeze(y_val_pred)
df = pd.DataFrame({'Actual':y_true,'Predicted':y_val_pred})
print(df)
df.to_csv('inference_result.csv', index=False)
#df['diagnosis'] = df['diagnosis'].astype(float)




