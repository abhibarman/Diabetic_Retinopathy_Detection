# To have reproducible results and compare them
nr_seed = 2019
import numpy as np
np.random.seed(nr_seed)
import tensorflow as tf
tf.random.set_seed(nr_seed)
import tensorflow as tf

from functools import partial

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# import libraries
#from tqdm import tqdm_notebook
import gc
import warnings

import cv2

import pandas as pd
import scipy
import matplotlib.pyplot as plt

from keras.applications import EfficientNetV2S
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import cohen_kappa_score

from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.applications import EfficientNetV2S

from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

from clearml import Task, TaskTypes
task = Task.init(project_name='Diabetic_Retinopathy_Detection',task_name='DRD_EfficientNet', task_type=TaskTypes.training)

warnings.filterwarnings("ignore")

def crop_image1(img,tol=7):
    #This function takes an input image img and a tolerance value tol. 
    #It creates a boolean mask where pixel values greater than tol are set to True, 
    #and all other pixel values are set to False.
    #It returns the cropped image where rows and columns with all False values are removed. 
        
    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]

def crop_image_from_gray(img,tol=7):
    #crops an RGB image based on a grayscale version of the image and a tolerance value.
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img

def preprocess_image(image_path, desired_size=224):
    #reads, crops, resizes, and applies a weighted sum and Gaussian blur to an input image.
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_image_from_gray(img)
    img = cv2.resize(img, (desired_size,desired_size))
    img = cv2.addWeighted(img,4,cv2.GaussianBlur(img, (0,0), desired_size/30) ,-4 ,128)
    return img

def preprocess_image_old(image_path, desired_size=224):
    # reads, resizes, and applies a weighted sum and Gaussian blur to an input image.
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = crop_image_from_gray(img)
    img = cv2.resize(img, (desired_size,desired_size))
    img = cv2.addWeighted(img,4,cv2.GaussianBlur(img, (0,0), desired_size/40) ,-4 ,128)
    
    return img

def display_samples(df, columns=4, rows=3):
    fig=plt.figure(figsize=(5*columns, 4*rows))

    for i in range(columns*rows):
        image_path = df.loc[i,'id_code']
        #print(f'image_path:{image_path}')
        image_id = df.loc[i,'diagnosis']
        img = cv2.imread(f'{image_path}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = crop_image_from_gray(img)
        img = cv2.resize(img, (im_size,im_size))
        img = cv2.addWeighted(img,4,cv2.GaussianBlur(img, (0,0), im_size/40) ,-4 ,128)
        
        fig.add_subplot(rows, columns, i+1)
        plt.title(image_id)
        plt.imshow(img)
    
    plt.tight_layout()


class Metrics(Callback):

    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data
        self.val_kappas = []

    def on_epoch_end(self, epoch, logs={}):
        X_val, y_val = self.validation_data
        
        y_pred = self.model.predict(X_val)
        
        coef = [0.5, 1.5, 2.5, 3.5]

        for i, pred in enumerate(y_pred):
            if pred < coef[0]:
                y_pred[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                y_pred[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                y_pred[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                y_pred[i] = 3
            else:
                y_pred[i] = 4

        _val_kappa = cohen_kappa_score(
            y_val,
            y_pred, 
            weights='quadratic'
        )

        self.val_kappas.append(_val_kappa)

        print(f"val_kappa: {_val_kappa:.4f}")
        
        if _val_kappa == max(self.val_kappas):
            print("Validation Kappa has improved. Saving model.")
            self.model.save('model.h5')

        return
    
def create_datagen():
    return ImageDataGenerator(
        horizontal_flip = True,
        vertical_flip = True,
        rotation_range = 160,
        zoom_range=0.35
    )   


def build_model(im_size ):
        efficientnet = EfficientNetV2S(
            include_top=False,
            weights="imagenet",
            input_shape=(im_size,im_size,3))
        
        model = tf.keras.models.Sequential()

        model.add(efficientnet)
        model.add(GlobalAveragePooling2D())
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='linear'))
        # model.add(Lambda(lambda x: x * 200.0)) output scaling

        model.compile(
            loss='mean_squared_error',
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            metrics=['mae']
        )

        return model


class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4

        ll = cohen_kappa_score(y, X_p, weights='quadratic')
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = scipy.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')
        print(-loss_partial(self.coef_['x']))

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        return self.coef_['x']
    

# Set Hyperparameters
im_size = 224
BATCH_SIZE = 16

#LOAD CSVs
new_train = pd.read_csv('/content/dataset/APTOS2019/train.csv')
old_train = pd.read_csv('/content/dataset/Diabetic Retinopathy (resized)/trainLabels.csv')
print(new_train.shape)
print(old_train.shape)

#set the same column names for both the dataframes
old_train.columns = new_train.columns
old_train.diagnosis.value_counts()
old_train.head()

#add the physical path to the images
new_train['id_code'] = '/content/dataset/APTOS2019/train_images/' + new_train['id_code'].astype(str) + '.png'
old_train['id_code'] = '/content/dataset/Diabetic Retinopathy (resized)/resized_train/resized_train/' + old_train['id_code'].astype(str) + '.jpeg'

train_df = old_train.copy()
val_df = new_train.copy()
train_df.head()

# Let's shuffle the datasets
train_df = train_df.sample(frac=1).reset_index(drop=True)
val_df = val_df.sample(frac=1).reset_index(drop=True)
print(train_df.shape)
print(val_df.shape)

#VISUALIZE FEW IMAGES SAMPLES
display_samples(train_df)

# ceate the validation dataset
N = val_df.shape[0]
x_val = np.empty((N, im_size, im_size, 3), dtype=np.uint8)
for i, image_id in enumerate(val_df['id_code']):
    preprocessed_image = preprocess_image(f'{image_id}', desired_size=im_size)
    x_val[i, :, :, :] = preprocessed_image

#Assign "diagnosis' values 0,1,2,3,4 from train and validation to y_train and y_val numpy arrays.
y_train = train_df['diagnosis'].values
y_val = val_df['diagnosis'].values

print(y_train.shape)
print(x_val.shape)
print(y_val.shape)

# delete the uneeded df
del new_train
del old_train
del val_df
gc.collect()

#!pip install -q efficientnet

#BUILD MODEL
model = build_model(im_size=im_size)
#model.summary()

#CREATE TRAINING DATA BUCKETS
num_bucket = 4
div = round(train_df.shape[0]/num_bucket)

results = pd.DataFrame({
                        'val_loss': [0.0],
                        'val_mean_absolute_error': [0.0],
                        'loss': [0.0], 
                        'mean_absolute_error': [0.0],
                        'bucket': [0.0]
                        })

# I found that changing the nr. of epochs for each bucket helped in terms of performances
epochs = [5,5,10,10]#10,15,15,15,20,25]
kappa_metrics = Metrics(validation_data=(x_val, y_val))
kappa_metrics.val_kappas = []

#START TRAINING
for i in range(0,num_bucket):
    if i != (num_bucket-1):
        print("Bucket Nr: {}".format(i))
        
        N = train_df.iloc[i*div:(1+i)*div].shape[0]
        x_train = np.empty((N, im_size, im_size, 3), dtype=np.uint8)
        for j, image_id in enumerate(train_df.iloc[i*div:(1+i)*div,0]):
            x_train[j, :, :, :] = preprocess_image_old(f'{image_id}', desired_size = im_size)

        data_generator = create_datagen().flow(x_train, y_train[i*div:(1+i)*div], batch_size=BATCH_SIZE)
        print('STARTING TRAINING')
        history = model.fit(
                        data_generator,
                        steps_per_epoch=x_train.shape[0] / BATCH_SIZE,
                        epochs=epochs[i],
                        validation_data=(x_val, y_val),
                        callbacks=[kappa_metrics],
                        max_queue_size=2, 
                        workers=4,
                        use_multiprocessing=True,
                        )
        
        dic = history.history
        df_model = pd.DataFrame(dic)
        df_model['bucket'] = i
    else:
        print("Bucket Nr: {}".format(i))
        
        N = train_df.iloc[i*div:].shape[0]
        x_train = np.empty((N, im_size, im_size, 3), dtype=np.uint8)
        for j, image_id in enumerate(train_df.iloc[i*div:,0]):
            x_train[j, :, :, :] = preprocess_image_old(f'{image_id}', desired_size = im_size)
        data_generator = create_datagen().flow(x_train, y_train[i*div:], batch_size=BATCH_SIZE)
        print('STARTING TRAINING-ELSE BLOCK')
        history = model.fit(
                        data_generator,
                        steps_per_epoch=x_train.shape[0] / BATCH_SIZE,
                        epochs=epochs[i],
                        validation_data=(x_val, y_val),
                        callbacks=[kappa_metrics],
                        max_queue_size=2, 
                        workers=4,
                        use_multiprocessing=True,
                        )
        
        dic = history.history
        df_model = pd.DataFrame(dic)
        df_model['bucket'] = i

    results = results.append(df_model)
    
    del data_generator
    del x_train
    gc.collect()
    
    print('-'*40)

backup = results
results = results.iloc[1:]
results['kappa'] = kappa_metrics.val_kappas
results = results.reset_index()
results = results.rename(index=str, columns={"index": "epoch"})
print(f'max(results.kappa) :{max(results.kappa)}')


results[['loss', 'val_loss']].plot()
results[['mean_absolute_error', 'val_mean_absolute_error']].plot()
results[['kappa']].plot()
results.to_csv('model_results.csv',index=False)


model.load_weights('model.h5')
y_val_pred = model.predict(x_val)

optR = OptimizedRounder()
optR.fit(y_val_pred, y_val)
coefficients = optR.coefficients()
print(f'Coefficients: {coefficients}')
y_val_pred = optR.predict(y_val_pred, coefficients)

score = cohen_kappa_score(y_val_pred, y_val, weights='quadratic')

print('Optimized Validation QWK score: {}'.format(score))
print('Not Optimized Validation QWK score: {}'.format(max(results.kappa)))

model.save('final_model.h5')



# Load the model and predict on the validation data
model.load_weights('model.h5')
y_val_pred = model.predict(x_val)

# Instantiate an OptimizedRounder object and fit on the validation data
optR = OptimizedRounder()
optR.fit(y_val_pred, y_val)
coefficients = optR.coefficients()
print(f'Coefficients: {coefficients}')

# Get the predicted labels using the optimized coefficients
y_val_pred_rounded = optR.predict(y_val_pred, coefficients)


# Compute confusion matrix
cm = confusion_matrix(y_val, y_val_pred_rounded)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', ax=ax)

# Add labels, title, and axis ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['No-DR', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 'PDR'])
ax.yaxis.set_ticklabels(['No-DR', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 'PDR'])
plt.show()


# Map 0 to 'No DR' and 1,2,3,4 to 'DR' in y_true
y_true_binary = np.where(y_val == 0, 0, 1)
y_val_pred_rounded_binary = np.where(y_val_pred_rounded == 0, 0, 1)

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_true_binary, y_val_pred_rounded_binary, pos_label=1)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(loc="lower right")
plt.show()

# Compute confusion matrix
cm = confusion_matrix(y_true_binary, y_val_pred_rounded_binary)

# Calculate specificity and sensitivity
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)

# Display confusion matrix with percentages
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', ax=ax)

# Add labels, title, and axis ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix (Specificity={:.2f}, Sensitivity={:.2f})'.format(specificity, sensitivity))
ax.xaxis.set_ticklabels(['No-DR', 'DR'])
ax.yaxis.set_ticklabels(['No-DR', 'DR'])
plt.show()


task.close()