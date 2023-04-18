import numpy as np
import pandas as pd
import cv2
from functools import partial
from sklearn.metrics import cohen_kappa_score
import scipy
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import EfficientNetV2S
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
import os
import seaborn as sns
from PIL import Image


mapping = {0:'No DR',1:'Mild',2:'Moderate',3:'Severe',4:'Proliferative DR'} 

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
    
def load_data(folder='/content/dataset/APTOS2019/'):
    train = pd.read_csv(folder+'train.csv')
    test = pd.read_csv(folder+'test.csv')
    
    train_dir = os.path.join(folder,'train_images/')
    test_dir = os.path.join(folder,'test_images/')
    
    train['file_path'] = train['id_code'].map(lambda x: os.path.join(train_dir,'{}.png'.format(x)))
    test['file_path'] = test['id_code'].map(lambda x: os.path.join(test_dir,'{}.png'.format(x)))
    
    train['file_name'] = train["id_code"].apply(lambda x: x + ".png")
    test['file_name'] = test["id_code"].apply(lambda x: x + ".png")
    
    train['diagnosis'] = train['diagnosis'].astype(str)
    
    return train,test

def plot_classes(df):
    df_group = pd.DataFrame(df.groupby('diagnosis').agg('size').reset_index())
    df_group.columns = ['diagnosis','count']

    sns.set(rc={'figure.figsize':(10,5)}, style = 'whitegrid')
    sns.barplot(x = 'diagnosis',y='count',data = df_group,palette = "Blues_d")
    plt.title('Output Class Distribution')
    plt.legend()
    plt.show() 

'''This Function converts a color image to gray scale image'''

def conv_gray(img, IMG_SIZE = 200):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
    return img
  



'''
This Function shows the visual Image photo of 'n x 5' points (5 of each class)
'''
def visualize_imgs(df,pts_per_class,color_scale,IMG_SIZE = 200, title = "Sample Training Images"):
    df = df.groupby('diagnosis',group_keys = False).apply(lambda df: df.sample(pts_per_class))
    df = df.reset_index(drop = True)
    
    plt.rcParams["axes.grid"] = False
    mapping = {'0':'No DR','1':'Mild','2':'Moderate','3':'Severe','4':'Proliferative DR'} 
    #plt.title(title)
    for pt in range(pts_per_class):
        f, axarr = plt.subplots(1,5,figsize = (15,15))
        for ax in axarr:
            ax.set_axis_off()
        #axarr[0].set_ylabel("Sample Data Points")
        
        df_temp = df[df.index.isin([pt + (pts_per_class*0),pt + (pts_per_class*1), pt + (pts_per_class*2),pt + (pts_per_class*3),pt + (pts_per_class*4)])]
        for i in range(5):
            if color_scale == 'gray':
                img = conv_gray(cv2.imread(df_temp.file_path.iloc[i]))
                axarr[i].set_axis_on()
                #axarr[i].set_title(title)
                axarr[i].imshow(img,cmap = color_scale)
            else:
                axarr[i].set_axis_on()
                #axarr[i].set_title(title)
                axarr[i].imshow(Image.open(df_temp.file_path.iloc[i]).resize((IMG_SIZE,IMG_SIZE)))
            axarr[i].set_xlabel(mapping.get(df_temp.diagnosis.iloc[i]))

        plt.title(title)
        plt.show()

   
    
def circle_crop(img, sigmaX):   
    """
    Create circular crop around image centre    
    """    
    img = crop_image_from_gray(img)    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    height, width, depth = img.shape    
    
    x = int(width/2)
    y = int(height/2)
    r = np.amin((x,y))
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    img=cv2.addWeighted(img,4, cv2.GaussianBlur( img , (0,0) , sigmaX) ,-4 ,128)
    return img 


def preprocess_display_samples(df, columns=4, rows=3,im_size = 224):
    fig=plt.figure(figsize=(5*columns, 4*rows))

    for i in range(columns*rows):
        image_path = df.loc[i,'file_path']
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

def display_samples(df, columns=4, rows=3,im_size = 224):
    fig=plt.figure(figsize=(5*columns, 4*rows))

    for i in range(columns*rows):
        image_path = df.loc[i,'file_path']
        #print(f'image_path:{image_path}')
        image_id = df.loc[i,'diagnosis']
        img = cv2.imread(f'{image_path}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = crop_image_from_gray(img)
        img = cv2.resize(img, (im_size,im_size))
        #img = cv2.addWeighted(img,4,cv2.GaussianBlur(img, (0,0), im_size/40) ,-4 ,128)
        
        fig.add_subplot(rows, columns, i+1)
        plt.title(image_id)
        plt.imshow(img)
    
    plt.tight_layout()
