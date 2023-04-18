# Basic Libs..
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import os
print('CWD is ',os.getcwd())

# Vis Libs..
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["axes.grid"] = False

import os

#
from utility import load_data, plot_classes,visualize_imgs, circle_crop
from clearml import Task, TaskTypes

task = Task.init(project_name='Diabetic_Retinopathy_Detection', 
                 task_name='EDA_Diabetic_Retinopathy_Detection', 
                 task_type=TaskTypes.data_processing,
                 reuse_last_task_id=False
                 )

""" os.chdir('/content/initial_data')
print("We are currently in the folder of ",os.getcwd()) """
IMG_SIZE = 224


df_train,df_test = load_data('')
print(df_train.shape,df_test.shape,'\n')

plot_classes(df_train)

visualize_imgs(df_train,3,color_scale = None)
""" 
visualize_imgs(df_train,2,color_scale = 'gray')


'''
This section of code applies gaussian blur on top of image
'''

rn = np.random.randint(low = 0,high = len(df_train) - 1)

img = cv2.imread(df_train.file_path.iloc[rn])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))

img_t = cv2.addWeighted(img,4, cv2.GaussianBlur(img , (0,0) , 30) ,-4 ,128)

f, axarr = plt.subplots(1,2,figsize = (11,11))
axarr[0].imshow(img)
axarr[1].imshow(img_t)
plt.title('After applying Gaussian Blur')
plt.show()

'''Perform Image Processing on a sample image'''

rn = np.random.randint(low = 0,high = len(df_train) - 1)

#img = img_t
img = cv2.imread(df_train.file_path.iloc[rn])
img_t = circle_crop(img,sigmaX = 30)

f, axarr = plt.subplots(1,2,figsize = (11,11))
axarr[0].imshow(cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),(IMG_SIZE,IMG_SIZE)))
axarr[1].imshow(img_t)
plt.title('After applying Circular Crop and Gaussian Blur')
plt.show()
 """
task.close()