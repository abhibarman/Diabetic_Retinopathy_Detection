import pandas as pd
import matplotlib.pyplot as plt
import cv2
import shutil




train = pd.read_csv('train.csv')
df0 = train[train['diagnosis'] ==0].head(3)
df1 = train[train['diagnosis'] ==1].head(3)
df2 = train[train['diagnosis'] ==2].head(3)
df3 = train[train['diagnosis'] ==3].head(3)
df4 = train[train['diagnosis'] ==4].head(3)
df= pd.concat([df0,df1,df2,df3,df4])
train = df
print(df.shape)
train.to_csv('initial_data/train.csv',index=False)
test = pd.read_csv('test.csv').head(5)
test.to_csv('initial_data/test.csv',index=False)

for code in train.id_code.values.tolist():
    train_file_path = 'train_images/' + code + '.png'
    shutil.copyfile(train_file_path, 'initial_data/train_images/'+code+'.png')
for code in test.id_code.values.tolist():
    train_file_path = 'test_images/' + code + '.png'
    shutil.copyfile(train_file_path, 'initial_data/test_images/'+code+'.png')

""" print(train_file_path)

    fig = plt.figure(figsize=(25, 16))
    image = cv2.imread(train_file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) """