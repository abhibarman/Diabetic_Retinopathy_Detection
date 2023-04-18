import pandas as pd
import shutil
import os

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

for i in range(5):
    df = train[train.diagnosis==i].head(5*(i+1))
    for code in df.id_code.values:
        fileName ='train_images/' + code +'.png'
        outpath = 'jarvis_data/Train/'+str(i)+'/'
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        outpath = outpath+code +'.png'
        shutil.copy(fileName, outpath)





