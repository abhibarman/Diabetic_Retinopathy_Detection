from clearml import Dataset
from pathlib import Path
import os

data_dir = './jarvis_data/Train'

ds = Dataset.create(dataset_name='APTOS_DRD_Dataset',dataset_project='Diabetic_Retinopathy_Detection')
ds.add_files(path=data_dir)

counts = []
folders = sorted(os.listdir(data_dir))

for folder in folders:
    counts.append([len(os.listdir(data_dir +'/'+ folder))])

ds.get_logger().report_histogram(
    title='Dataset Statistics',
    series='Train Data',
    labels=folders,
    values=counts
)

ds.upload()
ds.finalize()