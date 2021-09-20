
import torch

from IPython.display import Image, clear_output  # to display images
# from utils.google_utils import gdrive_download  # to download models/datasets

# clear_output()
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))

# Commented out IPython magic to ensure Python compatibility.
import json
import math
import os

import cv2
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
import scipy
from tqdm import tqdm
import gc
from functools import partial
from sklearn import metrics
from collections import Counter
import json
import itertools
import numpy as np
from sklearn.model_selection import GroupKFold
# from keras.applications.resnet50 import ResNet50
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
import efficientnet.tfkeras as efn 
# %matplotlib inline

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

"""# Loading & Preprocessing"""
k = 5
image_dir = '/content/drive/MyDrive/pkm/stress detection/dataset/wawancara/spectogram_pcen_15w_cropped'
df_dir = '/content/drive/MyDrive/pkm/stress detection/dataset/wawancara/df_wawancara.xlsx'
best_model_filepath=f'/content/drive/MyDrive/pkm/stress detection/dataset/wawancara/efficientnetb0_15w_2_{k}.best.hdf5'
last_model_filepath = f'f"/content/drive/MyDrive/pkm/stress detection/dataset/wawancara/efficientnetb0_15w_2_{k}.last.hdf5"'
normal_list = glob(f'{image_dir}/normal_*/*')
stress_list = glob(f'{image_dir}/stress_*/*')

df_label = pd.read_excel(df_dir)
df_label = df_label[~df_label['skor_dass'].isna()]
df_label['status'] = df_label.skor_dass.apply(lambda p: 'normal' if p<=14 else 'stress')
df_label['id_file'] = df_label.apply(lambda x:x['status']+'_'+x['nama_file'].split('.')[0],axis = 1)
df_label.sort_values('id_file')
df_file = pd.DataFrame(normal_list+stress_list,columns = ['filename'])
df_file['id_file'] = df_file.filename.apply(lambda p: p.split('/')[-2].split('.')[0])
df_file = df_file.merge(df_label,on='id_file')

#Transfer 'jpg' images to an array IMG
def Dataset_loader(image_list):
    IMG = []
    read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
    for IMAGE_NAME in tqdm(image_list):
      img = read(IMAGE_NAME)
      img = cv2.resize(img, (800,160))
#             img = segment_image(img)
      IMG.append(np.array(img))
    return np.array(IMG)

X = Dataset_loader(list(df_file.filename))
y = df_file.status.apply(lambda p: 0 if p=='normal' else 1)
y = tf.keras.utils.to_categorical(y, num_classes= 2)

"""# Create Label"""


groups = df_file.id_file
group_kfold = GroupKFold(n_splits=5)
group_kfold.get_n_splits(X, y, groups)
print(group_kfold)

i = 0
for train_index, test_index in group_kfold.split(X, y, groups):
  i+=1
  if i==5:
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_valid = X[train_index], X[test_index]
    y_train, y_valid = y[train_index], y[test_index]
    del X
    break

"""# Model:"""

from tensorflow.keras import regularizers
def build_model(backbone, lr=1e-4):
    model = tf.keras.Sequential()
    model.add(backbone)
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(512, activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(512, activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(512, activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(512, activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
    model.add(tf.keras.layers.Dropout(0.5))
    
    # model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(2, activation='sigmoid'))
    
    return model

efficientnetb0 = efn.EfficientNetB0(
        # weights='imagenet',
        weights=None,
        input_shape=(160,800,3),
        include_top=False
                   )

model = build_model(efficientnetb0)
model.summary()

from tensorflow.keras.optimizers import Adam,RMSprop,SGD,Nadam
os.environ['TF_KERAS'] = '1'

model.compile(
        loss='binary_crossentropy',
        optimizer = Adam(learning_rate = 1e-3),
        metrics=['accuracy']
    )

# Learning Rate Reducer
learn_control = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', 
                                  patience=5,
                                  verbose=1,
                                  factor=0.2, 
                                  min_lr=1e-7)

# Checkpoint
checkpoint = tf.keras.callbacks.ModelCheckpoint(best_model_filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

"""# Training & Evaluation"""

BATCH_SIZE = 32
history = model.fit(
    x = X_train,
    y = y_train,
    validation_data= (X_valid,y_valid),
    callbacks=[learn_control,checkpoint],
    batch_size = BATCH_SIZE,
    epochs=30,
)

model.save_weights(last_model_filepath)