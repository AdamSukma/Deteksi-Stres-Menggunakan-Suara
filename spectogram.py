import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
from tqdm import tqdm
from glob import glob
import librosa
import librosa.display
from PIL import Image 
import PIL
import cv2
dir = f'/content/drive/MyDrive/pkm/stress detection/dataset/wawancara/spectogram_pcen_{num_word}w'
wav_dir = '/content/drive/MyDrive/pkm/stress detection/dataset/wawancara/wav'
window_length = 20
num_word = 15
if not os.path.exists(dir):
  os.mkdir(dir)
for filename in tqdm(glob(f'{wav_dir}/*')[:]):
  if not os.path.exists(f'{dir}/{filename.split("/")[-1][:-4]}'):
    os.mkdir(f'{dir}/{filename.split("/")[-1][:-4]}')
  y, sr = librosa.load(filename)
  S1 = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
                                      fmax=10000)
  Dp1 = librosa.pcen(S1 * (2**31), sr=sr, gain=1.0, hop_length=512, bias=2, power=0.5, time_constant=0.8, eps=1e-06, max_size=2)
  word_arr = []
  c = 0;
  i=0
  index = []
  word_index = [0]
  flag = False
  while i in range(Dp1.shape[1]-window_length):
    rata = Dp1[:,i:i+window_length].mean()
    word_arr+=[rata]
    index+=[i]
    if(rata>0.2):
      if(flag==False):
        c+=1
        flag = True
        word_index += [i]
    if(rata<0.1):
      flag=False
    i+=1
  
  for index in range(1,len(word_index[1:-num_word])):
      fig, ax = plt.subplots(1,1,figsize=(num_word*5,3))
      img = librosa.display.specshow(Dp1[:,word_index[index]:word_index[index+num_word]], x_axis='time',
                              y_axis='mel', sr=sr,
                              fmax=10000, ax=ax)
      ax.set(title='Mel-frequency spectrogram')
      fig.colorbar(img, ax=ax, format='%+2.0f dB')
      img = Image.fromarray(Dp1, 'RGB')
      img.save(f'{dir}/{filename.split("/")[-1][:-4]}/{index}.png')
      