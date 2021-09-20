from glob import glob
from tqdm import tqdm
import cv2
import os
num_word = 5
dir = '/content/drive/MyDrive/pkm/stress detection/dataset/wawancara/spectogram_pcen_15w_cropped'
image_dir = '/content/drive/MyDrive/pkm/stress detection/dataset/wawancara/spectogram_pcen_15w'
image_list = (glob(f'{image_dir}/*/*'))


if not os.path.exists(dir):
  os.mkdir(dir)
for filename in tqdm(image_list[:]):
  try:
    if not os.path.exists(f'{dir}/{filename.split("/")[-2]}'):
      os.mkdir(f'{dir}/{filename.split("/")[-2]}')
    img = cv2.imread(filename)
    # crop_img = img[26:190, 180:1073]
    if(num_word==5):
      crop_img = img[26:190, 225:1342]
    elif (num_word==5):
      crop_img = img[26:190, 450:2683]
    elif (num_word==15):
      crop_img = img[26:190, 678:4024]
    else: 
      print('Jumlah kata tidak sesuai')
      break
    cv2.imwrite(f'{dir}/'+filename.split("/")[-2]+"/"+filename.split("/")[-1], crop_img)
  except:
    print(filename)