# -*- coding: utf-8 -*-
from model.Densenet_rewrite import Densenet
from model.config import config
import os
from skimage import io,transform
import numpy as np
from io import BytesIO
from utils.data import Dataset

base_path = os.getcwd()
model_path = os.path.join(base_path,"model_save")
model_filename = "Desenet_imagenet_0200.h5"

class My_config(config):
    Name = "ImageNet"
    Num_categrade = 13
    Batch_size = 1

real_config = My_config()

model = Densenet(model_dir = model_path,config=real_config,model_filename=model_filename,mode='inference')

#############################################
data_path='./data/train+val.txt'
fp=open(data_path,mode='r')
img_infos=fp.readlines()

for i in range(10):
    info = img_infos[i]
    info = info.replace("\n","")
    img_id = info.split(",")[0]
    img_url = info.split(",")[1]
    img_labels = info.split(",")[2:]
    img_labels = list(map(int, img_labels))

    print("label",[img_labels])

    img = io.imread(img_url)
    img = transform.resize(img,(224,224,3))
    img = img * 255  # 将图片的取值范围改成（0~255）
    img = img.astype(np.uint8)

    predect = model.detect([img])
    print("prede",predect)


