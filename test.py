# -*- coding: utf-8 -*-
from model.Densenet_rewrite import Densenet
from model.config import config
import os
from utils.data import Dataset

base_path = os.getcwd()
model_path = os.path.join(base_path,"model_save")
train_datapath = os.path.join(base_path,"data","train.tfrecords")
val_datapath = os.path.join(base_path,"data","test.tfrecords")

class My_config(config):
    Name = "ImageNet"
    Num_categrade = 13
    Batch_size = 10
    Use_learning_rate_reduce = False

real_config = My_config()

train_dataset = Dataset(train_datapath,config=real_config)
val_dataset = Dataset(val_datapath,config=real_config)
# a = next(train_dataset.data_generater())

model = Densenet(model_dir = model_path,config=real_config,mode='training')

model.train(train_dataset=train_dataset,
            val_dataset=val_dataset,
            learning_rate=real_config.Learning_rate,
            epochs=200,
            layers='heads')

print(model)
