import tensorflow as tf
import os
import random
import math
import sys
from PIL import Image
import numpy as np
from skimage import io,transform
from io import BytesIO
_NUM_TEST = 4000
TFRECORD_DIR = "./data"
split_name = "train"

data_path='./data/train+val.txt'

#img_info=fp.readline().strip('\n').split(',')


def array_pic_to_stream(pic):
    '''
    将数组图片转化为byte
    :param pic:
    :return:
    '''
    stream = BytesIO()
    pic = Image.fromarray(pic)
    pic.save(stream, format="JPEG")
    jepg = stream.getvalue()
    stream.close()
    return jepg

def int64_feature(values):
    if not isinstance(values,(tuple,list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def image_to_tfexample(image,label):
    file = dict()
    file['image'] = bytes_feature(image)
    for index,i in enumerate(label):
        file['label_{}'.format(index)] = int64_feature(int(i))
    # label = list(map(int, label))
    # file['label'] = int64_feature(label)
    return tf.train.Example(features=tf.train.Features(feature=file))


def cov_dataset(split_name,filenames):
    with tf.Session() as sess:
        # 定义tfrecord文件路径+名字
        output_filename = os.path.join(TFRECORD_DIR, split_name + '.tfrecords')
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            for i,info in enumerate(filenames):
                try:
                    sys.stdout.write('\r>>Converting image %d/%d' % (i + 1, len(filenames)))
                    sys.stdout.flush()
                    info = info.replace("\n","")
                    img_id = info.split(",")[0]
                    img_url = info.split(",")[1]
                    img_labels = info.split(",")[2:]

                    #读取图片
                    img = io.imread(img_url)
                    img = transform.resize(img,(224,224,3))
                    img = img * 255  # 将图片的取值范围改成（0~255）
                    img = img.astype(np.uint8)
                    # 转化为bytes
                    # img = img.tobytes()
                    img = array_pic_to_stream(img)

                    example = image_to_tfexample(img,img_labels)
                    tfrecord_writer.write(example.SerializeToString())
                except BaseException as e:
                    print('Cannot read file:',img_id)
                    print('Error:', e)
                    print('Skip it!')
        sys.stdout.write('\n')
        sys.stdout.flush()

fp=open(data_path,mode='r')
img_infos=fp.readlines()

# 数据集进行切分打乱
random.seed(42)
random.shuffle(img_infos)
training_file = img_infos[_NUM_TEST:]
testing_file = img_infos[:_NUM_TEST]
print("生成train文件:",len(training_file))
cov_dataset('train', training_file)
print("生成test文件",len(testing_file))
cov_dataset('test', testing_file)
