import tensorflow as tf
from PIL import Image
import numpy as np
from io import BytesIO

class Dataset():
    def __init__(self,filename,config,data_type="train"):
        if not (data_type == "train" or data_type == "val"):
            raise ValueError ("the type of data_type must be \"train\" or \"val\"")
        self.record_filename = filename
        self.error_message = 0
        self.image, self.label = self.__read_and_decode(self.record_filename, config.Num_categrade)
        # 取出一个batch的image,label
        if(data_type == "train"):
            self.image_batch, self.label_batch = tf.train.shuffle_batch([self.image, self.label],
                                                              batch_size=config.Batch_size, capacity=200,
                                                              min_after_dequeue=100,
                                                              num_threads=1)
        else:
            self.image_batch, self.label_batch = tf.train.shuffle_batch([self.image, self.label],
                                                              batch_size=config.Val_batch_size, capacity=200,
                                                              min_after_dequeue=100,
                                                              num_threads=1)

    def __byte_to_img(self,byte):
        str_buf = BytesIO(byte)
        img = np.array(Image.open(str_buf))
        str_buf.close()
        return img

    def __read_and_decode(self,filename, num_class):
        # 根据文件名生成一个队列
        filename_queue = tf.train.string_input_producer([filename])
        reader = tf.TFRecordReader()
        # 返回文件名和文件
        _, serialized_example = reader.read(filename_queue)
        feature = dict()
        feature['image'] = tf.FixedLenFeature([], tf.string)
        for i in range(num_class):
            feature['label_{}'.format(i)] = tf.FixedLenFeature([], tf.int64)
        features = tf.parse_single_example(serialized_example,
                                           features=feature)
        # 获取图片数据
        # image = tf.decode_raw(features['image'], tf.uint8)
        image = features['image']
        # # 没有经过预处理的灰度图
        # image_raw = tf.reshape(image, [224, 224])
        # # tf.train.shuffle_batch必须确定shape
        # image = tf.reshape(image, [224, 224])
        # # 图片预处理
        # image = tf.cast(image, tf.float32) / 255.0
        # image = tf.subtract(image, 0.5)
        # image = tf.multiply(image, 2.0)
        # 获取label
        label = list()
        for i in range(num_class):
            label.append(tf.cast(features['label_{}'.format(i)], tf.int32))

        return image, label


    def mold_image(self,images):
        """Takes RGB images with 0-255 values and subtraces
        the mean pixel and converts it to float. Expects image
        colors in RGB order.
        """
        return images.astype(np.float32) - np.array([170.8, 162.6, 159.5])

    def data_generater(self):
            try:
                sess = tf.Session()
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord,sess=sess)
                while(True):
                    example, label = sess.run([self.image_batch,self.label_batch])#在会话中取出image和label
                    img = list()
                    for index,ex in enumerate(example):
                        img.append(self.__byte_to_img(ex))#jpg解码
                    img = np.array(img)
                    label = np.transpose(label,(1,0))
                    label = [label[x] for x in range(label.shape[0])]
                    # coord.request_stop()
                    # coord.join(threads)
                    yield [img,label]
            except (GeneratorExit, KeyboardInterrupt):
                raise
            except:
                # Log it and skip the image
                print("Error!")
                self.error_message += 1
                if self.error_message > 5:
                    coord.request_stop()
                    coord.join(threads)
                    raise
