# -*- coding: utf-8 -*-
"""DenseNet models for Keras.

# Reference paper

- [Densely Connected Convolutional Networks]
  (https://arxiv.org/abs/1608.06993) (CVPR 2017 Best Paper Award)

# Reference implementation

- [Torch DenseNets]
  (https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua)
- [TensorNets]
  (https://github.com/taehoonlee/tensornets/blob/master/tensornets/densenets.py)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Activation
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.layers import Lambda
from keras.layers import GlobalMaxPooling2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.utils.data_utils import get_file
import re
import datetime
import numpy as np
import tensorflow as tf
from keras.engine import get_source_inputs

DENSENET121_WEIGHT_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.8/densenet121_weights_tf_dim_ordering_tf_kernels.h5'
DENSENET121_WEIGHT_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.8/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5'
DENSENET169_WEIGHT_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.8/densenet169_weights_tf_dim_ordering_tf_kernels.h5'
DENSENET169_WEIGHT_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.8/densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5'
DENSENET201_WEIGHT_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.8/densenet201_weights_tf_dim_ordering_tf_kernels.h5'
DENSENET201_WEIGHT_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.8/densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5'


class Densenet():
    """
    the init of the mode
    """
    def __init__(self,
                 model_dir = None,
                 model_filename = None,
                 config = None,
                 mode = 'training',
                 log_dir = None,
                ):
        if not (config.Base_model_name=="Densenet121" or config.Base_model_name=="Densenet169" or config.Base_model_name=="Densenet201"):
            raise ValueError('The input arg model_name must be chosen in \'Densenet121\',\'Densenet169\' and \'Densenet201\'!')

        if not(len(config.Input_shape) == 3):
            raise ValueError('The input shape must be a 3D tunple,like(224,224,3)!')

        if not(config.Pooling_type=='avg' or config.Pooling_type=='max' or config.Pooling_type==None):
            raise ValueError('The input pooling arg must be chosen in \'avg\',\'max\' and \'None\'!')

        if not (os.path.exists(model_dir) or model_dir==None):
            raise ValueError('The model_dir is not exist,Check the dic and try again!')

        if not (mode=="training" or mode=="inference"):
            raise ValueError('The input arg mode must be chosen in \'training\',\'inference\'!')

        if not (len(config.Sampledata_distributes) == config.Num_categrade):
            raise ValueError('The length of data distributes must equal to the num of categrade!')

        if(config.Base_model_name == 'Densenet121'):
            self.blocks = [6, 12, 24, 16]
        elif(config.Base_model_name == 'Densenet169'):
            self.blocks = [6, 12, 32, 32]
        else:
            self.blocks = [6, 12, 48, 32]

        self.name = config.Name
        self.model_name = config.Base_model_name
        self.input_shape = config.Input_shape
        self.pooling = config.Pooling_type
        self.total_category = config.Num_categrade
        self.model_dir = model_dir
        self.model_filename = model_filename
        self.learningrate = config.Learning_rate
        self.use_learningrate_reduce = config.Use_learning_rate_reduce
        self.learningrate_decay = config.Learning_rate_decay
        self.mode = mode
        self.batch_size = config.Batch_size
        self.step_per_epoch = config.Step_per_epoch
        self.Validation_steps = config.Validation_steps
        if(log_dir == None):
            if(os.path.exists('./log')):
                self.log_dir = './log'
            else:
                os.makedirs('./log')
                self.log_dir = './log'
        else:
            self.log_dir = log_dir
        self.weights = config.Sampledata_distributes/np.sum(config.Sampledata_distributes)
        self.set_log_dir()
        self.build()

    def __dense_block(self, x, blocks, name):
        """A dense block.

        # Arguments
            x: input tensor.
            blocks: integer, the number of building blocks.
            name: string, block label.

        # Returns
            output tensor for the block.
        """
        for i in range(blocks):
            x = self.__conv_block(x, 32, name=name + '_block' + str(i + 1))
        return x


    def __transition_block(self, x, reduction, name):
        """A transition block.

        # Arguments
            x: input tensor.
            reduction: float, compression rate at transition layers.
            name: string, block label.

        # Returns
            output tensor for the block.
        """
        bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
        x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                               name=name + '_bn')(x)
        x = Activation('relu', name=name + '_relu')(x)
        x = Conv2D(int(K.int_shape(x)[bn_axis] * reduction), 1, use_bias=False,
                   name=name + '_conv')(x)
        x = AveragePooling2D(2, strides=2, name=name + '_pool')(x)
        return x


    def __conv_block(self,x , growth_rate, name):
        """A building block for a dense block.

        # Arguments
            x: input tensor.
            growth_rate: float, growth rate at dense layers.
            name: string, block label.

        # Returns
            output tensor for the block.
        """
        bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
        x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                name=name + '_0_bn')(x)
        x1 = Activation('relu', name=name + '_0_relu')(x1)
        x1 = Conv2D(4 * growth_rate, 1, use_bias=False,
                    name=name + '_1_conv')(x1)
        x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                name=name + '_1_bn')(x1)
        x1 = Activation('relu', name=name + '_1_relu')(x1)
        x1 = Conv2D(growth_rate, 3, padding='same', use_bias=False,
                    name=name + '_2_conv')(x1)
        x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
        return x

    def Fc_block(self,x,name):
        x = Conv2D(64,3, padding='same',use_bias=False,name=name+'/cov')(x)
        x = GlobalAveragePooling2D(name=name + 'avgpool')(x)
        x = Dense(1,activation='sigmoid',name=name+'/sigmoid')(x)
        return x
    def build(self):
        img_input = Input(shape=self.input_shape)

        bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

        '''
        Define the layer of Densenet.
        '''
        x = ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
        x = Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
        x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                               name='conv1/bn')(x)
        x = Activation('relu', name='conv1/relu')(x)
        x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
        x = MaxPooling2D(3, strides=2, name='pool1')(x)

        x = self.__dense_block(x, self.blocks[0], name='conv2')
        x = self.__transition_block(x, 0.5, name='pool2')
        x = self.__dense_block(x, self.blocks[1], name='conv3')
        x = self.__transition_block(x, 0.5, name='pool3')
        x = self.__dense_block(x, self.blocks[2], name='conv4')
        x = self.__transition_block(x, 0.5, name='pool4')
        x = self.__dense_block(x, self.blocks[3], name='conv5')

        x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                               name='bn')(x)

        # x = GlobalAveragePooling2D(name='avg_pool')(x)

        '''
        Define the outputs of muti-categrades,the length of output equal the number of the grand-categrade. 
        '''
        outputs=[]

        for categrade_num in range(self.total_category):
            y = self.Fc_block(x,name = 'fc_layer/Fc_{}'.format(categrade_num))
            outputs.append(y)
            # if(categrade_num == 0):
            #     z=y
            # else:
            #     outputs = Lambda(lambda x:tf.concat([x,z],axis=1),name='output')(y)
            #     z = tf.concat([z,y],axis=1)

        # outputs.append(Lambda(lambda x: tf.concat(x, axis=1), name='output')(outputs))
        '''
        Define the input size as the arg input shape.
        '''
        inputs = img_input

        # Create model.
        if self.blocks == [6, 12, 24, 16]:
            self.model = Model(inputs, outputs, name='Densenet121')
        elif self.blocks == [6, 12, 32, 32]:
            self.model = Model(inputs, outputs, name='Densenet169')
        elif self.blocks == [6, 12, 48, 32]:
            self.model = Model(inputs, outputs, name='Densenet201')
        else:
            self.model = Model(inputs, outputs, name='Densenet')

        '''
        Loading weight from an existing file if posible,
        If pre-training model not exist, download a pre-training model on imagenet in the URL.
        '''
        if (self.model_dir != None and self.model_filename != None):
            self.load_weight(os.path.join(self.model_dir,self.model_filename),by_name=True)
        elif (self.model_dir != None and self.model_filename == None):
            self.model_filename = self.download_weight()
            self.load_weight(self.model_filename,by_name=True)
        return self.model

    def load_weight(self, filepath, by_name=False, exclude=None):
        """Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to excluce
        """
        import h5py
        from keras.engine import topology

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        print("Loading weights from {}".format(filepath))
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            topology.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            topology.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

        # Update the log directory
        # self.set_log_dir(filepath)

    def download_weight(self):
        print("Downloading model {}!".format(self.model_name))
        if self.model_name == "Densenet121":
            weights_path = get_file(
                os.path.join(self.model_dir, 'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5'),
                DENSENET121_WEIGHT_PATH_NO_TOP,
                cache_subdir='./model_save',
                file_hash='4912a53fbd2a69346e7f2c0b5ec8c6d3')
        elif self.model_name == "Densenet169":
            weights_path = get_file(
                os.path.join(self.model_dir, 'densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5'),
                DENSENET169_WEIGHT_PATH_NO_TOP,
                cache_subdir='./model_save',
                file_hash='50662582284e4cf834ce40ab4dfa58c6')
        elif self.model_name == "Densenet201":
            weights_path = get_file(
                os.path.join(self.model_dir, 'densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5'),
                DENSENET201_WEIGHT_PATH_NO_TOP,
                cache_subdir='./model_save',
                file_hash='1c2de60ee40562448dbac34a0737e798')
        return weights_path

    def complie(self):
        if(self.use_learningrate_reduce):
            optimizer = keras.optimizers.SGD(lr=self.learningrate, momentum=0.9,
                                             clipnorm=5.0, decay= self.learningrate_decay, nesterov=True)
        else:
            optimizer = keras.optimizers.SGD(lr=self.learningrate, momentum=0.9,
                                             clipnorm=5.0)

        #Add muti-losses fuctions in the training task
        losses = dict()
        losses_weight = dict()
        for categrade_num in range(self.total_category):
            losses['fc_layer/Fc_{}/sigmoid'.format(categrade_num)] = 'binary_crossentropy'
            losses_weight['fc_layer/Fc_{}/sigmoid'.format(categrade_num)] = self.weights[categrade_num]
        # losses['output'] = 'binary_crossentropy'
        # losses = 'binary_crossentropy'
        self.model.compile(optimizer=optimizer,loss=losses,loss_weights=losses_weight,metrics=['accuracy'])

    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            print("Selecting layers to train")

        keras_model = keras_model or self.model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                self.set_trainable(
                    layer_regex, keras_model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainble layer names
            if trainable and verbose > 0:
                print("{}{:20}   ({})".format(" " * indent, layer.name,
                                            layer.__class__.__name__))

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:

            regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/mask\_rcnn\_\w+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                self.epoch = int(m.group(6)) + 1

        # Directory for training logs
        self.log_dir = os.path.join(self.log_dir, "{}{:%Y%m%dT%H%M}".format(
            self.name.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "Desenet_{}_*epoch*.h5".format(
            self.name.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")


    def train(self, train_dataset=None, val_dataset=None, learning_rate=None, epochs=None, layers=None):

        assert self.mode == "training", "Create model in training mode."

        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads": r"(fc_layer/.*)",
            # From a specific Resnet stage and up
            # "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data keypoint generators

        train_generator = train_dataset.data_generater()
        val_generator = val_dataset.data_generater()

        # Callbacks
        if not (self.use_learningrate_reduce):
            callbacks = [
                keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                            histogram_freq=0, write_graph=True, write_images=False),
                keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                                verbose=0, save_weights_only=True),
            ]
        else:
            callbacks = [
                keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                            histogram_freq=0, write_graph=True, write_images=False),
                keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                                verbose=0, save_weights_only=True),
                keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.01), cooldown=0, patience=5, min_lr=0.1e-6)
            ]

        # Train
        print("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        print("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.complie()

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if os.name is 'nt':
            workers = 0
        else:
            workers = max(self.batch_size // 2, 2)

        self.model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.step_per_epoch,
            callbacks=callbacks,
            validation_data=next(val_generator),
            validation_steps=self.Validation_steps,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=True,
        )
        self.epoch = max(self.epoch, epochs)

    def detect(self,images):
        assert self.mode == "inference", "Create model in inference mode."
        images = np.array(images)
        detections = self.model.predict(images, verbose=0)#output shape [13,(?,1)]
        detections = np.stack(detections,axis=1)#to shape (?,13,1)
        detections = np.reshape(detections,(-1,self.total_category))#to shape(?,13)
        detections[:, :] = np.where(detections[:,:]>0.5,1,0)
        return detections