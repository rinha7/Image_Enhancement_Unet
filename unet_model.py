import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, Concatenate, Conv2DTranspose, Lambda

import numpy as np
import matplotlib.pyplot as plt
import os

def psnr(label, prediction):
    label = tf.clip_by_value(0.5 * (label + 1.0), 0.0, 1.0)
    prediction = tf.clip_by_value(0.5 * (prediction + 1.0), 0.0, 1.0)
    psnr = tf.reduce_mean(tf.image.psnr(label, prediction, max_val=1.0))
    return psnr


def ssim(label, prediction):
    label = tf.clip_by_value(0.5 * (label + 1.0), 0.0, 1.0)
    prediction = tf.clip_by_value(0.5 * (prediction + 1.0), 0.0, 1.0)
    ssim = tf.reduce_mean(tf.image.ssim(label, prediction, max_val=1.0))
    return ssim

class DataGenerator(keras.utils.Sequence):

    def __init__(self, path='data/Adobe5k/', batch_size=1, isTrain=True,
                 img_dim=(512, 512, 3), shuffle=True):
        self.batch_size = batch_size
        self.isTrain = isTrain
        self.img_dim = img_dim
        self.shuffle = shuffle
        if self.isTrain:
            self.indexes = list(range(1, 4501))

        else:
            self.indexes = list(range(4501, 5001))
        self.x_path = path + 'input/'
        self.y_path = path + 'user-c/'
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def read_image(self, path, img_idx):
        img = tf.io.decode_image(tf.io.read_file(path + img_idx + '.jpg'))
        img = tf.image.convert_image_dtype(img, dtype='float32')
        img = 2.0 * img - 1.0
        img = tf.image.resize(img, [self.img_dim[0], self.img_dim[1]], method='bilinear')
        return img

    def __getitem__(self, index):
        index = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        x = np.empty((self.batch_size, *self.img_dim))
        y = np.empty((self.batch_size, *self.img_dim))

        for batch_idx, img_idx in enumerate(index):
            x[batch_idx] = self.read_image(self.x_path, str(img_idx).zfill(4))
            y[batch_idx] = self.read_image(self.y_path, str(img_idx).zfill(4))

        return x, y

    def __len__(self):
        return int(np.floor(len(self.indexes) / self.batch_size))


def Unet_Residual():
    input_shape = (512, 512, 3,)
    input_layer = Input(shape=input_shape)

    # Encoding
    # 512 * 512 * 3
    conv1 = Conv2D(16, kernel_size=(5, 5), strides=1, padding='same', activation='selu')(input_layer)
    batch1 = BatchNormalization(momentum=0.9)(conv1)
    # 512 * 512 * 16
    conv2 = Conv2D(32, kernel_size=(5, 5), strides=2, padding='same', activation='selu')(batch1)
    batch2 = BatchNormalization(momentum=0.9)(conv2)
    # 256 * 256 * 32
    conv3 = Conv2D(64, kernel_size=(5, 5), strides=2, padding='same', activation='selu')(batch2)
    batch3 = BatchNormalization(momentum=0.9)(conv3)
    # 128 * 128 * 64
    conv4 = Conv2D(128, kernel_size=(5, 5), strides=2, padding='same', activation='selu')(batch3)
    batch4 = BatchNormalization(momentum=0.9)(conv4)
    # 64 * 64 * 128
    conv5 = Conv2D(256, kernel_size=(5, 5), strides=2, padding='same', activation='selu')(batch4)
    batch5 = BatchNormalization(momentum=0.9)(conv5)
    # 32 * 32 * 256
    conv6 = Conv2D(512, kernel_size=(5, 5), strides=2, padding='same', activation='selu')(batch5)
    batch6 = BatchNormalization(momentum=0.9)(conv6)
    # 16 * 16 * 512

    # Decoding
    conv7 = Conv2DTranspose(256, kernel_size=(5, 5), strides=2, padding='same', activation='selu')(batch6)
    batch7 = BatchNormalization(momentum=0.9)(conv7)
    concat7 = Concatenate()([batch5, batch7])
    # 32 * 32 * 512
    conv8 = Conv2DTranspose(128, kernel_size=(5, 5), strides=2, padding='same', activation='selu')(concat7)
    batch8 = BatchNormalization(momentum=0.9)(conv8)
    concat8 = Concatenate()([batch4, batch8])
    # 64 * 64 * 256
    conv9 = Conv2DTranspose(64, kernel_size=(5, 5), strides=2, padding='same', activation='selu')(concat8)
    batch9 = BatchNormalization(momentum=0.9)(conv9)
    concat9 = Concatenate()([batch3, batch9])
    # 128 * 128 * 128
    conv10 = Conv2DTranspose(32, kernel_size=(5, 5), strides=2, padding='same', activation='selu')(concat9)
    batch10 = BatchNormalization(momentum=0.9)(conv10)
    concat10 = Concatenate()([batch2, batch10])
    # 256 * 256 * 64
    conv11 = Conv2DTranspose(16, kernel_size=(5, 5), strides=2, padding='same', activation='selu')(concat10)
    batch11 = BatchNormalization(momentum=0.9)(conv11)
    concat11 = Concatenate()([batch1, batch11])
    # 512 * 512 * 32
    conv12 = Conv2D(16, kernel_size=(5, 5), strides=1, padding='same', activation='selu')(concat11)
    batch12 = BatchNormalization(momentum=0.9)(conv12)
    # 512 * 512 * 16


    # Residual Block
    conv13 = Conv2D(3, (5,5),1,'same',activation='selu')(batch12)
    residual = Lambda(lambda  tensors : tf.add(tensors[0],tensors[1]), output_shape=(512,512,3))([input_layer, conv13])
    outputs = Conv2D(3, kernel_size=(5, 5), strides=1, padding='same', activation='tanh')(residual)



    model = Model(inputs=input_layer, outputs=outputs)

    return model
