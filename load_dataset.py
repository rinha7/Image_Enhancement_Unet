import tensorflow as tf
import numpy as np
from tensorflow import keras

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
        self.y_path = path + 'label/'
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
