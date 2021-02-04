import tensorflow as tf
from tensorflow import keras


class Unet_Model(keras.Model):
    def __init__(self):
        super(Unet_Model,self).__init__()
        # Encoder Part
        self.conv1 = keras.layers.Conv2D(16, (5, 5), 1, 'same', kernel_initializer="he_normal")
        self.act1 = keras.layers.Activation('selu')
        self.batch1 = keras.layers.BatchNormalization(momentum=0.9)

        self.conv2 = keras.layers.Conv2D(32, (5, 5), 2, 'same', kernel_initializer="he_normal")
        self.act2 = keras.layers.Activation('selu')
        self.batch2 = keras.layers.BatchNormalization(momentum=0.9)

        self.conv3 = keras.layers.Conv2D(64, (5, 5), 2, 'same', kernel_initializer="he_normal")
        self.act3 = keras.layers.Activation('selu')
        self.batch3 = keras.layers.BatchNormalization(momentum=0.9)

        self.conv4 = keras.layers.Conv2D(128, (5, 5), 2, 'same', kernel_initializer="he_normal")
        self.act4 = keras.layers.Activation('selu')
        self.batch4 = keras.layers.BatchNormalization(momentum=0.9)

        self.conv5 = keras.layers.Conv2D(256, (5, 5), 2, 'same', kernel_initializer="he_normal")
        self.act5 = keras.layers.Activation('selu')
        self.batch5 = keras.layers.BatchNormalization(momentum=0.9)

        self.conv6 = keras.layers.Conv2D(512, (5, 5), 2, 'same', kernel_initializer="he_normal")
        self.act6 = keras.layers.Activation('selu')
        self.batch6 = keras.layers.BatchNormalization(momentum=0.9)

        # Decoder Part
        self.deconv1 = keras.layers.Conv2DTranspose(256, (5, 5), 2, 'same', kernel_initializer="he_normal")
        self.act7 = keras.layers.Activation('selu')
        self.batch7 = keras.layers.BatchNormalization(momentum=0.9)

        self.deconv2 = keras.layers.Conv2DTranspose(128, (5, 5), 2, 'same', kernel_initializer="he_normal")
        self.act8 = keras.layers.Activation('selu')
        self.batch8 = keras.layers.BatchNormalization(momentum=0.9)

        self.deconv3 = keras.layers.Conv2DTranspose(64, (5, 5), 2, 'same', kernel_initializer="he_normal")
        self.act9 = keras.layers.Activation('selu')
        self.batch9 = keras.layers.BatchNormalization(momentum=0.9)

        self.deconv4 = keras.layers.Conv2DTranspose(32, (5, 5), 2, 'same', kernel_initializer="he_normal")
        self.act10 = keras.layers.Activation('selu')
        self.batch10 = keras.layers.BatchNormalization(momentum=0.9)

        self.deconv5 = keras.layers.Conv2DTranspose(16, (5, 5), 2, 'same', kernel_initializer="he_normal")
        self.act11 = keras.layers.Activation('selu')
        self.batch11 = keras.layers.BatchNormalization(momentum=0.9)

        self.deconv6 = keras.layers.Conv2D(16, (5, 5), 2, 'same', kernel_initializer="he_normal")
        self.act12 = keras.layers.Activation('selu')
        self.batch12 = keras.layers.BatchNormalization(momentum=0.9)

        self.f_conv = keras.layers.Conv2D(3, kernel_size=(5, 5), strides=1, padding='same', activation='tanh')

    def call(self, inputs, training=False, mask=None):
        x = self.conv1(inputs)
        x = self.act1(x)
        batch1 = self.batch1(x)

        x = self.conv2(batch1)
        x = self.act2(x)
        batch2 = self.batch2(x)

        x = self.conv3(batch2)
        x = self.act3(x)
        batch3 = self.batch3(x)

        x = self.conv4(batch3)
        x = self.act4(x)
        batch4 = self.batch4(x)

        x = self.conv5(batch4)
        x = self.act5(x)
        batch5 = self.batch5(x)

        x = self.conv6(batch5)
        x = self.act6(x)
        batch6 = self.batch6(x)

        x = self.deconv1(batch6)
        x = self.act7(x)
        batch7 = self.batch7(x)
        cat1 = tf.concat([batch5, batch7],axis=-1)

        x = self.deconv2(cat1)
        x = self.act8(x)
        batch8 = self.batch8(x)
        cat2 = tf.concat([batch4, batch8],axis=-1)

        x = self.deconv3(cat2)
        x = self.act9(x)
        batch9 = self.batch9(x)
        cat3 = tf.concat([batch3, batch9],axis=-1)

        x = self.deconv4(cat3)
        x = self.act10(x)
        batch10 = self.batch10(x)
        cat4 = tf.concat([batch2, batch10],axis=-1)

        x = self.deconv5(cat4)
        x = self.act11(x)
        batch11 = self.batch11(x)
        cat5 = tf.concat([batch1, batch11],axis=-1)

        output = self.f_conv(cat5)

        return output
