import tensorflow as tf

def Unet_Residual():
    input_shape = (512, 512, 3,)
    input_layer = tf.keras.layers.Input(shape=input_shape)

    # Encoding
    # 512 * 512 * 3
    conv1 = tf.keras.layers.Conv2D(16, kernel_size=(5, 5), strides=1, padding='same', activation='selu')(input_layer)
    batch1 = tf.keras.layers.BatchNormalization(momentum=0.9)(conv1)
    # 512 * 512 * 16
    conv2 = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), strides=2, padding='same', activation='selu')(batch1)
    batch2 = tf.keras.layers.BatchNormalization(momentum=0.9)(conv2)
    # 256 * 256 * 32
    conv3 = tf.keras.layers.Conv2D(64, kernel_size=(5, 5), strides=2, padding='same', activation='selu')(batch2)
    batch3 = tf.keras.layers.BatchNormalization(momentum=0.9)(conv3)
    # 128 * 128 * 64
    conv4 = tf.keras.layers.Conv2D(128, kernel_size=(5, 5), strides=2, padding='same', activation='selu')(batch3)
    batch4 = tf.keras.layers.BatchNormalization(momentum=0.9)(conv4)
    # 64 * 64 * 128
    conv5 = tf.keras.layers.Conv2D(256, kernel_size=(5, 5), strides=2, padding='same', activation='selu')(batch4)
    batch5 = tf.keras.layers.BatchNormalization(momentum=0.9)(conv5)
    # 32 * 32 * 256
    conv6 = tf.keras.layers.Conv2D(512, kernel_size=(5, 5), strides=2, padding='same', activation='selu')(batch5)
    batch6 = tf.keras.layers.BatchNormalization(momentum=0.9)(conv6)
    # 16 * 16 * 512

    # Decoding
    conv7 = tf.keras.layers.Conv2DTranspose(256, kernel_size=(5, 5), strides=2, padding='same', activation='selu')(batch6)
    batch7 = tf.keras.layers.BatchNormalization(momentum=0.9)(conv7)
    concat7 = tf.keras.layers.Concatenate()([batch5, batch7])
    # 32 * 32 * 512
    conv8 = tf.keras.layers.Conv2DTranspose(128, kernel_size=(5, 5), strides=2, padding='same', activation='selu')(concat7)
    batch8 = tf.keras.layers.BatchNormalization(momentum=0.9)(conv8)
    concat8 = tf.keras.layers.Concatenate()([batch4, batch8])
    # 64 * 64 * 256
    conv9 = tf.keras.layers.Conv2DTranspose(64, kernel_size=(5, 5), strides=2, padding='same', activation='selu')(concat8)
    batch9 = tf.keras.layers.BatchNormalization(momentum=0.9)(conv9)
    concat9 = tf.keras.layers.Concatenate()([batch3, batch9])
    # 128 * 128 * 128
    conv10 = tf.keras.layers.Conv2DTranspose(32, kernel_size=(5, 5), strides=2, padding='same', activation='selu')(concat9)
    batch10 = tf.keras.layers.BatchNormalization(momentum=0.9)(conv10)
    concat10 = tf.keras.layers.Concatenate()([batch2, batch10])
    # 256 * 256 * 64
    conv11 = tf.keras.layers.Conv2DTranspose(16, kernel_size=(5, 5), strides=2, padding='same', activation='selu')(concat10)
    batch11 = tf.keras.layers.BatchNormalization(momentum=0.9)(conv11)
    concat11 = tf.keras.layers.Concatenate()([batch1, batch11])
    # 512 * 512 * 32
    conv12 = tf.keras.layers.Conv2D(16, kernel_size=(5, 5), strides=1, padding='same', activation='selu')(concat11)
    batch12 = tf.keras.layers.BatchNormalization(momentum=0.9)(conv12)
    # 512 * 512 * 16


    # Residual Block
    conv13 = tf.keras.layers.Conv2D(3, (5,5),1,'same',activation='selu')(batch12)
    residual = tf.keras.layers.Lambda(lambda  tensors : tf.add(tensors[0],tensors[1]), output_shape=(512,512,3))([input_layer, conv13])
    outputs = tf.keras.layers.Conv2D(3, kernel_size=(5, 5), strides=1, padding='same', activation='tanh')(residual)



    model = tf.keras.Model(inputs=input_layer, outputs=outputs)

    return model
