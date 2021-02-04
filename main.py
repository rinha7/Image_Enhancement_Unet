import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from metrics import *
from load_dataset import *
from unet_model import *

def main():
    train_datagen = DataGenerator(batch_size=1,isTrain=True)
    test_datagen = DataGenerator(isTrain=False)

    model = Unet_Model()

    with tf.device("CPU:0"):
        model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-05),loss="mse",metrics=[psnr,ssim])

        model.fit(train_datagen,validation_data=test_datagen)


if __name__ == '__main__':
    main()