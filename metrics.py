import tensorflow as tf


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