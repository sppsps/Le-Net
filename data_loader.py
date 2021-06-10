import tensorflow as tf
import keras

def dataload():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)
    x_train = x_train.astype(float)
    y_train = y_train.astype(float)
    y_test = y_test.astype(float)
    x_train = x_train/255.0
    x_test = x_test/255.0
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    return x_train, y_train, x_test, y_test
