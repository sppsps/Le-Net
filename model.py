import tensorflow as tf
import keras
import tensorflow.keras.layers as tfl
def le_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.Input((28,28)))
    model.add(tfl.Conv1D(filters = 6, kernel_size = 5, padding = 'same',strides = 1))
    model.add(tfl.AveragePooling1D(pool_size = 2, strides = 2))
    model.add(tfl.Conv1D(filters = 16, kernel_size = 5, padding = 'same', strides = 1))
    model.add(tfl.AveragePooling1D(pool_size = 2, strides = 2))
    model.add(tfl.Conv1D(filters = 120, kernel_size = 5, padding = 'same', strides = 1))
    model.add(tfl.AveragePooling1D(pool_size = 2, strides = 2))
    model.add(tfl.Flatten())
    model.add(tfl.Dense(units = 84, activation = 'tanh'))
    model.add(tfl.Dense(units = 120, activation = 'tanh'))
    model.add(tfl.Dense(units = 10, activation = 'softmax'))   
    opt = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name="Adam"
    )
    model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return  model