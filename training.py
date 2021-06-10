from data_loader import dataload
from model import le_model
def train():
    my_model = le_model()
    x_train, y_train, x_test, y_test = dataload()
    history = my_model.fit(x_train, y_train, batch_size = 64,epochs = 50, validation_data = (x_test, y_test))
    return history
train()