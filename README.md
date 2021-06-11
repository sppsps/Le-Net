# Le-Net
# Contributors
- [Pranjal Sharma](https://github.com/sppsps)
# Summary

## Introduction
This model is implementing the architecture and training techniques used in Le-net paper. We used MNIST dataset by keras.<br />The preprocessing :<br />
A) Dividing all the pixels of images by 225<br />
B) Converting y_train(labels) into float type.<br />
C) Using to_categorical() on training labels to use categorical loss as the loss function while training.<br />
# Architecture
![alt text](https://miro.medium.com/max/3436/1*ddbd4IrPvGrBNdcZtboLeA.png)
## Loss function
Categorical crossentropy: It is a loss function that is used in multi-class classification tasks. These are tasks where an example can only belong to one out of many possible categories, and the model must decide which one.
Formally, it is designed to quantify the difference between two probability distributions.
## Training
For training this model, adam optimizer is used with learning rate 0.01. Categorical loss Function is used as the loss function, and I ran it for 50 epochs, resulting in training accuracy of 99.29%, validation accuracy of 98.34%. An early stop can also be used with regard to validation loss.
Finally, We have plotted the accuracy vs epochs with loss function which is shown using matplotlib.pyplot in training.py script.

