import tensorflow.keras as keras
from tensorflow.keras.datasets import cifar10
import models.cifar as models

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def main():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    print('train shape:', x_train.shape, y_train.shape)
    print('test shape:', x_test.shape, y_test.shape)


    (x_train, y_train), (x_test, y_test) = preprocess(x_train, y_train, x_test, y_test)

    model = models.model1()
    history = model.fit(x_train, y_train,
                        batch_size=1024,
                        epochs=10,
                        verbose=1,
                        validation_data=(x_test, y_test))



def preprocess(x_train, y_train, x_test, y_test):
    # normalize
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # one-hot
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


if __name__=='__main__':
    main()