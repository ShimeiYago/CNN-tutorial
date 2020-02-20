import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
import models.mnist as models


def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

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
    # reshape (samples, 28, 28, channels)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

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