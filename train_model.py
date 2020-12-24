import tensorflow as tf
import numpy as np

from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

from load_module import load_train_test_data


def prepare_data() -> tuple:
    x_train, y_train, x_test, y_test = load_train_test_data(0.8)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    return x_train, y_train, x_test, y_test


def train_model():
    x_train, y_train, x_test, y_test = prepare_data()
    input_shape = (28, 28, 1)

    n_folds = 3

    k_fold = StratifiedKFold(n_splits=n_folds, shuffle=True)

    i = 0
    for train, test in k_fold.split(x_train, y_train):
        print('-------------------------')
        print("Running Fold", i + 1, "/", n_folds)
        model = Sequential()
        model.add(Conv2D(28, input_shape=input_shape, kernel_size=(3, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation=tf.nn.relu))
        model.add(Dropout(0.2))
        model.add(Dense(4, activation=tf.nn.softmax))

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train[train], y_train[train], epochs=100)
        print('----------Score----------')
        print(model.evaluate(x_train[test], y_train[test]))
        i += 1
    print(model.evaluate(x_test, y_test))


if __name__ == '__main__':
    train_model()
