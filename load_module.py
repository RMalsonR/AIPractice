import os
import cv2
import numpy as np

from settings import PATH_TO_IMAGES


def collect_and_transform_data() -> list:
    def get_valid_label_value_for_keras(folder_int: int):
        return folder_int / 10

    tuple_of_zipped_category = []
    for folder in os.listdir(PATH_TO_IMAGES):
        if folder == '.DS_Store':
            continue
        data = []
        labels = []
        folder_path = os.path.join(PATH_TO_IMAGES, folder)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if file == '.DS_Store':
                continue
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (28, 28))
            image = image.astype('float32')
            image /= 255
            data.append(image)
            labels.append(get_valid_label_value_for_keras(int(folder)))
        tuple_of_zipped_category.append((data, labels))
    return tuple_of_zipped_category


def load_train_test_data(data_split_percent: float) -> tuple:
    if not isinstance(data_split_percent, float):
        data_split_percent /= 100
    x_train = []
    y_train = []

    x_test = []
    y_test = []

    tuple_data = collect_and_transform_data()

    for batch in tuple_data:
        data, labels = batch
        data = np.array(data)
        labels = np.array(labels)

        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)

        data = data[indices]
        labels = labels[indices]

        end_idx = int(len(data) * data_split_percent)
        for idx, val in enumerate(data[:end_idx]):
            x_train.append(data[idx])
            y_train.append(labels[idx])

        for idx, val in enumerate(data[end_idx+1:]):
            x_test.append(data[idx])
            y_test.append(labels[idx])

    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
