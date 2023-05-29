import sys
import os

module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)

import pandas as pd
from src.metrics import *
from settings import *
from src.data import generate
import random
import keras
import os, glob
import logging
import librosa, librosa.display

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from keras.models import model_from_json
from keras import backend as K

from src.processing import train_test_split, features_target_split, reshape_feature_CNN, one_hot_encode

from src.model import CNN
from src.data.preprocessing import get_most_shape
from setup_logging import setup_logging

def my_train(cnn, train_data, test_data, size, epochs=15):
    setup_logging()
    logger = logging.getLogger('src.train')

    logger.info(f"Number of train samples: {len(train_data)}")
    logger.info(f"Number of test samples: {len(test_data)}")
    # most_shape = get_most_shape(dataset)
    #train_data, test_data = train_test_split(dataset, augmented=augmented, split_ratio=0.65)

    X_train, y_train = features_target_split(train_data)
    X_test, y_test = features_target_split(test_data)

    # Reshape for CNN input
    X_train, X_test = reshape_feature_CNN(X_train, size=size), reshape_feature_CNN(X_test, size=size)

    # Preserve y_test values
    y_test_values = y_test.copy()

    # One-Hot encoding for classes
    y_train, y_test = one_hot_encode(y_train), one_hot_encode(y_test)

    # Instance of CNN model
    logger.info(str(cnn))

    cnn.train(X_train, y_train, X_test, y_test, epochs=epochs)
    cnn.evaluate(X_train, y_train, X_test, y_test)

    if tf.__version__ != '1.8.0':
        predict_x=cnn.model.predict(X_test)
        predictions = np.argmax(predict_x,axis=1)
    else:    
        predictions = cnn.model.predict_classes(X_test)
    conf_matrix=confusion_matrix(y_test_values, predictions, labels=range(10))
    logger.info('Confusion Matrix for classes {}:\n{}'.format(CLASSES, conf_matrix))

    return cnn

def load_data(RAW_PATH, AUGMENTED_PATH, instruments=['Guitar'], instruments_aug=['Accordion', 'Violin', 'Piano']):
    datasets_raw = [pd.read_pickle(os.path.join(RAW_PATH, f'data_{instrument.lower()}.pkl')) for instrument in instruments]
    datasets_augmented = [pd.read_pickle(os.path.join(AUGMENTED_PATH, f'data_{instrument.lower()}.pkl')) for instrument in instruments_aug]

    from src.data.preprocessing import get_max_shape
    max_spectrogram_size = max(map(lambda df: get_max_shape(df), datasets_raw+datasets_augmented))

    from src.data.preprocessing import uniform_shape
    uniform = lambda df: uniform_shape(df, max_spectrogram_size)

    datasets_raw = list(map(uniform,datasets_raw))
    datasets_augmented = list(map(uniform,datasets_augmented))
    datasets_augmented = list(map(lambda df: df[['spectrogram','class_ID', 'class_name','augmentation']],datasets_augmented))
    datasets_augmented = list(map(lambda df: df.reset_index(drop=True), datasets_augmented))

    return datasets_raw, datasets_augmented, max_spectrogram_size

def train_test(datasets_raw, datasets_augmented):
    train_datas = []
    test_datas = []

    for dataset in datasets_raw:
        train_data, test_data = train_test_split(dataset, augmented=False, split_ratio=0.65)
        train_datas.append(train_data)
        test_datas.append(test_data)

    for dataset in datasets_augmented:
        train_data, test_data = train_test_split(dataset, augmented=True, split_ratio=0.65)
        train_datas.append(train_data)
        test_datas.append(test_data)

    train_data = pd.concat(train_datas)
    test_data = pd.concat(test_datas)

    train_data = train_data.sample(frac=1).reset_index(drop=True)
    test_data = test_data.sample(frac=1).reset_index(drop=True)

    return train_data, test_data, test_datas

def test_by_instrument(cnn, test_datas, size,instruments=['Guitar'], instruments_aug=['Accordion', 'Violin', 'Piano']):
    test_instruments = instruments + instruments_aug
    for test_data, instrument in zip(test_datas, test_instruments):
        X_test = test_data['spectrogram']
        X_test = np.array([x.reshape( (128, size, 1) ) for x in X_test])
        y_test = test_data['class_ID']

        y_test_values=y_test
        y_test = np.array(keras.utils.to_categorical(y_test, 10))

        score = cnn.model.evaluate(X_test,y_test)
        print(f'Test score for instrument: {instrument}')
        print('\tTest loss:', score[0])
        print('\tTest accuracy:', score[1])
        print('\tTest precision:', score[2])
        print('\tTest recall:', score[3])
        print('\tTest f1-score:', score[4])

#cnn.save_model(name="model_all_data_augment_1")