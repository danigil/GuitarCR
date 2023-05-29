import pandas as pd
import os, sys

import tensorflow as tf

module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

from settings import *
from notebooks.my_train import *
from src.model import CNN, CNN_nodropout
import librosa

#input_path = './data/audio/jim2012Chrords/combined/guitar_piano/a/a1_a1.wav'

input_path = './data/audio/jim2012Chrords/Guitar_Only/a/a1.wav'
assert os.path.isfile(input_path)
y, sr = librosa.load(input_path, sr=None, duration=2)
spectrogram = librosa.util.normalize(np.log(librosa.feature.melspectrogram(y=y,sr=sr, n_mels=128) + 1e-9))

most_shape = (128, 213)

spectrogram = np.pad(spectrogram, ((0, 0), (0, most_shape[1]-spectrogram.shape[1])), 'constant')
spectrogram = spectrogram.reshape((1,)+most_shape+(1,))

experiment1 = CNN(most_shape)
experiment1.load_model('experiment1_model')

predict_x=experiment1.model.predict(spectrogram, batch_size=1)
predictions = np.argmax(predict_x,axis=1)

print(f'EXPERIMENT1 MODEL OUTPUT: {CLASSES[predictions[0]]}')

experiment3 = CNN_nodropout(most_shape)
experiment3.load_model('experiment3_model_5epochs_better')

predict_x=experiment3.model.predict(spectrogram, batch_size=1)
predictions = np.argmax(predict_x,axis=1)

print(f'EXPERIMENT3 MODEL OUTPUT: {CLASSES[predictions[0]]}')
