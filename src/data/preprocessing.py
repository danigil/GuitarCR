import os, glob
import logging
import librosa, librosa.display
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler

from settings import *
#from setup_logging import setup_logging

#setup_logging()
logger = logging.getLogger('src.data.preprocessing')

def df_info(f):
    def inner(df, *args, **kwargs):
        result = f(df, *args, **kwargs)
        logger.info(f"After applying {f.__name__}, shape of df = {result.shape }")
        logger.info(f"Columns of df are {df.columns}\n")
        return result
    return inner

def get_path(instrument='Guitar', ood=False):
    if instrument=='Guitar':
        path = DATA_DIR_GUITAR
        s = 'Only'
    else:
        s = instrument
        if instrument=='Piano':
            path = DATA_DIR_PIANO
        elif instrument=='Accordion':
            path = DATA_DIR_ACCORDION
        elif instrument=='Violin':
            path = DATA_DIR_VIOLIN
        
    if ood:
        return DATA_DIR_DATASET_OOD1_GUITAR_SPLIT, 'guitar_splits'
            
    return path, s

@df_info
def construct_dataframe(df, instrument='Guitar', ood=False):
    """
    Construct Dataframe with all required values
    """

    path, s = get_path(instrument, ood)

    logger.info(f"Construct DataFrame for raw data ({instrument})")
    file_path = glob.glob(path + "**/*.wav")
    df['file_path'] = file_path
    df['file_path'] = df['file_path'].map(lambda x: x[x.rindex(f'{s}/')+len(f'{s}/'):])
    df['file_name'] = df['file_path'].map(lambda x: x[x.rindex('/')+1:])
    df['class_name'] = df['file_path'].map(lambda x: x[:x.index('/')])
    df['class_ID'] = df['class_name'].map(lambda x: CLASSES_MAP[x])
    logger.info(f"Construct DataFrame for raw data ({instrument}) completed")
    print(df.head())
    return df.copy()

@df_info
def get_spectrogram(df, instrument='Guitar', normalized=False, ood=False):
    logger.info("Extract spectrogram")
    """Extract spectrogram from audio"""
    
    path, _ = get_path(instrument, ood)
    if not normalized:
        df['audio_series'] = df['file_path'].map(lambda x: librosa.load(path \
                                                                        + x, duration=2))
        df['y'] = df['audio_series'].map(lambda x: x[0])
        df['sr'] = df['audio_series'].map(lambda x: x[1])
        df['spectrogram'] = df.apply(lambda row: librosa.feature.melspectrogram(y=row['y'],\
            sr=row['sr']), axis=1)
        df.drop(columns='audio_series', inplace=True)
    else:
        df['audio_series'] = df['file_path'].map(lambda x: librosa.load(path \
                                                                        + x, duration=2, sr=None))
        df['y'] = df['audio_series'].map(lambda x: librosa.util.normalize(x[0]))
        df['sr'] = df['audio_series'].map(lambda x: x[1])
        #df['stft'] = df['y'].map(lambda x: librosa.core.stft(x, n_fft = 256, hop_length=16))
        #df['spectrogram'] = df.apply(lambda row: librosa.util.normalize(np.log(librosa.feature.melspectrogram(S=row['stft']), axis=1)))
        #df.drop(columns=['audio_series','stft'], inplace=True)
        df['spectrogram'] = df.apply(lambda row: librosa.util.normalize(np.log(librosa.feature.melspectrogram(y=row['y'], sr=row['sr']) + 1e-9)), axis=1)
        df.drop(columns='audio_series', inplace=True)
    logger.info("Extract spectorgram completed")
    return df

@df_info
def add_shape(df):
    df['shape'] = df['spectrogram'].map(lambda x: x.shape)
    return df

def get_most_shape(df):
    most_shape = df['spectrogram'].map(lambda x: x.shape).value_counts().index[0]
    print(f"The most frequent shape is {most_shape}")
    return most_shape

from itertools import groupby

def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)

def get_max_shape(df):
    assert all_equal(df['spectrogram'].map(lambda x: x.shape[0]))

    max_size = max(df['spectrogram'].map(lambda x: x.shape[1]))
    return max_size

# Maintain same shape
@df_info
def clean_shape(df):
    logger.info("Filter data using most frequent shape")
    most_shape = get_most_shape(df)
    df = df[df['shape']==most_shape]
    df.drop(columns='shape', inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info("Filter data using most frequent shape completed")
    return df

def uniform_shape(df, size=None):
    if size is None:
        max_size = get_max_shape(df)

    max_size = size
    assert get_max_shape(df) <= max_size
    
    df['spectrogram'] = df['spectrogram'].apply(lambda x: np.pad(x, ((0, 0), (0, max_size-x.shape[1])), 'constant'))
    return df



# Create processed dataframe
@df_info
def process(df):
    logger.info("Process raw data")
    df = (df.pipe(clean_shape)
                .pipe(over_sample)
    )
    df = df[['spectrogram','class_ID', 'class_name']]
    logger.info("Process raw data completed")
    return df

def my_process(df, size=None):
    df = uniform_shape(df, size)
    df = df[['spectrogram','class_ID', 'class_name']]

    return df

def get_count(df):
    return df['class_name'].value_counts()

def get_class(class_ID):
    return list(CLASSES_MAP.keys())[list(CLASSES_MAP.values()).index(class_ID)]

@df_info
def over_sample(df):
    logger.info("Oversample data to balance classes")
    oversample = RandomOverSampler(sampling_strategy='auto')
    X, y = df['spectrogram'].values, df['class_ID'].values
    X = X.reshape(-1, 1)
    X, y = oversample.fit_resample(X, y)
    df = pd.DataFrame()
    df['spectrogram'] = pd.Series([np.array(x[0]) for x in X])
    df['class_ID'] = pd.Series(y)
    df['class_name'] = df['class_ID'].map(lambda x: get_class(x))
    logger.info("Oversample data to balance classes completed")
    return df