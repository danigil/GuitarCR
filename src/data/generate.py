import os, glob
import logging
import librosa, librosa.display
import pandas as pd

from settings import *
from src.data.preprocessing import *
from setup_logging import setup_logging

setup_logging()
logger = logging.getLogger('src.data.generate')

def run():
    # Generate raw data
    
    
    
    for instrument in instruments:
    
        logger.info("Generating Raw Metadata")
        data_df_raw = (pd.DataFrame().pipe(construct_dataframe, instrument)
                                    .pipe(get_spectrogram, instrument)
                                    .pipe(add_shape)
                    )
        logger.info("Raw Metadata Generated")

        # Save raw data
        data_df_raw.to_csv(os.path.join(METADATA_DIR_RAW, f'data_{instrument.lower()}.csv'), index=False)
        data_df_raw.to_pickle(os.path.join(METADATA_DIR_RAW, f'data_{instrument.lower()}.pkl'))

        logger.info("Raw Metadata saved to"+METADATA_DIR_RAW)
        logger.info(get_count(data_df_raw))

        # Process and save processed data
        data_df_processed = process(data_df_raw)

        data_df_processed.to_csv(os.path.join(METADATA_DIR_PROCESSED, f'data_{instrument.lower()}.csv'), index=False)
        data_df_processed.to_pickle(os.path.join(METADATA_DIR_PROCESSED, f'data_{instrument.lower()}.pkl'))

        logger.info("Processed Metadata saved to"+METADATA_DIR_PROCESSED)
        logger.info(get_count(data_df_processed))

def my_run(instruments = ['Guitar', 'Accordion', 'Piano', 'Violin'], normalized=False, ood=False):
    max_size = -np.inf

    for instrument in instruments:
        data_df_raw = (pd.DataFrame().pipe(construct_dataframe, instrument, ood)
                                    .pipe(get_spectrogram, instrument, normalized, ood)
                                    .pipe(add_shape)
                    )
        
        if(instrument == 'Guitar'):
            print('Guitar DF raw size: ' + str(data_df_raw.shape))
        
        current_size = get_max_shape(data_df_raw)
        max_size = max(max_size, current_size)
        
        if normalized:
            if ood:
                SAVE_PATH_RAW = METADATA_DIR_RAW_NORMALIZED_OOD
                SAVE_PATH_PROCESSED = METADATA_DIR_PROCESSED_NORMALIZED_OOD
            else:
                SAVE_PATH_RAW = METADATA_DIR_RAW_NORMALIZED
                SAVE_PATH_PROCESSED = METADATA_DIR_PROCESSED_NORMALIZED
        else:
            SAVE_PATH_RAW = METADATA_DIR_RAW
            SAVE_PATH_PROCESSED = METADATA_DIR_PROCESSED

        data_df_raw.to_csv(os.path.join(SAVE_PATH_RAW, f'data_{instrument.lower()}.csv'), index=False)
        data_df_raw.to_pickle(os.path.join(SAVE_PATH_RAW, f'data_{instrument.lower()}.pkl'))
    
    for instrument in instruments:
        data_df_raw = pd.read_pickle(os.path.join(SAVE_PATH_RAW, f'data_{instrument.lower()}.pkl'))

        data_df_processed = my_process(data_df_raw, max_size)
        data_df_processed.to_csv(os.path.join(SAVE_PATH_PROCESSED, f'data_{instrument.lower()}.csv'), index=False)
        data_df_processed.to_pickle(os.path.join(SAVE_PATH_PROCESSED, f'data_{instrument.lower()}.pkl'))





if  __name__ =='__main__':
    my_run(instruments = ['Guitar'], normalized=True, ood=True)