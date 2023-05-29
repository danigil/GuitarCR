from pydub import AudioSegment
from settings import *
import os
import glob
from pathlib import Path

instruments = ['guitar', 'piano']
from itertools import combinations

paths = {
    'guitar': DATA_DIR_GUITAR,
    'violin': DATA_DIR_VIOLIN,
    'accordion': DATA_DIR_ACCORDION,
    'piano': DATA_DIR_PIANO
}

for inst1, inst2 in combinations(sorted(instruments), 2):
    export_root_dir = os.path.join(DATA_DIR_COMBINED, f'{inst1}_{inst2}')
    os.makedirs(export_root_dir, exist_ok=True)
    
    for filename in os.listdir(paths[inst1]):
        folder_inst1 = os.path.join(paths[inst1], filename)
        folder_inst2 = os.path.join(paths[inst2], filename)
        
        if not os.path.isdir(folder_inst1) or not os.path.isdir(folder_inst2):
            print("ERROR")
        
        export_note_dir = os.path.join(export_root_dir, filename)
        os.makedirs(export_note_dir, exist_ok=True)
        
        for aud1 in glob.glob(folder_inst1 + "/*.wav"):
            for aud2 in glob.glob(folder_inst2 + "/*.wav"):
                sound1 = AudioSegment.from_file(aud1)
                sound2 = AudioSegment.from_file(aud2)

                combined = sound1.overlay(sound2)
                
                stem1 = Path(aud1).stem
                stem2 = Path(aud2).stem
                
                export_path = os.path.join(export_note_dir, f'{stem1}_{stem2}.wav')
                combined.export(export_path, format='wav')
        
        

