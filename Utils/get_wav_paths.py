import os
from .config import DATA_DIR


def get_wav_paths(speaker):
    speaker_path = DATA_DIR + speaker
    all_paths = [item for item in os.listdir(speaker_path)]
    return all_paths
