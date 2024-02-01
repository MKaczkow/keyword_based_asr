import librosa
import numpy as np
import matplotlib.pyplot as plt
from .config import IMG_DIR


def plot_single_wav(
    file_path: str, output_file: str = f"{IMG_DIR}/spectrogram.png"
) -> None:
    """Plot single wav file.

    Args:
        file_path (str): Directory to search for files, which should be enumarated like 0.wav, 1.wav, 2.wav, etc.
        output_file (str, optional): File to which output spectrogram should be saved. Defaults to f'{IMG_DIR}/spectrogram.png'.. Defaults to f'{IMG_DIR}/spectrogram.png':str.
    """
    y, sr = librosa.load(file_path, sr=None)

    D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)

    plt.figure(figsize=(10, 6))
    librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram")
    plt.savefig(output_file)
    plt.show()
    plt.close()
