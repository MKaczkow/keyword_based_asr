import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from .config import IMG_DIR


def plot_multiple_wav_grid(
    directory_path: str,
    grid_size: int = 4,
    output_file: str = f"{IMG_DIR}/spectrogram_grid.png",
) -> None:
    """Plot multiple wav files in a grid.

    Args:
        directory_path (str): Directory to search for files, which should be enumarated like 0.wav, 1.wav, 2.wav, etc.
        grid_size (int, optional): Number of images in grid's row (for given N, grid consists of N*N images). Defaults to 4.
        output_file (str, optional): File to which output spectrogram should be saved. Defaults to f'{IMG_DIR}/spectrogram_grid.png'.
    """
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))

    for i in range(grid_size * grid_size):
        file_path = os.path.join(directory_path, f"{i}.wav")
        y, sr = librosa.load(file_path, sr=None)

        D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)

        row, col = divmod(i, grid_size)
        ax = axes[row, col]
        librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log", ax=ax)
        ax.set_title(f"Spectrogram {i}")
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(output_file)
    plt.show()
    plt.close(fig)
