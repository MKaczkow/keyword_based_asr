import tensorflow as tf
from .config import DATA_DIR


def load_wav(wav_path, speaker):
    wav_path = DATA_DIR + speaker + "/" + wav_path

    with open(wav_path, "rb") as f:
        wav_data = f.read()

    wav_data, _ = tf.audio.decode_wav(wav_data, desired_channels=1)
    wav_data = tf.reshape(wav_data, [1, -1])

    return wav_data
