from tqdm import tqdm
import tensorflow as tf
from Utils.add_awgn import add_awgn
from Utils.load_wav import load_wav


def generate_training_data(
    speaker_paths, speaker, label, snr_dB=15, add_noise=True, trim=True, trim_len=16000
):
    wavs, labels = [], []
    for i in tqdm(speaker_paths):
        wav = load_wav(i, speaker)

        if trim:
            wav = wav[:, :trim_len]

        if wav.shape == (1, 14336):
            wav = tf.pad(wav, [[0, 0], [0, 16000 - 14336]])

        if add_noise:
            wav = add_awgn(wav, snr_dB)

        if wav.shape == (1, 16000):
            wavs.append(wav)
            labels.append(label)

    return wavs, labels
