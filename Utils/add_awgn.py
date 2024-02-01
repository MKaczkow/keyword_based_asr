import numpy as np


def add_awgn(signal, snr_dB):
    # Calculate noise power
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_dB / 10.0))

    # Generate Gaussian noise
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))

    # Add noise to the signal
    noisy_signal = signal + noise
    return noisy_signal
