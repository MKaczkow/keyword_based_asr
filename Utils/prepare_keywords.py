from pathlib import Path

import numpy as np
import librosa
import scipy
import nemo.collections.asr as nemo_asr

from config import UBU_DATA_DIR

print("Librosa version: ", librosa.__version__)

asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(
    model_name="QuartzNet15x5Base-En", strict=False
)

# uncomment to use proper audio file
speaker_name = "Hillary_Clinton"
# speaker_name = "Elizabeth_Glaser"
# speaker_name = 'Yamini_Ravindran'
# speaker_name = 'Barack_Obama'
# speaker_name = 'Winston_Churchill'
# speaker_name = 'John_F_Kennedy'
# speaker_name = 'Ronald_Reagan'
# audio_filename = f'{UBU_DATA_DIR}/work/split10/_rr_mono.wav'
# audio_filename = f'{UBU_DATA_DIR}/work/split10/_hc_mono.wav'
# audio_filename = f"{UBU_DATA_DIR}/work/split10/_eg_mono.wav"
# audio_filename = f'{UBU_DATA_DIR}/work/split10/_bo_mono.wav'

# uncomment to use proper audio file
audio_filename = f"{UBU_DATA_DIR}/work/split10/hc.wav"
# audio_filename = f"{UBU_DATA_DIR}/work/split10/barrackobama.wav"

signal, sample_rate = librosa.load(audio_filename, sr=None)
print("loaded")

# calculate amplitude spectrum
time_stride = 0.01
hop_length = int(sample_rate * time_stride)
n_fft = 512

# linear scale spectrogram
s = librosa.stft(y=signal, n_fft=n_fft, hop_length=hop_length)
s_db = librosa.power_to_db(np.abs(s) ** 2, ref=np.max, top_db=100)
print("librosa calculated")

# Convert our audio sample to text
files = [audio_filename]
transcript = asr_model.transcribe(paths2audio_files=files)[0]
print(f'Transcript: "{transcript}"')


# softmax implementation in NumPy
def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum(axis=-1).reshape([logits.shape[0], 1])


# let's do inference once again but without decoder
logits = asr_model.transcribe(files, logprobs=True)[0]
probs = softmax(logits)

# 20ms is duration of a timestep at output of the model
time_stride = 0.02

# get model's alphabet
labels = list(asr_model.decoder.vocabulary) + ["blank"]
labels[0] = "space"

spaces = []

state = ""
idx_state = 0

if np.argmax(probs[0]) == 0:
    state = "space"

for idx in range(1, probs.shape[0]):
    current_char_idx = np.argmax(probs[idx])
    if state == "space" and current_char_idx != 0 and current_char_idx != 28:
        spaces.append([idx_state, idx - 1])
        state = ""
    if state == "":
        if current_char_idx == 0:
            state = "space"
            idx_state = idx

if state == "space":
    spaces.append([idx_state, len(pred) - 1])

# calibration offset for timestamps: 180 ms
offset = -0.18

saved_words = []

# split the transcript into words
words = transcript.split()

pos_prev = 0

Path(f"{UBU_DATA_DIR}/keywords/{speaker_name}").mkdir(parents=False, exist_ok=True)

for j, spot in enumerate(spaces):
    saved_words.append(words[j])
    pos_end = offset + (spot[0] + spot[1]) / 2 * time_stride
    audio_chunk = signal[int(pos_prev * sample_rate) : int(pos_end * sample_rate)]
    # Save the audio chunk to a file (adjust filename as needed)
    filename = f"{UBU_DATA_DIR}/keywords/{speaker_name}/{words[j]}.wav"
    scipy.io.wavfile.write(filename, sample_rate, audio_chunk)
    pos_prev = pos_end
