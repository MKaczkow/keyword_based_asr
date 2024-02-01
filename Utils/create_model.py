import tensorflow as tf
from spela.spectrogram import Spectrogram
from spela.melspectrogram import Melspectrogram


def create_model(
    speech_feature: str, num_classes: int = 9, input_shape: tuple = (1, 16000)
):
    model = tf.keras.Sequential()
    if speech_feature == "spectrogram":
        model.add(
            Spectrogram(
                n_dft=512,
                n_hop=256,
                input_shape=input_shape,
                return_decibel_spectrogram=True,
                power_spectrogram=2.0,
                trainable_kernel=False,
                name="static_stft",
            )
        )
    elif speech_feature == "melspectrogram":
        model.add(
            Melspectrogram(
                sr=input_shape[1],
                n_mels=128,
                n_dft=512,
                n_hop=256,
                input_shape=input_shape,
                return_decibel_melgram=True,
                trainable_kernel=False,
                name="melgram",
            )
        )

    model.add(tf.keras.layers.Conv2D(16, (2, 2), activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=3e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
