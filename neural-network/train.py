import os
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

from model import ExportModel, get_spectrogram, get_hamming

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

DATASET_PATH = 'assetss'
EPOCHS = 10
RATE = 44100
FRAGMENT_LENGTH = int(RATE / 10)
WITH_HUMMING = True
DURATION = round(1 / (RATE / FRAGMENT_LENGTH), 2)

# utils


def squeeze(audio, labels):
    audio = tf.squeeze(audio, axis=-1)
    return audio, labels


def ds_to_spectrogram(ds):
    return ds.map(
        map_func=lambda audio, label: (get_spectrogram(audio), label),
        num_parallel_calls=tf.data.AUTOTUNE)

def ds_to_hamming(ds):
    return ds.map(
        map_func=lambda audio, label: (get_hamming(audio), label),
        num_parallel_calls=tf.data.AUTOTUNE)


# Form data storage
data_dir = pathlib.Path(os.path.join(DATASET_PATH, 'data_set_{}s'.format(DURATION), 'train'))
train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=data_dir,
    batch_size=32,
    validation_split=0.2,
    seed=0,
    output_sequence_length=FRAGMENT_LENGTH,
    subset='both')
label_names = np.array(train_ds.class_names)

# Prepare data - wave to spectrogram
train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)
val_ds = val_ds.shard(num_shards=2, index=1)

train_hamming_ds = ds_to_hamming(train_ds)
val_hamming_ds = ds_to_hamming(val_ds)


train_spectrogram_ds = ds_to_spectrogram(train_ds if WITH_HUMMING == False else train_hamming_ds)
val_spectrogram_ds = ds_to_spectrogram(train_ds if WITH_HUMMING == False else train_hamming_ds)


# Training model
train_spectrogram_ds = train_spectrogram_ds.cache().shuffle(
    10000).prefetch(tf.data.AUTOTUNE)
val_spectrogram_ds = val_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)

for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
    input_shape = example_spectrograms.shape[1:]
    num_labels = len(label_names)

    # Instantiate the `tf.keras.layers.Normalization` layer.
    norm_layer = layers.Normalization()
    # Fit the state of the layer to the spectrograms
    # with `Normalization.adapt`.
    norm_layer.adapt(data=train_spectrogram_ds.map(
        map_func=lambda spec, label: spec))

    model = models.Sequential([
        layers.Input(shape=input_shape),
        # Downsample the input.
        layers.Resizing(32, 32),
        # Normalize.
        norm_layer,
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels),
    ])

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    model.fit(
        train_spectrogram_ds,
        validation_data=val_spectrogram_ds,
        epochs=EPOCHS,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
    )

# Save model
export = ExportModel(model=model, label_names=label_names, hamming=WITH_HUMMING, fragment_length=FRAGMENT_LENGTH)
model_dir = pathlib.Path(os.path.join(DATASET_PATH, 'rln-model_{}s'.format(DURATION)))
tf.saved_model.save(export, model_dir)

print('Model is saved to: {}'.format(model_dir))
