import tensorflow as tf

RATE = 44100
FRAGMENT_LENGTH = int(RATE / 10)

@tf.function(input_signature=[tf.TensorSpec(shape=[None, FRAGMENT_LENGTH], dtype=tf.float32)])
def get_hamming(waveform):
    window = tf.signal.hamming_window(window_length=waveform.shape[1])
    return waveform * window

@tf.function(input_signature=[tf.TensorSpec(shape=[None, FRAGMENT_LENGTH], dtype=tf.float32)])
def get_spectrogram(waveform):
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(
        waveform, frame_length=256, frame_step=128)

    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram


class ExportModel(tf.Module):
    def __init__(self, model, label_names, hamming, fragment_length = FRAGMENT_LENGTH):
        self.model = model
        self.label_names = label_names
        self.hamming = hamming
        self.fragment_length = fragment_length

        self.__call__.get_concrete_function(
            x=tf.TensorSpec(shape=[None, FRAGMENT_LENGTH], dtype=tf.float32))

    @tf.function
    def __call__(self, x):
        x = get_spectrogram(x)
        result = self.model(x, training=False)

        class_ids = tf.argmax(result, axis=-1)
        class_names = tf.gather(self.label_names, class_ids)
        return {'predictions': result,
                'class_ids': class_ids,
                'class_names': class_names,
                'label_names': self.label_names}
    