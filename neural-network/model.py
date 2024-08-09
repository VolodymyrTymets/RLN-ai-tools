import tensorflow as tf

def get_hamming(waveform):
    window = tf.signal.hamming_window(window_length=waveform.shape[1])
    return waveform * window

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
    def __init__(self, model, label_names, hamming, fragment_length):
        self.model = model
        self.label_names = label_names
        self.hamming = hamming
        self.fragment_length = fragment_length

        # Accept either a string-filename or a batch of waveforms.
        # YOu could add additional signatures for a single wave, or a ragged-batch.
        self.__call__.get_concrete_function(
            x=tf.TensorSpec(shape=(), dtype=tf.string))
        self.__call__.get_concrete_function(
            x=tf.TensorSpec(shape=[None, fragment_length], dtype=tf.float32))

    @tf.function
    def __call__(self, x):
        # If they pass a string, load the file and decode it.
        if x.dtype == tf.string:
            x = tf.io.read_file(x)
            x, _ = tf.audio.decode_wav(
                x, desired_channels=1, desired_samples=self.fragment_length,)
            x = tf.squeeze(x, axis=-1)
            x = x[tf.newaxis, :]
        if(self.hamming):    
            # hummming before hamming    
            x = get_hamming(x)
        x = get_spectrogram(x)
        result = self.model(x, training=False)

        class_ids = tf.argmax(result, axis=-1)
        class_names = tf.gather(self.label_names, class_ids)
        return {'predictions': result,
                'class_ids': class_ids,
                'class_names': class_names,
                'label_names': self.label_names}
