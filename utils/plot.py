import os
import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt
from audio_features.audio_features import TimeDomainFeatures, FrequencyDomainFeatures


DATASET_PATH = '../sanbox/assets'
frame_length = 128
hop_length = frame_length // 4

b_1, sr = librosa.load(os.path.join(DATASET_PATH, 'breath', '1.wav'))
b_2, _ = librosa.load(os.path.join(DATASET_PATH, 'breath', '2.wav'))
b_3, _ = librosa.load(os.path.join(DATASET_PATH, 'breath', '3.wav'))

n_1, _ = librosa.load(os.path.join(DATASET_PATH, 'nerve', '1.wav'))
n_2, _ = librosa.load(os.path.join(DATASET_PATH, 'nerve', '2.wav'))
n_3, _ = librosa.load(os.path.join(DATASET_PATH, 'nerve', '3.wav'))


t_domain = TimeDomainFeatures()
f_domain = FrequencyDomainFeatures()

figsize = (7,3)

## Amplitude envelope
b_1_AE = t_domain.AE(signal=b_1, frame_length=frame_length, hop_length=hop_length)
n_1_AE = t_domain.AE(signal=n_1, frame_length=frame_length, hop_length=hop_length)

# plt.figure(figsize=figsize)
# # ax = plt.subplot(1, 1, 1)
# # #librosa.display.waveshow(b_1, color="black")
# # plt.plot(librosa.frames_to_time(frames=range(len(b_1_AE)), hop_length=hop_length), b_1_AE, color="black")
# # plt.ylim((-1, 1))
# # plt.title("Breath")
#
# plt.subplot(1, 1, 1)
# # librosa.display.waveshow(n_1, color="black")
# plt.plot(librosa.frames_to_time(frames=range(len(n_1_AE)), hop_length=hop_length), n_1_AE, color="black")
# plt.ylim((-1, 1))
# plt.title("Stimulation")

# plt.show()


## Amplitude envelope
b_1_RSME = t_domain.RSME(signal=b_1, frame_length=frame_length, hop_length=hop_length)
n_1_RSME = t_domain.RSME(signal=n_1, frame_length=frame_length, hop_length=hop_length)

# plt.figure(figsize=figsize)
# plt.xlabel('Frames')
# plt.ylabel('RMS Energy')
# ax = plt.subplot(1, 1, 1)
# plt.plot(b_1_RSME, color="black")
# plt.title("Breath")
#
# # plt.subplot(1, 1, 1)
# # plt.plot( n_1_RSME, color="black")
# # plt.title("Stimulation")
#
# plt.show()


## Zero crossing rate
# b_1_ZCR = t_domain.ZCR(signal=b_1, frame_length=frame_length, hop_length=hop_length)
# n_1_ZCR = t_domain.ZCR(signal=n_1, frame_length=frame_length, hop_length=hop_length)
#
# plt.figure(figsize=figsize)
# plt.xlabel('Frames')
# plt.ylabel('ZCR')
#
# # ax = plt.subplot(1, 1, 1)
# # plt.plot(b_1_ZCR, color="black")
# # plt.title("Breath")
#
# plt.subplot(1, 1, 1)
# plt.plot( n_1_ZCR, color="black")
# plt.title("Stimulation")
#
# plt.show()



## spectr
# magnitude_b, frequency_b = f_domain.fft(signal=b_1, sr=sr, frame_length=frame_length, hop_length=hop_length)
# magnitude_n, frequency_n = f_domain.fft(signal=n_1, sr=sr, frame_length=frame_length, hop_length=hop_length)
#
# plt.figure(figsize=(7,4))
# # plt.xlabel('Frequency Hz')
# # plt.ylabel('Magnitude')
# # plt.plot( frequency_b, magnitude_b, color="black")
# # plt.title("Breath")
#
# plt.xlabel('Frequency Hz')
# plt.ylabel('Magnitude')
# plt.plot( frequency_n, magnitude_n, color="black")
# plt.title("Stimulation")
#
# plt.show()



## spectrogram
# y_b = f_domain.stft(signal=b_1, frame_length=frame_length, hop_length=hop_length)
# y_n = f_domain.stft(signal=n_1, frame_length=frame_length, hop_length=hop_length)
#
# def plot_spectrogram(Y, sr, hop_length, y_axis="log"):
#   plt.figure(figsize=(10, 5))
#   librosa.display.specshow(Y,
#                            sr=sr,
#                            hop_length=hop_length,
#                            x_axis="time",
#                            y_axis=y_axis)
#   plt.colorbar(format="%+2.f db")
#
#
# plot_spectrogram(librosa.power_to_db(y_n), sr, hop_length)
# # plt.title("Breath")
# plt.title("Stimulation")
# plt.show()
## Band energy ratio
frame_length = 2048
hop_length = frame_length // 4
y_b = f_domain.stft(signal=b_1, frame_length=frame_length, hop_length=hop_length)
y_n = f_domain.stft(signal=n_1, frame_length=frame_length, hop_length=hop_length)
b_1_BER = f_domain.BER(stft=y_b, sr=sr, split_frequency=20000)
n_1_BER = f_domain.BER(stft=y_n, sr=sr, split_frequency=20000)
frames = range(len(b_1_BER))
t = librosa.frames_to_time(frames, hop_length=hop_length)
diff= int(len(b_1_BER) - len(n_1_BER))
n_1_BER = np.concatenate((n_1_BER, np.zeros(diff, dtype=int)))

plt.figure(figsize=(14, 5))


# plt.xlabel('Frames')
# plt.ylabel('BER')

plt.subplot(1, 2, 1)
plt.plot(b_1_BER, color="black")
plt.title("Breath")
plt.ylim((0, 140000))
plt.xlabel('Frames')
plt.ylabel('BER')

plt.subplot(1, 2, 2)
plt.plot( n_1_BER, color="black")
plt.xlabel('Frames')
plt.ylabel('BER')
plt.title("Stimulation")
# plt.tight_layout()
plt.show()

## melt spectrogram
# frame_length = 1024
# hop_length = frame_length // 4

# plt.figure(figsize=(10, 5))
# librosa.display.specshow(f_domain.melfilters(sr=sr, frame_length=frame_length, n_mels=10),
#                          y_axis="mel",
#                          # fmin=librosa.note_to_hz('C1'),
#                          sr=sr,
#                          x_axis="linear")
# plt.colorbar(format="%+2.f db")
# plt.show()
#
# plt.figure(figsize=(10, 5))
# mel_spectrogram = f_domain.melspectogram(signal=b_1, sr=sr, frame_length=frame_length, hop_length=hop_length, n_mels=10)
# plt.title("Breath")
# # mel_spectrogram = f_domain.melspectogram(signal=n_1, sr=sr, frame_length=frame_length, hop_length=hop_length, n_mels=10)
# # plt.title("Stimulation")
#
# librosa.display.specshow(mel_spectrogram, x_axis="time", y_axis="mel", sr=sr)
# plt.colorbar(format="%+2.f db")
# plt.show()

## MFCCs
# plt.figure(figsize=(10, 5))
# mfccs = f_domain.mfcc(signal=b_1, sr=sr, n_mfcc=12)
# plt.title("Breath")
# # mfccs = f_domain.mfcc(signal=n_1, sr=sr, n_mfcc=12)
# # plt.title("Stimulation")
# librosa.display.specshow(mfccs,
#                          x_axis="time",
#                          sr=sr)
# plt.colorbar(format="%+2.f")
# plt.show()