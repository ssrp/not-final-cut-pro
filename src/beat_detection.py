from random import sample
import numpy as np
import librosa


def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n))


def numpyEWMA(beats, windowSize, flip=False):
    if flip:
        beats = beats[::-1]
    weights = np.exp(np.linspace(-1.0, 0.0, windowSize))
    weights /= weights.sum()

    a2D = strided_app(beats, windowSize, 1)

    returnArray = np.empty((beats.shape[0]))
    returnArray.fill(0)
    for index in range(a2D.shape[0]):
        returnArray[index + windowSize - 1] = np.convolve(weights, a2D[index])[windowSize - 1 : -windowSize + 1]
    if flip:
        returnArray = returnArray[::-1]
    return returnArray


def find_onesets(
    wave,
    sample_rate,
    strength: int = 7,
):
    onset_env = librosa.onset.onset_strength(y=wave, sr=sample_rate, aggregate=np.median, max_size=strength)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sample_rate)
    times = librosa.times_like(onset_env, sr=sample_rate, hop_length=512)
    return times[beats], tempo
