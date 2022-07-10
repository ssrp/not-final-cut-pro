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
    sec_thres, max_peaks
):
    onset_env = librosa.onset.onset_strength(y=wave, sr=sample_rate, aggregate=np.median, max_size=7)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sample_rate)
    times = librosa.times_like(onset_env, sr=sample_rate, hop_length=512)

    
    outs = []
    wave = np.concatenate((wave, np.zeros(int(sample_rate/10.0))))
    for i in range(len(wave)):
        outs.append(np.sum(np.dot(wave[i:i+int(sample_rate/10.0)],wave[i:i+int(sample_rate/10.0)])))
    en = np.array(outs)


    sort_arr = np.argsort(en[:-int(sample_rate/10.0)])/sample_rate
    timestamps = []
    for timestamp in sort_arr:
        flag = True
        for ts in timestamps:
            if(np.abs(ts - timestamp) < sec_thres):
                flag = False
                break
        if flag:
            timestamps.append(timestamp)
            if(len(timestamps) > max_peaks):
                break
    timestamps = np.sort(timestamps)

    ix = 0
    iy = 0
    out = []
    while(ix < len(timestamps) and iy < len(times[beats])):
        if(times[beats][iy] < timestamps[ix]):
            iy += 1
        elif(times[beats][iy] >= timestamps[ix]):
            if(iy == 0):
                out.append(times[beats][iy])
            else:
                out.append(times[beats][iy-1])
            ix += 1
            iy += 1
    
    return out
