# AudioSet Model : https://github.com/kkoutini/passt
import torch

import numpy as np
import librosa
import cv2
import random

import ffmpeg
import soundfile as sf
from src.beat_detection2 import find_onesets
from src.helper import findAcoustics, enhanceAudios
from src.render import render_video, render_audio

import pdb
random.seed(10) # 10 for wonderful world # 1 for sleepwalk




# find beats in the audio
def findBeats(y, sr, sec_thres, max_peaks):
    # use Librosa API, keep it sparse, around min 5seconds for a video

    out = find_onesets(y, sr, sec_thres, max_peaks)
    beats = np.zeros(len(y))
    for each in out:
        if(each < sec_thres):
            continue
        beats[int(each*sr)] = 1
    return beats


def main():

    # video locations in a list
    video_locations = [
        "./videos/1.mp4",
        "./videos/2.mp4",
        "./videos/3.mp4",
        "./videos/4.mp4",
        "./videos/5.mp4",
        "./videos/6.mp4",
        "./videos/bg_speech.mp4"
        # "./videos/7.mp4", # landscape video
        # "./videos/bg_noise.mp4",
        # "./videos/bg_song.mp4",
        # "./videos/bg_speech.mp4",
    ]
    random.shuffle(video_locations)

    # load input audio
    y, sr = librosa.load("./sleepwalk.mp3")

    # find beats for the audio in a list
    
    sec_thres = 5 # 5 for what a wonderful world  # 4 for sleepwalk
    max_peaks = 10
    beats = findBeats(y, sr, sec_thres, max_peaks)
    print(beats)
    # sampling rate of audio
    # [0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 1 0 0]
    # [0 0 0 0 0 0 1 1 1 1 1 1 2 2 2 3 3 3]
    video_assignments = []
    idx = 0
    for i in range(len(beats)):
        video_assignments.append(idx)
        if beats[i] == 1:
            idx += 1

    # find reverb/environment conditions for each video, returned for each video, as a list
    conditions = findAcoustics(video_locations)

    # enhance the audio and generate different audios as a list
    audios = enhanceAudios(y, sr, conditions)

    # render the final video
    final_vid_flag = render_video(video_locations, video_assignments, sr)

    # render the final audio
    alpha = 0.7
    final_audio = render_audio(video_locations, audios, sr, video_assignments, alpha)

    sf.write("output_audio.wav", final_audio, sr, subtype="PCM_24")
    
    print(video_locations)

if __name__ == "__main__":
    main()
