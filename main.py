import torch
import json
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


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

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


def main(video_locations = None, audio_input = None, shuffle_videos = None, sec_threshold = 3, videochange_threshold = 10):

    print(video_locations, audio_input)

    if video_locations == None or audio_input == None:
            raise "Please add video files list and audio file location."
    
    if video_locations is not None:
        # video locations in a list
        video_locations = [
            "./videos/1.mp4",
            "./videos/2.mp4",
            "./videos/3.mp4",
            "./videos/4.mp4",
            "./videos/5.mp4",
            "./videos/6.mp4",
            "./videos/bg_speech.mp4",
            "./videos/8.mp4"
        ]
    if shuffle_videos == 1:
        random.shuffle(video_locations)

    # load input audio
    y, sr = librosa.load(audio_input)

    # find beats for the audio in a list
    sec_thres = sec_threshold # 5 for what a wonderful world  # 4 for sleepwalk
    max_peaks = videochange_threshold # max number of peaks in the audio
    beats = findBeats(y, sr, sec_thres, max_peaks)

    # video cut assignments
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
    scenes, audios = enhanceAudios(y, sr, conditions, video_locations)

    # render the final video
    final_vid_flag = render_video(video_locations, video_assignments, sr)

    # render the final audio
    final_audio = render_audio(scenes, audios, sr, video_assignments)

    sf.write("./tmp/output_audio.wav", final_audio, sr, subtype="PCM_24")
    
if __name__ == "__main__":
    config = json.load(open("./config.json", "r"))
    config = AttrDict(config)
    main(video_locations=config.videos, audio_input=config.audio, shuffle_videos=config.shuffle_videos,sec_threshold = config.sec_threshold, videochange_threshold = config.videochange_threshold)
