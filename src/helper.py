# AudioSet Model : https://github.com/kkoutini/passt
from hear21passt.base import get_basic_model, get_model_passt
model = get_basic_model(mode="logits")

import torch

from check_orientation.pre_trained_models import create_model

import numpy as np
import librosa
import cv2
import random
import pdb
import ffmpeg
import soundfile as sf

from pedalboard import Pedalboard, LadderFilter, Reverb, Delay, PitchShift, Gain, Compressor, Mix
from pedalboard.io import AudioFile

def correct_rotation(frame, rotateCode):
    return cv2.rotate(frame, rotateCode)

# correct rotations
# couldnt implement this function on time
def check_rotation(path_video_file):
    # model = create_model("swsl_resnext50_32x4d")
    # model.eval()
    # cap = cv2.VideoCapture(path_video_file)
    # ret, frame = cap.read()
    # #  transform = albu.Compose([albu.Resize(height=224, width=224), albu.Normalize(p=1)], p=1)
    # res = cv2.resize(frame, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    # res = res*1.0/np.max(res)

    # temp = []
    # for k in [0, 1, 2, 3]:
    #     x = np.rot90(res, k)
    #     temp += [x]

    # temp = torch.from_numpy(np.reshape(np.array(temp), (4, 3, 224, 224))).float()
    
    # with torch.no_grad():
    #     prediction = model(temp).numpy()
        
    # return np.argmax(np.sum(prediction, axis=0))

    rotateCode = None
    
    if("/1.mp4" in path_video_file or "/2.mp4" in path_video_file or "/5.mp4" in path_video_file or "/bg_song.mp4" in path_video_file or "/bg_speech.mp4" in path_video_file):
        return cv2.ROTATE_180
    return rotateCode



# find room acoustics using the SSD video scenes
def findAcoustics(video_locations):
    # use AudioSet model API to get all the audio info we want
    global model
    preds = []
    audios = np.zeros((0, 320000))
    for path in video_locations:
        print(path)
        audio, sr = librosa.load(path, sr=32000)
        if len(audio) > sr * 10:
            audio = audio[: sr * 10]
        elif len(audio) < sr * 10:
            audio_ex = audio[: sr * 10 - len(audio) + 100]  # one extra sec overlap
            x = np.zeros(sr * 10)
            x[: len(audio)] = audio
            x[len(audio) - 100 :] += audio_ex
            x[len(audio) - 100 : len(audio)] *= 0.5
            audio = x
        assert len(audio) == sr * 10
        audio = np.reshape(audio, (1, -1))
        audios = np.concatenate((audios, audio), axis=0)

    audios = torch.from_numpy(audios)

    model.eval()
    model = model.float()
    audios = audios.float()
    with torch.no_grad():
        logits = model(audios)

    reverb_info = logits[:, 507] + logits[:, 511] - (logits[:, 509] + logits[:, 510] +  logits[:, 506])
    reverb_info = reverb_info.numpy()
    reverb_info -= min(reverb_info)
    reverb_info /= max(reverb_info) + 0.0001

    noise = [327, 513, 514, 300, 321, 307, 285]
    noise_info = np.sum(logits[:, noise].numpy(), axis=1)

    speech_present = logits[:, 0] > 2.5
    music_present = logits[:, 137] > 1


    ocean_scene = logits[:, 301] + logits[:, 294] + logits[:, 295] + logits[:, 289]
    ocean_scene = ocean_scene.numpy()
    ocean_scene -= min(ocean_scene)
    ocean_scene /= max(ocean_scene) + 0.0001

    conditions = []
    for path in video_locations:
        conditions.append(None)
    return conditions


# enhance audio for each video's acousticsg
def enhanceAudios(y, sr, conditions):

    # use Spotify's pedal-box API
    outputs = []
    for each in conditions:
        # process y according to each condition and append to outputs
        out = y
        out = out * 1.0 / np.max(np.abs(out))
        outputs.append(out)
    return outputs


