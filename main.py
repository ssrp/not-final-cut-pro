# AudioSet Model : https://github.com/kkoutini/passt
from hear21passt.base import get_basic_model, get_model_passt
import torch

# model = get_basic_model(mode="logits")
import numpy as np
import librosa
import cv2
import random
import pdb
import ffmpeg
import soundfile as sf
from src.beat_detection import find_onesets
from check_orientation.pre_trained_models import create_model

random.seed(10)


# correct rotationa
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


def correct_rotation(frame, rotateCode):
    return cv2.rotate(frame, rotateCode)


# find beats in the audio
def findBeats(y, sr, splits=5):
    # use Librosa API, keep it sparse, around min 5seconds for a video

    out, tempo = find_onesets(y, sample_rate=sr)
    out = [int(i[0] * sr) for i in np.array_split(out, splits + 1)]
    beats = np.zeros(len(y))
    for each in out[1:]:
        beats[each] = 1
    return beats


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

    # model.eval()
    # model = model.float()
    # audios = audios.float()
    # with torch.no_grad():
    #     logits = model(audios)

    # CLASSES WE NEED TO CHECK:

    # Reverb info
    # 506 - Inside, small room
    # 507 - Inside, large room or hall
    # 508 - Inside, public space
    # 509 - Outside, urban or manmade
    # 510 - Outside, rural or natural
    # 511 - Reverberation

    # 512 - Echo

    # noise
    # 327 - Traffic noise, roadway noise
    # 513 - Noise
    # 514 - Environmental noise

    # specific cases
    # 0 - Speech > 2.5
    # 137 - Music > 1

    # ocean/water
    # 301 - water vehicle
    # 294 - ocean
    # 295 - Waves, surf
    # 289 - Rain

    # travel
    # 300 - vehicle
    # 321 - bus
    # 307 - car

    # microphone wind
    # 285 - Wind noise (microphone)

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


# create the mashup video using the video
def render_video(video_locations, video_assignments, sr):
    # it's all math

    video_assignments = video_assignments  # convert sampling rate to frame rate

    frames = []
    caps = []
    rotateCodes = []
    for path in video_locations:
        # print(path)
        caps.append(cv2.VideoCapture(path))
        rotateCodes.append(check_rotation(path))
        # print(check_rotation(path))
    frame_width = int(caps[0].get(3))
    frame_height = int(caps[0].get(4))

    out = cv2.VideoWriter(
        "./tmp/mashup_vid.avi", cv2.VideoWriter_fourcc("M", "J", "P", "G"), 30, (frame_width, frame_height)
    )

    count = 0
    for i in range(int(len(video_assignments) * 30.0 / sr)):
        assignment_idx = int(i * sr / 30.0)
        vid_idx = video_assignments[assignment_idx] % len(video_locations)
        cap = caps[vid_idx]
        rotateCode = rotateCodes[vid_idx]
        if cap.isOpened():
            
            ret, frame = cap.read()
            if not ret:
                caps[vid_idx].set(2,0);
                ret, frame = caps[vid_idx].read()
                if not ret:
                    break
            
            if rotateCode is not None:
                frame = correct_rotation(frame, rotateCode)

            out.write(frame)
            frames.append(frame)
            count += 1
    print("num of frames: " + str(count))
    print("duration: " + str(count / 30.0))

    return True


# create the mashup audio
def render_audio(video_locations, ys, sr, video_assignments, alpha):
    # it's all math

    audios = []
    for path in video_locations:
        audio, sr = librosa.load(path)
        audio = audio * 1.0 / np.max(np.abs(audio))
        audios.append(audio)

    idx = np.zeros(len(video_locations), dtype=int)
    out_wav = []
    for i in range(len(video_assignments)):
        video_idx = video_assignments[i] % len(video_locations)

        out =  audios[video_idx][idx[video_idx]%len(ys[video_idx])] # * (1 - alpha) + ys[video_idx][i] * alpha 

        idx[video_idx] += 1
        if(idx[video_idx] >= len(audios[video_idx])):
            idx[video_idx] = 0
        out_wav.append(out)

    out_wav = np.array(out_wav)

    sf.write("output_audio.wav", out_wav, sr, subtype="PCM_24")

    return None


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
    y, sr = librosa.load("./input_audio.mp3")

    # find beats for the audio in a list
    
    beats = findBeats(y, sr, splits=len(video_locations))
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
    final_audio_flag = render_audio(video_locations, audios, sr, video_assignments, alpha)

    print(video_locations)

    return None


if __name__ == "__main__":
    main()
