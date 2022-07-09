# AudioSet Model : https://github.com/kkoutini/passt
from hear21passt.base import get_basic_model,get_model_passt
import torch
model = get_basic_model(mode="logits")
import numpy as np
import librosa
import cv2
import random
import ffmpeg
import pdb
import soundfile as sf

random.seed(10)

# correct rotation 
def check_rotation(path_video_file):
    # this returns meta-data of the video file in form of a dictionary
    meta_dict = ffmpeg.probe(path_video_file)

    # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
    # we are looking for
    rotateCode = None
    if 'rotate' not in meta_dict['streams'][0]['tags']:
        return None

    if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
        rotateCode = cv2.ROTATE_90_CLOCKWISE
    elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
        rotateCode = cv2.ROTATE_180
    elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
        rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE

    return rotateCode
def correct_rotation(frame, rotateCode):  
    return cv2.rotate(frame, rotateCode)


# find beats in the audio
def findBeats(y, sr):
    # use Librosa API, keep it sparse, around min 5seconds for a video
    
    out = np.zeros(len(y))
    out[int(6.117*22050)] = 1
    out[int(10.104*22050)] = 1
    out[int(18.115*22050)] = 1
    out[int(26.088*22050)] = 1
    out[int(34.100*22050)] = 1
    out[int(45.100*22050)] = 1
    return out

# find room acoustics using the SSD video scenes
def findAcoustics(video_locations):
    # use AudioSet model API to get all the audio info we want
    global model
    preds = []
    audios = np.zeros((0, 320000))
    for path in video_locations:
        print(path)
        audio, sr = librosa.load(path, sr=32000)
        if(len(audio) > sr*10):
            audio = audio[:sr*10]
        elif(len(audio) < sr*10):
            audio_ex = audio[:sr*10 - len(audio)+100] # one extra sec overlap
            x = np.zeros(sr*10)
            x[:len(audio)] = audio
            x[len(audio)-100:] += audio_ex
            x[len(audio)-100:len(audio)] *= 0.5
            audio = x
        assert len(audio) == sr*10
        audio = np.reshape(audio, (1, -1))
        audios = np.concatenate((audios, audio), axis= 0)

    audios = torch.from_numpy(audios)


    model.eval()
    model = model.float()
    audios = audios.float()
    with torch.no_grad():
        logits=model(audios) 

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
        out = out*1.0/np.max(np.abs(out))
        outputs.append(out)
    return outputs

# create the mashup video using the video
def render_video(video_locations, video_assignments, sr):
    # it's all math

    video_assignments = video_assignments # convert sampling rate to frame rate

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

    out = cv2.VideoWriter('./tmp/mashup_vid.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))  

    count = 0 
    for i in range(int(len(video_assignments)*30.0/sr)):
        assignment_idx = int(i*sr/30.0)
        cap = caps[video_assignments[assignment_idx]%len(caps)]
        rotateCode = rotateCodes[video_assignments[assignment_idx]%len(caps)]
        if cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # if rotateCode is not None:
            #     frame = correct_rotation(frame, rotateCode)

            out.write(frame)
            frames.append(frame)
            count += 1
    print("num of frames: " + str(count))
    print("duration: " + str(count/30.0))

    return True

# create the mashup audio
def render_audio(video_locations, ys, sr, video_assignments, alpha):
    # it's all math

    
    audios = []
    for path in video_locations:
        audio, sr = librosa.load(path)
        audio = audio*1.0/np.max(np.abs(audio))
        audios.append(audio)

    idx = np.zeros(len(video_locations), dtype = int)
    out_wav = []
    for i in range(len(video_assignments)):
        video_idx = video_assignments[i]
        
        try: out = ys[video_idx%len(video_locations)][i] * alpha + audios[video_idx%len(video_locations)][idx[video_idx]]*(1-alpha)
        except: 
            print("err")
            print(video_locations[video_idx]%len(video_locations))
            print(i)
            print(len(audios[video_idx]))
            print(idx[video_idx])
        idx[video_idx] += 1
        out_wav.append(out)

    out_wav = np.array(out_wav)

    sf.write('output_audio.wav', out_wav, sr, subtype='PCM_24')

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
        "./videos/4.mp4",
        # "./videos/7.mp4", # landscape video
        # "./videos/bg_noise.mp4",
        # "./videos/bg_song.mp4",
        # "./videos/bg_speech.mp4",
    ]
    random.shuffle(video_locations)

    # load input audio 
    y, sr = librosa.load("./input_audio.mp3")

    # find beats for the audio in a list
    beats = findBeats(y, sr)

    # sampling rate of audio
    # [0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 1 0 0]
    # [0 0 0 0 0 0 1 1 1 1 1 1 2 2 2 3 3 3]
    video_assignments = []
    idx = 0
    for i in range(len(beats)):
        video_assignments.append(idx)
        if(beats[i] == 1):
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
