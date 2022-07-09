import numpy as np
import librosa
import cv2
import random
import moviepy.editor as mp
import pdb

random.seed(10)
    
# find beats in the audio
def findBeats(y, sr):
    # use Librosa API, keep it sparse, around min 5seconds for a video
    
    out = np.zeros(len(y))
    val = len(y)/5.0
    for i in range(5):
        out[int(i*val)] = 1
    return out

# find room acoustics using the SSD video scenes
def findAcoustics(videos):
    # use SSD object detection APIs
    conditions = []
    for i in range(len(videos)):
        conditions.append(None)
    return conditions

# enhance audio for each video's acoustics
def enhanceAudios(y, sr, conditions):
    # use Spotify's pedal-box API
    outputs = []
    for each in conditions:
        # process y according to each condition and append to outputs
        outputs.append(y)
    return outputs

# create the mashup video using the video
def render_video(video_locations, video_assignments, sr):
    # it's all math

    video_assignments = video_assignments # convert sampling rate to frame rate

    frames = []
    caps = []
    for path in video_locations:
        caps.append(cv2.VideoCapture(path))

    frame_width = int(caps[0].get(3))
    frame_height = int(caps[0].get(4))

    out = cv2.VideoWriter('./tmp/mashup_vid.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))  

    count = 0 
    for i in range(int(len(video_assignments)*30.0/sr)):
        assignment_idx = int(i*sr/30.0)
        cap = caps[video_assignments[assignment_idx]]
        if cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            out.write(frame)
            frames.append(frame)
            count += 1
    print("num of frames: " + str(count))
    print("duration: " + str(count/30.0))

    return True

# create the mashup audio
def render_audio(video_locations, y, sr, video_assignments):
    # it's all math

    return y, sr
    
def main():

    # video locations in a list 
    video_locations = [
        "./videos/1.mp4",
        "./videos/2.mp4",
        "./videos/3.mp4",
        "./videos/4.mp4",
        "./videos/5.mp4",
        "./videos/6.mp4",
        # "./videos/7.mp4", # landscape video
        "./videos/bg_noise.mp4",
        "./videos/bg_song.mp4",
        "./videos/bg_speech.mp4",
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
    final_audio, sr = render_audio(audios, sr, video_assignments)

    return None

if __name__ == "__main__":
    main()
