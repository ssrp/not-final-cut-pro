import cv2
import numpy as np
import librosa
from src.helper import check_rotation, correct_rotation

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

        out =  audios[video_idx][idx[video_idx]%len(ys[video_idx])] * (1 - alpha) + ys[video_idx][i] * alpha 

        idx[video_idx] += 1
        if(idx[video_idx] >= len(audios[video_idx])):
            idx[video_idx] = 0
        out_wav.append(out)

    out_wav = np.array(out_wav)

    return out_wav
