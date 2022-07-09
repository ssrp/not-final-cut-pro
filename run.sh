python main.py
ffmpeg -i ./tmp/mashup_vid.avi -i ./input_audio.mp3 -c:v copy -c:a aac ./tmp/output.mp4