python main.py
# best quality video, more space
# ffmpeg -i ./tmp/mashup_vid.avi -i ./output_audio.wav -c:v copy -c:a copy ./tmp/output.avi
# ok quality, space efficient
ffmpeg -i ./tmp/mashup_vid.avi -i ./tmp/output_audio.wav -b 1600k ./output/output.mp4
