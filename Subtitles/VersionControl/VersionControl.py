import whisper
import re
import subprocess
import os

VIDEOFILE = "ThisVideo.mp4"
SUBTITLEFILE = "subtitle.srt"
OUTPUTAUDIOFILE = "output.wav"
OUTPUTTEXTFILE = "transcription.txt"
OUTPUTVIDEO = "output_video.mp4"

if os.path.exists(OUTPUTTEXTFILE):
    os.remove(OUTPUTTEXTFILE)
if os.path.exists(OUTPUTAUDIOFILE):
    os.remove(OUTPUTAUDIOFILE)
if os.path.exists(SUBTITLEFILE):
    os.remove(SUBTITLEFILE)

# Extract the audio from the video file
subprocess.run(["ffmpeg", "-i", VIDEOFILE, "-ab", "160k", "-ac", "2", "-ar", "44100", "-vn", OUTPUTAUDIOFILE])

model = whisper.load_model("base")
result = model.transcribe("output.wav", fp16=False)
text = result["text"]
modified_text = re.sub(r'([.!?])', r'\1\n', text)

with open(OUTPUTTEXTFILE, "w") as f:
    f.write(modified_text)

# Create srt file
with open(SUBTITLEFILE, "w") as f:
    f.write("1\n00:00:00,000 --> 99:99:99,999\n" + modified_text)

# Add subtitles to the video
subprocess.run(["ffmpeg", "-i", VIDEOFILE, "-vf", f"subtitles={SUBTITLEFILE}:force_style='Fontsize=24'", "-c:a", "copy", OUTPUTVIDEO])

os.remove(OUTPUTAUDIOFILE)
os.remove(SUBTITLEFILE)