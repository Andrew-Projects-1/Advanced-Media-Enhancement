import whisper
import re
import subprocess
import os

VIDEOFILE = "ThisVideo.mp4"
SUBTITLEFILE = "subtitle.srt"
OUTPUTAUDIOFILE = "output.wav"
OUTPUTTEXTFILE = 'transcription.txt'

if os.path.exists(OUTPUTTEXTFILE):
    os.remove(OUTPUTTEXTFILE)
if os.path.exists(OUTPUTAUDIOFILE):
    os.remove(OUTPUTAUDIOFILE)
if os.path.exists(SUBTITLEFILE):
    os.remove(SUBTITLEFILE)

# Extract the audio from the video file
subprocess.run(["ffmpeg", "-i", VIDEOFILE, "-ab", "160k", "-ac", "2", "-ar", "44100", "-vn", "output.wav"])

model = whisper.load_model("base")
result = model.transcribe("output.wav", fp16=False)
text = result["text"]
modified_text = re.sub(r'([.!?])', r'\1\n', text)

with open("transcription.txt", "w") as f:
    f.write(modified_text)

# Create srt file
with open(SUBTITLEFILE, "w") as f:
    lines = modified_text.split('\n')
    for i, line in enumerate(lines):
        if line:
            f.write(f"{i+1}\n")
            f.write(f"00:00:{i:02d},000 --> 00:00:{i+1:02d},000\n")
            f.write(line + "\n")
            f.write("\n")

# Add subtitles to the video
subprocess.run(["ffmpeg", "-i", VIDEOFILE, "-vf", f"subtitles={SUBTITLEFILE}:force_style='Fontsize=24'", "-c:a", "copy", "output_video.mp4"])

os.remove('output.wav')
os.remove(SUBTITLEFILE)