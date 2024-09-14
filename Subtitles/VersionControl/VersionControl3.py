import whisper
import re
import subprocess
import os

VIDEOFILE = "ThisVideo.mp4"
SUBTITLEFILE = "subtitle.srt"
OUTPUTAUDIOFILE = "output.wav"
OUTPUTTEXTFILE = "transcription.txt"
OUTPUTVIDEO = "output_video.mp4"

if os.path.exists(SUBTITLEFILE):
    os.remove(SUBTITLEFILE)
if os.path.exists(OUTPUTAUDIOFILE):
    os.remove(OUTPUTAUDIOFILE)
if os.path.exists(OUTPUTTEXTFILE):
    os.remove(OUTPUTTEXTFILE)
if os.path.exists(OUTPUTVIDEO):
    os.remove(OUTPUTVIDEO)

# Extract the audio from the video file
subprocess.run(["ffmpeg", "-i", VIDEOFILE, "-ab", "160k", "-ac", "2", "-ar", "44100", "-vn", OUTPUTAUDIOFILE])

model = whisper.load_model("base")
result = model.transcribe(OUTPUTAUDIOFILE, fp16=False)
text = result["text"]
modified_text = re.sub(r'([.!?])', r'\1\n', text)

with open(OUTPUTTEXTFILE, "w") as f:
    f.write(modified_text)

# Calculate the time that each word is spoken
words = modified_text.split(' ')
times = []
for i, word in enumerate(words):
    start_time = i * 0.2  # assume each word is spoken in 1 second
    end_time = (i + 1) * 0.2
    times.append((start_time, end_time))

# Create srt file
with open(SUBTITLEFILE, "w") as f:
    for i, word in enumerate(words):
        if word:
            f.write(f"{i+1}\n")
            f.write(f"00:00:0{int(times[i][0]):02d}.000 --> 00:00:0{int(times[i][1]):02d}.000\n")
            f.write(word + "\n")
            f.write("\n")

# Add subtitles to the video
subprocess.run(["ffmpeg", "-i", VIDEOFILE, "-vf", f"subtitles={SUBTITLEFILE}:force_style='Fontsize=24'", "-c:a", "copy", OUTPUTVIDEO])

os.remove(OUTPUTAUDIOFILE)
os.remove(SUBTITLEFILE)