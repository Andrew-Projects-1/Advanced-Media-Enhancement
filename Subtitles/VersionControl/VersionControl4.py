import whisper
import subprocess
import os
from datetime import timedelta

VIDEOFILE = "ThisVideo.mp4"
SUBTITLEFILE = "subtitle.srt"
OUTPUTAUDIOFILE = "output.wav"
OUTPUTVIDEO = "output_video.mp4"

def format_timedelta(td):
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

if os.path.exists(SUBTITLEFILE):
    os.remove(SUBTITLEFILE)
if os.path.exists(OUTPUTAUDIOFILE):
    os.remove(OUTPUTAUDIOFILE)
if os.path.exists(OUTPUTVIDEO):
    os.remove(OUTPUTVIDEO)

# Extract the audio from the video file
subprocess.run(["ffmpeg", "-i", VIDEOFILE, "-ab", "160k", "-ac", "2", "-ar", "44100", "-vn", OUTPUTAUDIOFILE])

# Load Whisper model and transcribe audio with timing information
model = whisper.load_model("base")
result = model.transcribe(OUTPUTAUDIOFILE, fp16=False, temperature=0)

# Use the segments provided by Whisper for accurate timing
segments = result["segments"]

# Create srt file with accurate timings
with open(SUBTITLEFILE, "w") as f:
    for i, segment in enumerate(segments):
        start_time = timedelta(seconds=segment['start'])
        end_time = timedelta(seconds=segment['end'])
        words = segment['text']
        f.write(f"{i+1}\n")
        f.write(f"{format_timedelta(start_time)} --> {format_timedelta(end_time)}\n")
        f.write(words + "\n\n")

# Add subtitles to the video
subprocess.run(["ffmpeg", "-i", VIDEOFILE, "-vf", f"subtitles={SUBTITLEFILE}:force_style='Fontsize=24'", "-c:a", "copy", OUTPUTVIDEO])

os.remove(OUTPUTAUDIOFILE)
os.remove(SUBTITLEFILE)
