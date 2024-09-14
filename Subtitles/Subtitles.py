# Andrew Moskowitz
##
# This program processes a video file by extracting its audio, generating subtitles using a speech-to-text model, 
# and embedding the subtitles back into the video. First, it checks if the input video file exists and handles 
# specific video formats (like .avi and .mov) by converting them to .mp4. It extracts the audio from the video 
# using ffmpeg, then uses the Whisper model to transcribe the audio and generate subtitles with precise timing. 
# The program writes these subtitles to an .srt file and adds them to the video, using ffmpeg to create a new 
# video with the subtitles overlaid. Afterward, it cleans up any temporary files created during the process.
##

import whisper
import subprocess
import os
from datetime import timedelta

# Input file
INPUT_VIDEO_FILE = "Test.mp4"

# Output file
OUTPUT_VIDEO_FILE = f"{INPUT_VIDEO_FILE.rsplit('.', 1)[0]}_Subtitles.{INPUT_VIDEO_FILE.rsplit('.', 1)[-1]}"

# Subtitle file
SUBTITLE_FILE = "subtitle.srt"

# Audio file
AUDIO_FILE = "output.wav"

# Check if input file exists
if not os.path.exists(INPUT_VIDEO_FILE):
    print("Input file does not exist.")
    exit()

# Remove output file if it already exists
if os.path.exists(OUTPUT_VIDEO_FILE):
    os.remove(OUTPUT_VIDEO_FILE)

# Check if the input file is .avi or .mov
if INPUT_VIDEO_FILE.endswith('.avi') or INPUT_VIDEO_FILE.endswith('.mov'):
    # Convert .avi or .mov file to mp4
    subprocess.run(["ffmpeg", "-i", INPUT_VIDEO_FILE, "-vcodec", "h264", "-acodec", "aac", "-strict", "2", "-preset", "slow", "temp.mp4"])
    INPUT_VIDEO_FILE = "temp.mp4"
    OUTPUT_VIDEO_FILE = f"{INPUT_VIDEO_FILE.rsplit('.', 1)[0]}_FER_Video.{INPUT_VIDEO_FILE.rsplit('.', 1)[-1]}"

# Function to convert a timedelta object into the timestamp format used in .srt subtitle files
def format_timedelta(td):
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

# Extract the audio from the video file
subprocess.run(["ffmpeg", "-i", INPUT_VIDEO_FILE, "-ab", "160k", "-ac", "2", "-ar", "44100", "-vn", AUDIO_FILE])

# Load Whisper model and transcribe audio with timing information
model = whisper.load_model("base")
result = model.transcribe(AUDIO_FILE, fp16=False, temperature=0)

# Use the segments provided by Whisper for accurate timing
segments = result["segments"]

# Create srt file with accurate timings
with open(SUBTITLE_FILE, "w") as f:
    for i, segment in enumerate(segments):
        start_time = timedelta(seconds=segment['start'])
        end_time = timedelta(seconds=segment['end'])
        words = segment['text']
        f.write(f"{i+1}\n")
        f.write(f"{format_timedelta(start_time)} --> {format_timedelta(end_time)}\n")
        f.write(words + "\n\n")

# Add subtitles to the video
subprocess.run(["ffmpeg", "-i", INPUT_VIDEO_FILE, "-vf", f"subtitles={SUBTITLE_FILE}:force_style='Fontsize=24'", "-c:a", "copy", OUTPUT_VIDEO_FILE])

# Remove temporary files
for file in [AUDIO_FILE, SUBTITLE_FILE, "temp.mp4"]:
    if os.path.exists(file):
        os.remove(file)
