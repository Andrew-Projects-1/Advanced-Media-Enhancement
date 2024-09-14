import speech_recognition as sr
import moviepy.editor as mp

    # Load the video
video = mp.VideoFileClip(r"ThisVideo.mp4")

# Extract the audio from the video
audio = video.audio

# Write the audio to a file
audio.write_audiofile(r"ThisVideo.wav")