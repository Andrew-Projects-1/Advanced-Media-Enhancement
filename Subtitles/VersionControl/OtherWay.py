import speech_recognition as sr
import moviepy.editor as mp
import time
import os

VIDEONAME = "ThisVideo.mp4"
SUBTITLEFILE = "subtitles.txt"

def main():
    # Load the video
    video = mp.VideoFileClip(VIDEONAME)

    # Initialize an empty string to hold the subtitles
    subtitles = ""

    # Process the video in 10 second chunks
    for i in range(0, int(video.duration), 10):
        # Calculate the end time of the segment
        end_time = min(i+12, video.duration)

        # Extract the audio from the video segment
        audio = video.subclip(i, end_time).audio

        # Write the audio to a file
        audio.write_audiofile(f"ThisVideo{i}.wav")

        sound = f"ThisVideo{i}.wav"

        r = sr.Recognizer()

        with sr.AudioFile(sound) as source:
            r.adjust_for_ambient_noise(source)

            print(f"Converting Audio File {i} To Text...")

            audio = r.listen(source)

            try:
                # Append the recognized text to the subtitles
                subtitles += r.recognize_google(audio) + "\n"
            except Exception as e:
                print(e)

        # Add a delay of 1 second between requests
        time.sleep(1)

        # Delete the .wav file
        os.remove(f"ThisVideo{i}.wav")

    # Write the subtitles to a file
    with open(SUBTITLEFILE, 'w') as f:
        f.write(subtitles)

    print(f"Converted Audio Is: \n {subtitles}")
    print(f"Subtitles written to {SUBTITLEFILE}")

if __name__ == "__main__":
    main()