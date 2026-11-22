import os
import glob
from pathlib import Path

import whisper


# Find FFmpeg automatically
ffmpeg_files = glob.glob(
    r"C:\Users\ashut\AppData\Local\Microsoft\WinGet\Packages\**\ffmpeg.exe",
    recursive=True
)

if not ffmpeg_files:
    raise FileNotFoundError("FFmpeg executable not found!")

ffmpeg_path = Path(ffmpeg_files[0])
ffmpeg_dir = ffmpeg_path.parent

os.environ["PATH"] = (
    str(ffmpeg_dir) + os.pathsep + os.environ["PATH"]
)

print("FFmpeg found:", ffmpeg_path)


# Audio file
audio_file = Path("audio/sample.mp3")

if not audio_file.exists():
    raise FileNotFoundError(
        f"Audio file not found: {audio_file}"
    )


# Load Whisper model
print("Loading Whisper model...")
model = whisper.load_model("base")


# Transcribe audio
print("Transcribing audio...")
result = model.transcribe(
    str(audio_file),
    fp16=False
)


# Display transcript
transcript = result["text"]

print("\n--- Transcription ---")
print(transcript)


# Save transcript
with open("transcript.txt", "w", encoding="utf-8") as file:
    file.write(transcript)

print("\nTranscript saved successfully!")