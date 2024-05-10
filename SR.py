import subprocess

def install_package(package):
    subprocess.check_call(["pip", "install", package])

def check_and_install(package):
    try:
        __import__(package)
    except ImportError:
        print(f"Package {package} is not installed. Installing...")
        install_package(package)

required_packages = [
    "numpy",
    "pyttsx3",
    "google.generativeai",
    "transformers",
    "sounddevice",
    "soundfile"
]

for package in required_packages:
    check_and_install(package)

import numpy as np
import pyttsx3
import google.generativeai as genai
from transformers import pipeline
import sounddevice as sd
import re

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")
genai.configure(api_key='AIzaSyA3qU83mCzg23ogwvPHx8mfFgCUNJtXvmk')  # Replace 'YOUR_API_KEY' with your actual API key
model = genai.GenerativeModel('gemini-pro')

previous_text = ""  # Variable to store the previously transcribed text
chat = model.start_chat()
chat.send_message('answer in short')

def transcribe(new_chunk):
    global previous_text
    sr, y = new_chunk
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    # Transcribe the latest chunk of audio data
    text = transcriber({"sampling_rate": sr, "raw": y})["text"]
    print("Transcribed Text:", text)

    # Only generate content if the text has changed and stop is not requested
    if text != previous_text:
        response = chat.send_message(text)
        generated_text = response.text.replace("*", "")
        generated_text = remove_urls(generated_text)
        print("Generated Text:", generated_text)
        return generated_text

def record_audio(duration=10, fs=16000):
    print("Recording...")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    return fs, audio_data[:, 0]  # Select only the first channel

def play_audio(audio_data, fs=16000):
    print("Playing...")
    sd.play(audio_data, samplerate=fs)
    sd.wait()

def remove_urls(text):
    # Regular expression to match URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    # Remove URLs from the text
    return url_pattern.sub('', text)

def main():
    engine = pyttsx3.init()
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate - 25)

    while True:
        fs, audio_data = record_audio()
        text = transcribe((fs, audio_data))
        if text:
            # Convert text to speech and play it
            engine.say(text)
            engine.runAndWait()

if __name__ == "__main__":
    main()
