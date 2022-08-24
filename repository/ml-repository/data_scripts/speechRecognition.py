from importlib.resources import path
import speech_recognition as sr 
import os 
from pydub import AudioSegment
from pydub.silence import split_on_silence

# # create a speech recognition object
# r = sr.Recognizer()

# # a function that splits the audio file into chunks
# # and applies speech recognition
# def get_large_audio_transcription(path):
#     """
#     Splitting the large audio file into chunks
#     and apply speech recognition on each of these chunks
#     """
#     # open the audio file using pydub
#     sound = AudioSegment.from_wav(path)  
#     # split audio sound where silence is 700 miliseconds or more and get chunks
#     chunks = split_on_silence(sound,
#         # experiment with this value for your target audio file
#         min_silence_len = 500,
#         # adjust this per requirement
#         silence_thresh = sound.dBFS-14,
#         # keep the silence for 1 second, adjustable as well
#         keep_silence=500,
#     )
#     folder_name = "audio-chunks"
#     # create a directory to store the audio chunks
#     if not os.path.isdir(folder_name):
#         os.mkdir(folder_name)
#     whole_text = ""
#     # process each chunk 
#     for i, audio_chunk in enumerate(chunks, start=1):
#         # export audio chunk and save it in
#         # the `folder_name` directory.
#         chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
#         audio_chunk.export(chunk_filename, format="wav")
#         # recognize the chunk
#         with sr.AudioFile(chunk_filename) as source:
#             audio_listened = r.record(source)
#             # try converting it to text
#             try:
#                 text = r.recognize_google(audio_listened)
#             except sr.UnknownValueError as e:
#                 print("Error:", str(e))
#             else:
#                 text = f"{text.capitalize()}. "
#                 print(chunk_filename, ":", text)
#                 whole_text += text
#     # return the text for all chunks detected
#     return whole_text


# # path = "7601-291468-0006.wav"
# import sys
# sys.path.append('../')
# from config import DATA_PATH
# path = DATA_PATH+'sample_track.wav'
# print("\nFull text:", get_large_audio_transcription(path))
# os.remove('audio-chunks')

# Python program to translate
# speech to text and text to speech


# Python program to translate
# speech to text and text to speech


import speech_recognition as sr
import pyttsx3

r = sr.Recognizer()

# Python program to translate
# speech to text and text to speech


import speech_recognition as sr
import pyttsx3

# Initialize the recognizer
r = sr.Recognizer()

# Function to convert text to
# speech
def SpeakText(command):
	
	# Initialize the engine
	engine = pyttsx3.init()
	engine.say(command)
	engine.runAndWait()
	
	
# Loop infinitely for user to
# speak

# while(True):
# 	try:
#         with sr.Microphone
#     finally:
#         break
