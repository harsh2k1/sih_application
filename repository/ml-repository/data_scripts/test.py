# from flask_socketio import SocketIO
# import flask_socketio
# from flask import render_template
# # print(flask_socketio.__version__)
# from flask import Flask
# app=Flask(__name__)
# socketio = SocketIO(app)

# @app.route('/')
# def home():
#     return render_template('index.html')

'''
    1. in mbtipredict, implement logic 16PF folder me .ipynb hai
    2. implement own voice app
    3. look for a way to integrate webcam in app with opencv at backend
    4. push code
'''

'''
    MBTIPREDICT logic
    1. create a separate df, usme har ques ki predict_proba store kro ek column me
    2. ek sample array of answers bnao like har ques k lie 0-5 score
    3. then lets say 1 user se 10 ques puche hai to un 10 ques k scores (1d array) 0-5 k beech me, and unki predict-proba, dono pass krne h algorithm() me
'''

import speech_recognition as sr
import sys
sys.path.append('../')
from config import DATA_PATH
# initialize the recognizer
r = sr.Recognizer()
# open the file
filename = DATA_PATH+'output10.wav'
with sr.AudioFile(filename) as source:
    # listen for the data (load audio to memory)
    audio_data = r.record(source)
    # recognize (convert from speech to text)
    text = r.recognize_google(audio_data)
    print(text)