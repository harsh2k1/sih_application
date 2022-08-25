# from crypt import methods
# from crypt import methods
import time
from flask import Flask
from flask import render_template, request
import sys
from flask import Flask, render_template, Response
import cv2
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras.preprocessing.image import load_img, img_to_array 
# from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import  load_model
import matplotlib.pyplot as plt
import numpy as np
import sys
import speech_recognition as sr 
import os 
from pydub import AudioSegment
from pydub.silence import split_on_silence
import speech_recognition as sr
import pyttsx3
import nltk
nltk.download('vader_lexicon')
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import spacy

# sys.path.append('../ml-repository/')
# from config import MODEL_PATH

MODEL_PATH = '../ml-repository/model_container/'
app = Flask(__name__)
t_end = time.time() + 120 # 120 secs
# r = sr.Recognizer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login')
def user_login():
    return render_template('login.html')

@app.route('/register')
def user_register():
    return render_template('register.html')

@app.route('/dashboard')
def user_dashboard():
    return render_template('Dashboard-index.html')

@app.route('/personality')
def personality():
    return render_template('personality.html')

@app.route('/aptitude')
def aptitude():
    return render_template('aptitude_ass.html')

@app.route('/user-profile')
def user_profile():
    return render_template('users-profile.html')

@app.route('/pages-faq')
def pages_faq():
    return render_template('pages-faq.html')

@app.route('/pages-contact')
def pages_contact():
    return render_template('pages-contact.html')

def SpeakText(command):
	
	# Initialize the engine
	engine = pyttsx3.init()
	engine.say(command)
	engine.runAndWait()

def clean_data(text):
    r = sr.Recognizer()

    lemmatizer = WordNetLemmatizer()
    stopwords_english = stopwords.words('english')
    text_clean = []
    text_tokens = word_tokenize(text)
    
    for word in text_tokens:
        if (word not in stopwords_english and # remove stopwords              
                word not in string.punctuation): # remove punctuation            
            stem_word = lemmatizer.lemmatize(word) # stemming word
            text_clean.append(stem_word)
    
    list_to_str = ' '.join([str(ele) for ele in text_clean])
    return list_to_str.lower()

def get_vader_sentiment(review): 
    sia = SentimentIntensityAnalyzer()
    sia = SentimentIntensityAnalyzer()
    analysis = sia.polarity_scores(review)
    return analysis['compound']

def gen_frames():  
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            # load model
            model = load_model(MODEL_PATH+"emotionRecognitionModel.h5")


            # face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            face_haar_cascade = cv2.CascadeClassifier('../ml-repository/data_scripts/haarcascade_frontalface_default.xml')


            cap = cv2.VideoCapture(0)
            seconds =0
            # while seconds<120:  # for 120 secs
            while time.time() < t_end:
                ret, test_img = cap.read()  # captures frame and returns boolean value and captured image
                if not ret:
                    continue
                gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

                faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

                for (x, y, w, h) in faces_detected:
                    cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
                    roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
                    roi_gray = cv2.resize(roi_gray, (224, 224))
                    img_pixels = image.img_to_array(roi_gray)
                    # img_pixels = img_to_array(roi_gray)
                    img_pixels = np.expand_dims(img_pixels, axis=0)
                    img_pixels /= 255

                    predictions = model.predict(img_pixels)

                    # find max indexed array
                    max_index = np.argmax(predictions[0])
                    print(predictions)

                    emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
                    # emotions = ('angry','disgust','fear','confident','neutral','sad','surprise')
                    predicted_emotion = emotions[max_index]

                    # cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    if seconds in range(20,40):
                        cv2.putText(test_img, 'happy', (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    seconds += 1

                resized_img = cv2.resize(test_img, (1000, 700))
                # cv2.imshow('Facial emotion analysis ', resized_img)

                if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
                    break
                ret, buffer = cv2.imencode('.jpg', resized_img)  #test_img
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


            # cap.release()
            # cv2.destroyAllWindows

            # ret, buffer = cv2.imencode('.jpg', frame)
            # frame = buffer.tobytes()
            # yield (b'--frame\r\n'
            #        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video-assessment-test')
def video_assessment_test():
    return render_template('video_test.html')

@app.route('/test', methods=['POST'])
def test():
    uname=request.form['uname']  
    passwrd=request.form['pass']  
    if uname=="harsh" and passwrd=="123":  
        return "Welcome %s" %uname  
    # return 'Hello'


if __name__ == "__main__":
    app.run(debug=True)