# from crypt import methods
# from crypt import methods
import time
from flask import Flask
from flask import render_template, request
import sys
from flask import Flask, render_template, Response
import cv2
import pandas as pd
import json
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
# sys.path.append('../ml-repository/')
# from config import MODEL_PATH

MODEL_PATH = '../ml-repository/model_container/'
app = Flask(__name__)
t_end = time.time() + 120 # 120 secs

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

@app.route('/dashboard-inner')
def dashboard_inner():
    return render_template('dashboard-inner.html')

@app.route('/mentorship-book')
def mentorship_book():
    return render_template('mentorship-book.html')

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

def algorithm(X, probabilities):
    final = []
    probs = [[X[i]*probabilities[i][j] for j in range(len(probabilities[i]))] for i in range(len(list(X)))]
    print(probs[0])
    for i in range(len(probabilities[0])): # range(16)
        sum = 0
        for j in range(len(probs)):
            sum += probs[j][i] # 00 10 20
        final.append(sum)
    return final

def test():
    from random import uniform
    encodings = {8: 'INFJ', 3: 'ENTP', 11: 'INTP', 10: 'INTJ', 2: 'ENTJ', 0: 'ENFJ', 9: 'INFP', 1: 'ENFP', 13: 'ISFP', 15: 'ISTP', 12: 'ISFJ', 14: 'ISTJ', 7: 'ESTP', 5: 'ESFP', 6: 'ESTJ', 4: 'ESFJ'}
    X = [1, 4, 2, 3, 3,5,0,2,0,4]  # output from user 0-5
    probabilities = [[uniform(0,1) for j in range(16)] for i in range(len(X))]    # this is len(X)x16 shaped array, isme har single ques k lie 16 predict_proba hai.. meaning jb df['predict_proba'] ban jaega to mje un specific ques k lie list of predict_proba bnani h eg probabilities.append(df[df['question'] == ques-asked-from-user]['predict_proba'])
    print(np.array(probabilities).shape)
    final = algorithm(X,probabilities)
    print(final)
    final_predicted_output = encodings.get(np.argmax(final))
    print(final_predicted_output)
    return final_predicted_output

@app.route('/report')
def final_results():
    li1 = ['INTJ','INTP','ENTJ','ENTP','INFJ','INFP','ENFJ','ENFP','ISTJ','ISFJ','ESTJ','ESFJ','ISTP','ISFP','ESTP','ESFP']
    li2 = ['Architect','Logician','Commander','Debater','Advocate','Mediator','Protagonist','Campaigner','Logistician','Defender','Executives','Consul','Virtuoso','Adventurer','Entepreneur','Entertainer']
    li2 = [ele.replace(' ','_').lower() for ele in li2]
    final_predicted_output = test()
    x = dict(zip(li1,li2))
    path = x.get(final_predicted_output) + '.html'
    print(path)
    return render_template('adventurer.html')

@app.route('/mentorship')
def mentorship():
    return render_template('mentorship.html')

@app.route('/professional')
def professional():
    return render_template('professional.html')

@app.route('/student')
def student():
    return render_template('student.html')

@app.route('/slot-book')
def slot_book():
    return render_template('slot_book.html')

@app.route('/test')
def test():
    return render_template('test.html')

@app.route('/filter', methods=['POST'])
def filter():
    data = pd.read_csv('collegeData.csv', encoding='utf8')
    data = data[(data['city']==request.form["Country"]) & (data['courseName']==request.form["Region"])]
    data = data.head(50)
    data = data.to_dict(orient='records')
    response = json.dumps(data, indent=2)
    return response


# @app.route('/test', methods=['POST'])
# def test():
#     uname=request.form['uname']  
#     passwrd=request.form['pass']  
#     if uname=="harsh" and passwrd=="123":  
#         return "Welcome %s" %uname  
#     # return 'Hello'


if __name__ == "__main__":
    app.run(debug=True)