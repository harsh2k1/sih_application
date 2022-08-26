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
sys.path.append('../')
from config import MODEL_PATH

import pandas as pd
sys.path.append('../')
from config import DATA_PATH
from flask import Flask
from flask import render_template, request
app = Flask(__name__)

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
            face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


            cap = cv2.VideoCapture(0)
            seconds =0
            while seconds<120:
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

@app.route('/')
def home():
    import test2
    # test2.emotion_recognizer()
    return render_template('round2.html')

# import numpy as np
from flask import Flask, request, jsonify, render_template
# import pickle

# app = Flask(__name__)
# model = pickle.load(open('model.pkl', 'rb'))

@app.route('/test')
def test():
    return render_template('test.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    print(final_features)
    # prediction = model.predict(final_features)

    # output = round(prediction[0], 2)
    output = 1000

    return render_template('test.html', prediction_text='Employee Salary should be $ {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    # prediction = model.predict([np.array(list(data.values()))])

    # output = prediction[0]
    output = 1000
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)
