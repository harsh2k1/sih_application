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

import numpy as np
x = list(range(5))
y = [list(range(10)) for i in range(5)]
print(np.array(y).shape)
import pandas as pd
df = pd.DataFrame()
df['x'] = x
df['y'] = y
print(y)