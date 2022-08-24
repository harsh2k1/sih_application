import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import pickle
sys.path.append("../")
from config import DATA_PATH
from config import MODEL_PATH

def read_csv(path):
    return pd.read_csv(path)

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

def prepare(df):
    text = list(df.posts)
    vectorizer = CountVectorizer()
    vectorizer.fit(text)
    vector = vectorizer.transform(text)
    # X = vector.toarray()[:1000]
    # y = np.array(df.type).reshape(-1,1)[:1000]
    X = vector.toarray()[-1:]
    # y = np.array(df.type).reshape(-1,1)[-1000:]
    # X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30, random_state=42)
    # print(X_train.shape,X_test.shape)
    # return X_train,X_test,y_train,y_test
    return X

def evaluate(y_test,y_pred):
    accuracy = accuracy_score(y_test,y_pred)
    return accuracy

# def train():
#     df = read_csv(DATA_PATH+'mbti_1.csv')
#     X,y,X_train,X_test,y_train,y_test = prepare(df)
#     filename = MODEL_PATH + 'finalized_model.sav'
#     loaded_model = pickle.load(open(filename, 'rb'))
#     y_pred = loaded_model.predict(X_test)
#     accuracy = evaluate(y_test,y_pred)
#     print(accuracy)

def predict():
    encodings = {8: 'INFJ', 3: 'ENTP', 11: 'INTP', 10: 'INTJ', 2: 'ENTJ', 0: 'ENFJ', 9: 'INFP', 1: 'ENFP', 13: 'ISFP', 15: 'ISTP', 12: 'ISFJ', 14: 'ISTJ', 7: 'ESTP', 5: 'ESFP', 6: 'ESTJ', 4: 'ESFJ'}
    df = read_csv(DATA_PATH+'mbti_1.csv')
    print(df.loc[len(df)-1])
    X = prepare(df)
    print(X)
    filename = MODEL_PATH + 'finalized_model2.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    y_pred = loaded_model.predict_proba(X)
    final_predicted_output_array = algorithm(X[0],y_pred)
    final_predicted_output = encodings.get(np.argmax(final_predicted_output_array))
    print(final_predicted_output)

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


if __name__ == "__main__":
    # predict()
    test()
