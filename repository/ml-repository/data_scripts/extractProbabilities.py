import pandas as pd
import sys
sys.path.append('../')
from config import DATA_PATH,MODEL_PATH
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def prepare(df):
    text = list(df.posts)
    vectorizer = CountVectorizer()
    vectorizer.fit(text)
    vector = vectorizer.transform(text)
    X = vector.toarray()[-200:]  # taking last 200 questions
    
    return X

def prepare_df(df):
    X = df.iloc[-200:,:]
    X['questionID'] = list(range(len(X)))
    return X

def main():
    df = pd.read_csv(DATA_PATH+'mbti_1.csv')
    X = prepare(df)
    # df = pd.DataFrame()
    df = prepare_df(df)
    # print(X)
    filename = MODEL_PATH + 'finalized_model2.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    y_pred = loaded_model.predict_proba(X).tolist()
    print(np.array(y_pred).shape)
    # print(y_pred)
    df['predict_proba'] = y_pred
    print(df)
    df.to_csv(DATA_PATH+'final_questions.csv',index=None)

if __name__ == "__main__":
    main()