import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
sys.path.append("../")
from config import DATA_PATH

def read_csv(path):
    return pd.read_csv(path)

def prepare(df):
    text = list(df.posts)
    vectorizer = CountVectorizer()
    vectorizer.fit(text)
    vector = vectorizer.transform(text)
    X = vector.toarray()
    y = np.array(df.type).reshape(-1,1)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30, random_state=42)
    # print(X_train.shape,X_test.shape)
    return X_train,X_test,y_train,y_test

def train():
    df = read_csv(DATA_PATH+'mbti_1.csv')
    X_train,X_test,y_train,y_test = prepare(df)
    

if __name__ == "__main__":
    train()
