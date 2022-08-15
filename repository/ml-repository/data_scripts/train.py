import pandas as pd
import numpy as np
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import pickle
sys.path.append('../')
from config import MODEL_PATH, DATA_PATH

def save_model(Model_best):
    print(Model_best)
    filename = 'finalized_model2.sav'
    pickle.dump(Model_best, open(MODEL_PATH+ filename, 'wb'))
    
    # load the model from disk
    # loaded_model = pickle.load(open(filename, 'rb'))

def pre_processing(df):
    text = list(df.posts)
    vectorizer = CountVectorizer()
    vectorizer.fit(text)
    vector = vectorizer.transform(text)
    X = vector.toarray()[:2000]
    y = np.array(df.type).reshape(-1,1)[:2000]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30, random_state=42)
    return X_train,X_test,y_train,y_test

def train(df):
    X_train,X_test,y_train,y_test = pre_processing(df)
    Model = RandomForestClassifier(n_jobs=-1, random_state=100, class_weight='balanced')
    
    params = {'n_estimators':[100],
            'max_depth':[3,5,7,10,12,15],
            'max_features':[0.05,0.1,0.15,0.2],
            'criterion':["gini","entropy"]}

    grid_search = GridSearchCV(estimator=Model,param_grid=params,verbose=1,n_jobs=-1,scoring='accuracy')
    grid_search.fit(X_train,y_train)

    Model_best = grid_search.best_estimator_
    save_model(Model_best)

    y_train_pred = Model_best.predict(X_train)
    y_test_pred = Model_best.predict(X_test)


    print('Train Accuracy :',accuracy_score(y_train,y_train_pred))
    print('Test Accuracy :',accuracy_score(y_test,y_test_pred))
    print('Train Recall :',recall_score(y_train,y_train_pred, average='weighted'))
    print('Test Recall :',recall_score(y_test,y_test_pred, average='weighted'))


def main():
    df = pd.read_csv(DATA_PATH+'mbti_1.csv')
    train(df)

if __name__ == "__main__":
    main()