import pandas as pd
import sys
import pickle
sys.path.append('../')
from config import DATA_PATH, MODEL_PATH
from mbtitrain import read_csv, evaluate
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

def extract_questions(df):
    questions = [ele.strip('"').split('"')[0].strip() for ele in df['Description']]
    return questions

def main():
    df = read_csv(DATA_PATH+'raw_questions.csv')
    questions = extract_questions(df)
    df = pd.DataFrame(questions, columns=['posts'])
    df.to_csv(DATA_PATH+'questions.csv',index=None)
    

if __name__ == "__main__":
    main()
     
