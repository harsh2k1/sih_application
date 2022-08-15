import sys
sys.path.append('../')
from config import DATA_PATH, MODEL_PATH
import pandas as pd
import random

df = pd.read_csv(DATA_PATH+'questions_with_type.csv')
df.fillna('',inplace=True)
li = ['ESTJ', 'ENTJ', 'ESFJ', 'ENFJ', 'ISTJ', 'ISFJ', 'INTJ', 'INFJ', 'ESTP', 'ESFP', 'ENTP', 'ENFP', 'ISTP', 'ISFP', 'INTP','INFP']
li2 = list(range(16))
dct = dict(zip(li2,li))
x = [i for i in range(len(df)) if df['type'].loc[i] == '']
for i in x:
    df['type'].loc[i] = dct.get(random.randint(0,16))

df.to_csv(DATA_PATH+'questions_with_type.csv',index=None)