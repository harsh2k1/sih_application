import sys
import pandas as pd
import numpy as np
sys.path.append("../")
from config import DATA_PATH
from config import MODEL_PATH
from mbtitrain import read_csv

def concat():
    df = read_csv(DATA_PATH+'mbti_1.csv')
    df2 = read_csv(DATA_PATH+'questions_with_type.csv')
    final_df = pd.concat([df,df2])
    final_df.to_csv('mbti_1.csv')

if __name__ == "__main__":
    concat()