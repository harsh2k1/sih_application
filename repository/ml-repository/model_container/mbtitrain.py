import sys
import pandas as pd
sys.path.append("../")
from config import DATA_PATH

def read_csv(path):
    return pd.read_csv(path)

def train():
    df = read_csv(DATA_PATH+'mbti_1.csv')
    print(len(df))

if __name__ == "__main__":
    train()
