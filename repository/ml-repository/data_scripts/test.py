import sys
sys.path.append('../')
from config import DATA_PATH, MODEL_PATH
import pandas as pd

df = pd.read_csv(DATA_PATH+'raw_questions.csv')
df2 = pd.read_csv(DATA_PATH+'mbti_1.csv')
print(len(df))
final_df = pd.concat([df2,df])
print(final_df[-162:-161])