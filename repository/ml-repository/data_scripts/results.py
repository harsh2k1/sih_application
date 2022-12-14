from importlib.resources import path
from multiprocessing.managers import DictProxy
from operator import index
import mbtipredict
import pandas as pd
import sys
sys.path.append('../')
from config import DATA_PATH
from random import randint

def func():
    final_predicted_output = mbtipredict.test() # personality Type
    df = pd.read_csv(DATA_PATH+'recommendation.csv')
    career_options = df[df['Personalities'] == final_predicted_output]['Career Options']
    career_options = career_options.values[0].split(',')
    matches = [100/len(career_options)]*4

    def get_match_1(score, sents):
        index = randint(0,len(career_options))
        return career_options[index], index

    def get_match_2(score, path1):
        while True:
            index = randint(0,len(career_options))
            print(index)
            path = career_options[index]
            if path != path1:
                return path, index

    path1, index1 = get_match_1(20,0.5)
    path2, index2 = get_match_2(20,path1)

    matches[index1] += 50
    matches[index2] += 25

    print(dict(zip(career_options, matches)))


def evaluate_results():
    final_predicted_output = mbtipredict.test() # personality Type
    li1 = ['INTJ','INTP','ENTJ','ENTP','INFJ','INFP','ENFJ','ENFP','ISTJ','ISFJ','ESTJ','ESFJ','ISTP','ISFP','ESTP','ESFP']
    li2 = ['Architect','Logician','Commander','Debater','Advocate','Mediator','Protagonist','Campaigner','Logistician','Defender','Executives','Consul','Virtuoso','Adventurer','Entepreneur','Entertainer']
    li2 = [ele.replace(' ','_').lower() for ele in li2]
    # final_predicted_output = test()
    x = dict(zip(li1,li2))
    path = x.get(final_predicted_output) + '.html'
    print(path)
    return final_predicted_output


evaluate_results()
