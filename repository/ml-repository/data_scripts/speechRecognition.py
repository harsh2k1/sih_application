from importlib.resources import path
import speech_recognition as sr 
import os 
from pydub import AudioSegment
from pydub.silence import split_on_silence
import speech_recognition as sr
import pyttsx3
import nltk
nltk.download('vader_lexicon')
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import spacy

r = sr.Recognizer()

lemmatizer = WordNetLemmatizer()
stopwords_english = stopwords.words('english')

def clean_data(text):
    text_clean = []
    text_tokens = word_tokenize(text)
    
    for word in text_tokens:
        if (word not in stopwords_english and # remove stopwords              
                word not in string.punctuation): # remove punctuation            
            stem_word = lemmatizer.lemmatize(word) # stemming word
            text_clean.append(stem_word)
    
    list_to_str = ' '.join([str(ele) for ele in text_clean])
    return list_to_str.lower()

def get_vader_sentiment(review): 
    sia = SentimentIntensityAnalyzer()
    sia = SentimentIntensityAnalyzer()
    analysis = sia.polarity_scores(review)
    return analysis['compound']


import speech_recognition as sr
import pyttsx3

# Initialize the recognizer
r = sr.Recognizer()
import time

t_end = time.time() + 15  # 15 secs

    # do whatever you do
# Function to convert text to
# speech
def SpeakText(command):
	
	# Initialize the engine
	engine = pyttsx3.init()
	engine.say(command)
	engine.runAndWait()
	
	
# Loop infinitely for user to
# speak

while time.time() < t_end:
    try:
        with sr.Microphone() as source2:
            r.adjust_for_ambient_noise(source2,duration=0.2)
            audio2 = r.listen(source2)
            myText = r.recognize_google(audio2)
            print('Output Text: ', myText)
            cleanText= clean_data(myText)
            compound_score = get_vader_sentiment(cleanText)
            print('Compound Score: ', compound_score)
    except:
        break
