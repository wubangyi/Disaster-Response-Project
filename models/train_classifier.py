import sys
import sys
import re
import pandas as pd 
from sqlalchemy import create_engine
import sqlite3
import numpy as py
import nltk 
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, classification_report
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])

def load_data(database_filepath):
    # connect database and select data
    engine = create_engine('sqlite:///'+database_filepath)
    df= pd.read_sql_table('DisasterInfomation',engine)
    # make message as feature and all of the others columns as target
    X = df['message'].values
    y = df.iloc[:,3:]
    category_names = y.keys() 
    return X, y,   category_names 

def tokenize(text):
    # Normalization
    text = re.sub(r"[^a-zA-Z0-9]", ' ', text.lower())
    # tokenize
    words = word_tokenize(text)
    # remove stopwords 
    text = [w for w in words if w in stopwords.words('english')]
    
    #Lemmatize
    lemmatizer = WordNetLemmatizer()
    lemmed = [lemmatizer.lemmatize(w, pos='n').strip() for w in words]
    lemmed = [lemmatizer.lemmatize(w, pos='v').strip() for w in lemmed]
    
    return lemmed
