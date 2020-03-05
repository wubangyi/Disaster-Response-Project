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
    Y = df.iloc[:,3:]
    category_names = list(df.columns[3:])
    return X, Y,   category_names 

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

def build_model():
    #create pipeline based on vect, tfidf, clf
    pipeline = Pipeline(
                            [
                            ('vect', CountVectorizer(tokenizer = tokenize)),
                            ('tfidf', TfidfTransformer()),
                            ('clf', MultiOutputClassifier(RandomForestClassifier()))
                            ]
                            )
    
    parameters = {
                  'clf__estimator__n_estimators': [50, 100],
                  'clf__estimator__min_samples_split': [2, 3, 4],
                  'clf__estimator__criterion': ['entropy', 'gini']
                  }
    # GridSearchCV to test perfomances of each parameter and choose the best 
    cv = GridSearchCV(pipeline, param_grid=parameters)
   
    
    return cv
    

def evaluate_model(model, X_test, Y_test, category_names):
    #predict 
    y_pred = model.predict(X_test)
    #get the metrics and evaluate model performance 
    print(classification_report(y_pred, Y_test, target_names = Y_test.keys()))


def save_model(model, model_filepath):
    #open file 
    temp = open(model_filepath,'wb')
    #edit file 
    pickle.dump(model, temp)
    #close file
    temp.close()


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
        
        
        ###WILL NEED TO CXLEAN THIS UP
        print('TYPE OF MODEL')
        print(type(model))
        
        
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
