# Disaster Response Pipeline Project


## Table of Contents
1. [Prerequisites](#Prerequisites)
2. [Project_Motivation](#Project_Motivation)
3. [App_Description](#App_Description)
4. [Licensing_Authors_Acknowledgements](#Licensing_Authors_Acknowledgements)
5. [Instructions](#Instructions)
6. [Screenshots](#screenshots)

<a name="descripton"></a>
## Prerequisites
The following package of nltk need to be download. 
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

## Project Motivation
This project was done to complete the Udacity Data Scientitst Nanodegree. Using Figure Eight's data by applying ETL, ML Pipeline, NLP etc to classify the disaster messages.This project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data


## App Description
In this project, it contains three main folders:
1. Data : sources files, data_cleansing code and sql db 
2. Model : contain the ML pipeline codes and one pkl file for model 
3. app : the Flask file to run the web application

## Licensing, Authors, Acknowledgements
Credits must be given to Udacity for the starter codes and FigureEight for provding the data used by this project.

## Instructions
Run the following commands in the project's root directory to set up your database and model.

To run ETL pipeline that cleans data and stores in database 
  `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
  
To run ML pipeline that trains classifier and saves 
  `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
  
Run the following command in the app's directory to run your web app. 
  `python run.py`

Go to http://0.0.0.0:3001/
