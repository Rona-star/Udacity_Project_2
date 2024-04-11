# Disaster Response Pipeline
## Project Introduction
This project challenged my understanding in software engineering and data engineering in analyzing disaster data from Aspen, the former Figure Eight to build an API that classifies real disaster messages. 
The task involved creation of an ETL pipeline, ML pipeline and Web App that categorizes messages sent during disastrous events and ease message re-allocation to an appropriate disaster relief agency and display visualizations of data.

## File Descriptions
#### app
  | - template 
  
  ||- master.html # main page of web app 
  
  ||- go.html # classification result page of web app
  
  |- run.py # Flask file that runs app

#### data
  |- disaster_categories.csv # data to process 
  
  |- disaster_messages.csv # data to process 
  
  |- process_data.py # data cleaning pipeline 
  
  |- InsertDatabaseName.db # database to save clean data to

#### models
  |- train_classifier.py # machine learning pipeline 
  
  |- model.pkl # saved model

#### README.md

## Processes
1. ETL Pipeline
A Python script, process_data.py, this is a data cleaning pipeline that:
Loads the messages and categories datasets, Merges the two datasets Cleans the data and Stores it in a SQLite database A jupyter notebook was used for ETL Pipeline Preparation (EDA to prepare the process_data.py python script.)

2. ML Pipeline
A Python script, train_classifier.py, this is a machine learning pipeline that:
Loads data from the SQLite database and splits the dataset into training and test sets Builds a text processing and machine learning pipeline that trains and tunes a model using GridSearchCV Outputs results on the test set and exports the final model as a pickle file A jupyter notebook ML Pipeline Preparation was used to do EDA to prepare the train_classifier.py python script.

3. Flask Web App
This is where the emergency worker inputs a new message and gets the classifications and visualization of the dataThe web app will also display visualizations of the data.

## Pipelines and App Activation
1. Run the following commands in the project's root directory to set up the database and model.
- To run ETL pipeline that cleans data and stores in database
  
    'python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db'
  
- To run ML pipeline that trains classifier and saves
  
    'python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl'
  
2.Run the following command in the app's directory to run your web app.

    'python run.py'
    
3.Run this command to go to app's directory.

    'cd app'
    
4. Go to http://0.0.0.0:3001/
  
## Acknowledgements
Web App Code - Udacity (starter code for the web app)
