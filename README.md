# Disaster Response Pipeline Project
This repository contains files on the Udacity Project "Disaster Response Pipelines". It uses data on disaster messages, which is cleaned and saved into database. There is also a file, that trains a classifier that can categorize new diaster message. The classifier is available on a webpage.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Description of files
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- ETL Pipeline Preparation.ipynb
|- ML Pipeline Preparation.ibynb
|- process_data.py  # Script to clean and reshape the data and save into database
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py  # Gets data out of database and trains model on the data
|- classifier.pkl  # saved model 

- README.md # This file
