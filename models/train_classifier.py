import sys
# import libraries
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import time

# Tokenization
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords

# Text processing
from nltk.stem.wordnet import WordNetLemmatizer

# ML
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

# Export
from joblib import dump, load


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DataTable', engine)
    X = df['message']
    y = df.drop(['id', 'genre', 'message', 'original'], axis=1)
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    # Transform to lower case
    text = text.lower()
    # Normalize text
    table = str.maketrans({key: None for key in string.punctuation})
    text_1 = text.translate(table) 
    # Tokenize text
    word_tokens = word_tokenize(text_1) 
    filtered_sentence = [w for w in word_tokens if w not in stopwords.words("english")]  
    return filtered_sentence


def build_model():
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
             ('tfidf', TfidfTransformer()),
             ('moc', MultiOutputClassifier(RandomForestClassifier()))])   
    return pipeline

def train_model(model, X, y):
    # Set Parameters for GridSearch
    parameters = {
    'moc__estimator__min_samples_split': [2, 4]
    }
    # Apply GridSearch
    modelCV = GridSearchCV(model, param_grid=parameters)
    modelCV.fit(X, y)
    # Print meesage and best Parameters
    print('Best Parameters found with GridSearchCV:')
    print(modelCV.best_params_)
    return modelCV

def get_metrics(y_test, y_pred, category_names):
    # Get Scores
    f1_list = []
    precision_list = []
    recall_list = []
    #target_names = y.columns
    y_test =  np.array(y_test)
    y_pred =  np.array(y_pred)
    for n in range(y_test.shape[1]):    
        report = classification_report(y_test[:, n], y_pred[:, n])
        print(category_names[n] + ':')
        print(report)
        report_split = report.split()
        # report = classification_report(y_test[:, n], y_pred[:, n], output_dict=True)
        #f1_list.append(report['weighted avg']['f1-score'])
        #precision_list.append(report['weighted avg']['precision'])
        #recall_list.append(report['weighted avg']['recall'])
        f1_list.append(float(report_split[-2]))
        precision_list.append(float(report_split[-4]))
        recall_list.append(float(report_split[-3]))
    
    # Merge lists into Dataframe
    metrics = pd.DataFrame(
        {'Category': category_names,
         'Precision': precision_list,
         'Recall': recall_list,
         'F1-Score': f1_list
        })
    
    return metrics

def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    metrics = get_metrics(Y_test, Y_pred, category_names)
    print(metrics.mean())
    pass


def save_model(model, model_filepath):
    # Save the model to a pkl file
    dump(model, model_filepath) 
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model = train_model(model, X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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