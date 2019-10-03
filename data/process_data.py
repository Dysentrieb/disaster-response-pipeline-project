import sys
import re
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    # Load and merge data
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')

    # create a dataframe of the 36 individual category columns
    cats = df.categories
    # select the first row of the categories dataframe
    row = cats[0]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    row = re.sub(r'-.', '', row)
    category_colnames = pd.Series(row).str.split(';', expand=True).values.tolist()[0]
    cats_split = pd.Series(cats).str.split(';', expand=True)
    # rename the columns of `categories`
    cats_split.columns = category_colnames

    for column in cats_split:
    # set each value to be the last character of the string
        cats_split[column] = cats_split[column].map(lambda x: x.lstrip('[abcdefghijklmnopqrstuvwxyz]-_'))
        # convert column from string to numeric
        cats_split[column] = cats_split[column].astype(int)

    # drop the original categories column from `df`
    df = df.drop(['categories'], axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, cats_split], axis=1)
    return df


def clean_data(df):
    return df.drop_duplicates()


def save_data(df, database_filename):
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DataTable', engine, index=False)
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()