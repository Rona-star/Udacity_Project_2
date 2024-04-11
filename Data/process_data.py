# import libraries 
import sys
import pandas as pd
import numpy as np
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine 


class DataProcessor:

    def load_data(self, messages_filepath, categories_filepath):

        """

        Load data from CSV files and merge them into a single DataFrame.

        Parameters:

        messages_filepath (str): Filepath to the messages CSV file.

        categories_filepath (str): Filepath to the categories CSV file.

        Returns:

        DataFrame: Merged dataframe containing messages and categories data.

        """

        messages = pd.read_csv(messages_filepath)

        categories = pd.read_csv(categories_filepath)

        return pd.merge(messages, categories, on='id')

    def clean_data(self, df):

        """

        Clean the dataframe by splitting categories, renaming columns, converting types, and dropping duplicates.

        Parameters:

        df (DataFrame): Input dataframe.

        Returns:

        DataFrame: Cleaned dataframe.

        """

        # Split categories into separate columns

        categories = df['categories'].str.split(';', expand=True)

        # Rename the columns

        category_colnames = categories.iloc[0].apply(lambda x: x[:-2])

        categories.columns = category_colnames

        # Convert category values to just numbers 0 or 1

        for column in categories:

            categories[column] = categories[column].str[-1].astype(int)

        # Replace 2s with 1s in 'related' column

        categories['related'] = categories['related'].replace(2, 1)

        # Drop original categories column

        df = df.drop('categories', axis=1)

        # Concatenate the dataframe with the new categories

        df = pd.concat([df, categories], axis=1)

        # Drop duplicates

        df = df.drop_duplicates()

        return df

    def save_data(self, df, database_filepath):

        """

        Save the dataframe to a SQLite database.

        Parameters:

        df (DataFrame): Input dataframe.

        database_filepath (str): Filepath of the SQLite database.

        """

        engine = create_engine(f'sqlite:///{database_filepath}')

        df.to_sql('disaster_messages', engine, index=False, if_exists='replace')

def main():

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        processor = DataProcessor()

        print('Loading data...')

        df = processor.load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')

        df = processor.clean_data(df)

        print('Saving data...')

        processor.save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:

        print('Please provide the filepaths of the messages and categories datasets as the first and second argument respectively, as well as the filepath of the database to save the cleaned data to as the third argument.')

        print('Example: python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db')

if __name__ == '__main__':

    main()
