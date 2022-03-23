# import libraries
import sys
import pandas as pd
import sqlite3


def load_data(messages_filepath, categories_filepath):
    """
    Combine the categories and messages into a single DF with boolean values
    :param messages_filepath:
    :param categories_filepath:
    :return:
    """
    messages = pd.read_csv(messages_filepath)

    categories = pd.read_csv(categories_filepath)

    df = messages.merge(categories, how='left', on=['id'])

    return df


def clean_data(df):
    """
    Clean the data and prepare for load
    :param df:
    :return df:
    """
    categories = df['categories'].str.split(';', expand=True)

    row = categories.iloc[0]

    category_column_names = row.str.split('-', expand=True)[0]

    categories.columns = category_column_names

    # loop through each category and set the value to be an int
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # The categories' column is no longer needed since it's been split out into individual columns
    df.drop(['categories'], axis=1, inplace=True)

    # Concat the cleaned categories onto the end of the dataframe
    df = pd.concat([df, categories], axis=1)

    # Remove any duplicates so the model can be trained
    df.drop_duplicates(inplace=True)

    # Remove any non-binary results
    df.replace(2, 1, inplace=True)

    return df


def save_data(df, database_filename):
    """
    Save the data into the database for the ML pipeline
    :param df:
    :param database_filename:
    :return:
    """
    # Setup connection
    conn = sqlite3.connect(database_filename)

    # get a cursor
    cur = conn.cursor()

    # Drop the table if it exists
    cur.execute("DROP TABLE IF EXISTS categorized_messages")

    # create the table including id as a primary key
    cur.execute('CREATE TABLE categorized_messages '
                '(id TEXT PRIMARY KEY, message TEXT, original TEXT, genre TEXT, related TEXT, request TEXT, offer TEXT,'
                ' aid_related TEXT, medical_help TEXT, medical_products TEXT, search_and_rescue TEXT, security TEXT, '
                'military TEXT, child_alone TEXT, water TEXT, food TEXT, shelter TEXT, clothing TEXT, money TEXT, '
                'missing_people TEXT, refugees TEXT, death TEXT, other_aid TEXT, infrastructure_related TEXT, '
                'transport TEXT, buildings TEXT, electricity TEXT, tools TEXT, hospitals TEXT, shops TEXT, '
                'aid_centers TEXT, other_infrastructure TEXT, weather_related TEXT, floods TEXT, storm TEXT, '
                'fire TEXT, earthquake TEXT, cold TEXT, other_weather TEXT, direct_report TEXT);'
    )

    # insert data
    df.to_sql('categorized_messages', con=conn, if_exists='replace')

    # commit changes made to the database
    conn.commit()


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
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
