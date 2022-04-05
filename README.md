# Disaster Response Pipeline Project

## Project Motivation
Consume new messages from Figure Eight and automatically categorize them for dispatching. Once fully built out this system 
could be a part of the API so that new messages are automatically categorized. This would save time and energy of response
teams and would streamline response efforts.

### Instructions:
1. Run the following commands in the project's root directory to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.sqlite`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.sqlite models/model.sav`

2. Go to `app` directory: `cd app`

3. Run the web app: `python run.py`

4. In the browser navigate to [http://127.0.0.1:3000](http://127.0.0.1:3000)

5. Review the visualizations to get an over of the data

6. Enter a message and click 'Classify Message' to see the ML model results

### Dependencies:
- NLTK
- Pandas
- Sklearn
- Flask
- Joblib
- Plotly

### Files:
```
app
| - static
| |- android-chrome-192x192.png # android phone favicon
| |- android-chrome-512x512.png # android table favicon
| |- apple-touch-icon.png # iOS favicon
| |- favicon.ico # desktop favicon
| |- favicon-16x16.png # desktop qhd favicon
| |- favicon-32x32.png # desktop uhd favicon
| |- script.js # site scripts
| |- site.webmanifest # crawl directory listing
| |- styles.css # site styles
| - templates
| |- master.html # main page of web app
| |- results.html # categorization results page of web app
data
| |- disaster_categories.csv # categories data to train ML classifier
| |- disaster_messages.csv # messages darta to train ML classifier
| |- DisasterResponse.sqlite # sqlite database with ETL data results
| |- process_data.py # script to ETL data from .csv files to the database
models
| |- model.sav # pickled model for the web app
| |- tokenize.py # script to tokenize the message CSV corpus
| |- train_classifier # script to train the ML classifier
ETL Pipeline Preparation.ipynb # notebook to explore the data and help build out the process_data.py script
Home Page Graphs ETL.ipynb # notebook to explore the data and help build out the home page of the app
ML Pipeline Preparation.ipynb # notebook to explore the data and help build out the train_classifier.py
README.md
```

