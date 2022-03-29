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
- ETL Pipeline Preparation.ipynb - Extract, Transform and Load the data from a CSV into the database
- ML Pipeline Preparation.ipynb - Explore, and clean the data and train the ML model
- data/process_data.py - Script derived from the ETL notebook
- models/train_classifier.py - Script derived from ML notebook

