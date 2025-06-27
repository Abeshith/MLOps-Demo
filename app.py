import pickle
from flask import Flask, request, render_template
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from source.pipeline.prediction_pipeline import PredictionPipeline, CustomData


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'GET':
        return render_template('home.html')
    
    else:
        data = CustomData(
            gender = request.form.get('gender'),
            race_ethnicity = request.form.get('ethnicity'),
            parental_level_of_education = request.form.get('parental_level_of_education'),
            lunch = request.form.get('lunch'),
            test_preparation_course = request.form.get('test_preparation_course'),
            reading_score = int(request.form.get('reading_score')),
            writing_score = int(request.form.get('writing_score'))
        )

        data_df = data.get_data_as_dataframe()
        print(data_df)

        prediction_pipeline = PredictionPipeline()
        result = prediction_pipeline.predict(features=data_df)
        return render_template('home.html', results=f"{result[0]:.2f}")

if __name__ == '__main__':
    app.run(host= "0.0.0.0",debug=True)

