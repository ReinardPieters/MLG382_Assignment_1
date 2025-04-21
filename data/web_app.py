import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from dash import Dash, html, dcc, Input, Output
import os
import dash_bootstrap_components as dbc

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

with open("../src/scaler_test.pkl", "rb") as f:
    scaler = pickle.load(f)


with open("../src/xgboost_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

with open("../src/logistic_regression_model.pkl", "rb") as f:
    logistic_model = pickle.load(f)

with open("../src/random_forest_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

dl_model = load_model("../src/deep_learning_model.h5")

# Sample input layout
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Student Grade Predictor", className='text-center'), width=12)
    ]),
    dbc.Row([
        dbc.Col(html.P("Enter values below to predict the grade class", className='text-center'), width=12)
    ]),
    dbc.Row([
        dbc.Col(dcc.Input(id='age', type='number', placeholder='Age', className='form-control mb-3'), width="auto")
    ], justify='center'),
    dbc.Row([
        dbc.Col(dcc.Input(id='study_time', type='number', placeholder='Study Time Weekly', className='form-control mb-3'), width=4),
        dbc.Col(dcc.Input(id='absences', type='number', placeholder='Absences', className='form-control mb-3'), width=4)
    ], justify='center'),
    dbc.Row([
        dbc.Col(dcc.Dropdown(
            id='gender', 
            options=[{'label': 'Male - 0', 'value': 0}, {'label': 'Female - 1', 'value': 1}],
            placeholder="Select Gender", 
            className='form-control mb-3'), width=4), 
        dbc.Col(dcc.Dropdown(
            id='ethnicity',
            options=[
                {'label': 'Caucasian - 0', 'value': 0},
                {'label': 'African American - 1', 'value': 1},
                {'label': 'Asian - 2', 'value': 2},
                {'label': 'Other - 3', 'value': 3}
            ],
            placeholder="Select Ethnicity", 
            className='form-control mb-3'), width=4),
    ], justify='center'),
    dbc.Row([
        dbc.Col(dcc.Dropdown(
            id='parental_education',
            options=[
                {'label': 'None - 0', 'value': 0},
                {'label': 'High School - 1', 'value': 1},
                {'label': 'Some College - 2', 'value': 2},
                {'label': 'Bachelors - 3', 'value': 3},
                {'label': 'Higher Study - 4', 'value': 4}
            ],
            placeholder="Select Parental Education", 
            className='form-control mb-3'), width=4),
        dbc.Col(dcc.Dropdown(
            id='parental_support',
            options=[
                {'label': 'None - 0', 'value': 0},
                {'label': 'Low - 1', 'value': 1},
                {'label': 'Moderate - 2', 'value': 2},
                {'label': 'High - 3', 'value': 3},
                {'label': 'Very High - 4', 'value': 4}
            ],
            placeholder="Select Parental Support", 
            className='form-control mb-3'), width=4)
    ], justify='center'),
    dbc.Row([
        dbc.Col(dcc.Checklist(
            id='tutoring',
            options=[{'label': 'Tutoring', 'value': 1}],
            value=[], 
            labelStyle={'display': 'inline-block'},
            className='mb-3'), width=2),
        dbc.Col(dcc.Checklist(
            id='extracurricular',
            options=[{'label': 'Extracurricular', 'value': 1}],
            value=[], 
            labelStyle={'display': 'inline-block'},
            className='mb-3'), width=2),
        dbc.Col(dcc.Checklist(
            id='volunteering',
            options=[{'label': 'Volunteering', 'value': 1}],
            value=[], 
            labelStyle={'display': 'inline-block'},
            className='mb-3'), width=2),
    ], justify='center'),
    dbc.Row([
        dbc.Col(dcc.Checklist(
            id='sports',
            options=[{'label': 'Sports', 'value': 1}],
            value=[], 
            labelStyle={'display': 'inline-block'},
            className='mb-3'), width=2),
        dbc.Col(dcc.Checklist(
            id='music',
            options=[{'label': 'Music', 'value': 1}],
            value=[], 
            labelStyle={'display': 'inline-block'},
            className='mb-3'), width=2)   
    ], justify='center'),
    dbc.Row([
        dbc.Col(html.Button("Predict", id='predict_button', n_clicks=0, className='btn btn-primary btn-block'), width="auto"),
    ], justify='center', class_name='mb-3'),
    dbc.Row([
        dbc.Col(html.Div(id='prediction-output', className='mt-4 text-center'), width=12)
    ], justify='center'),
])

@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict_button', 'n_clicks'),
     Input('age', 'value'),
     Input('gender', 'value'),
     Input('study_time', 'value'),
     Input('absences', 'value'),
     Input('tutoring', 'value'),
     Input('extracurricular', 'value'),
     Input('sports', 'value'),
     Input('music', 'value'),
     Input('volunteering', 'value'),
     Input('ethnicity', 'value'),
     Input('parental_education', 'value'),
     Input('parental_support', 'value')]
)
def predict_grade(n_clicks, age, gender,study_time, absences, tutoring,extracurricular, sports, music, volunteering, ethnicity, parental_education, parental_support):
    if n_clicks > 0 and None not in (age, gender, study_time, absences, ethnicity, parental_education, parental_support):
         input_data = {
            'Age': [age],
            'Gender': [gender],
            'StudyTimeWeekly': [study_time],
            'Absences': [absences],
            'Tutoring': [1 if tutoring and 1 in tutoring else 0],
            'Extracurricular': [1 if extracurricular and 1 in extracurricular else 0],
            'Sports': [1 if sports and 1 in sports else 0],
            'Music': [1 if music and 1 in music else 0],
            'Volunteering': [1 if volunteering and  1 in volunteering else 0],
            'Ethnicity_0': [1 if ethnicity == 0 else 0],
            'Ethnicity_1': [1 if ethnicity == 1 else 0],
            'Ethnicity_2': [1 if ethnicity == 2 else 0],
            'Ethnicity_3': [1 if ethnicity == 3 else 0],
            'ParentalEducation_0': [1 if parental_education == 0 else 0],
            'ParentalEducation_1': [1 if parental_education == 1 else 0],
            'ParentalEducation_2': [1 if parental_education == 2 else 0],
            'ParentalEducation_3': [1 if parental_education == 3 else 0],
            'ParentalEducation_4': [1 if parental_education == 4 else 0],
            'ParentalSupport_0': [1 if parental_support == 0 else 0],
            'ParentalSupport_1': [1 if parental_support == 1 else 0],
            'ParentalSupport_2': [1 if parental_support == 2 else 0],
            'ParentalSupport_3': [1 if parental_support == 3 else 0],
            'ParentalSupport_4': [1 if parental_support == 4 else 0]
        }
         input_df = pd.DataFrame(input_data)

         input_df = input_df

         num_features = ['Age', 'StudyTimeWeekly', 'Absences']
         input_df[num_features] = scaler.transform(input_df[num_features])

         log_prediction = logistic_model.predict(input_df)
         xgb_prediction = xgb_model.predict(input_df) 
         dl_prediction = dl_model.predict(input_df)
         rf_prediction = rf_model.predict(input_df)

         if len(dl_prediction.shape) == 2 and dl_prediction.shape[1] > 1:
             class_prediction = np.argmax(dl_prediction)
             probability = np.max(dl_prediction)
         else:
             class_prediction = int(round(float(dl_prediction[0][0])))
             probability = float(dl_prediction[0][0]) if class_prediction == 1 else 1 - float(dl_prediction[0][0])

         probability_percent = probability * 100

         return f"XGBoost Prediction: {xgb_prediction[0]}, Logorithemtic Prediction:{log_prediction[0]}, Random Forrest Prediction: {rf_prediction[0]}, Deep Learning Prediction: {class_prediction} with probability: {probability_percent:.2f}%"
    return "Please fill in all fields."
        
if __name__ == "__main__":
    print("Launching Dash app...")
    try:
        app.run(debug=True, host='127.0.0.1', port=8050)
    except Exception as e:
        print("Failed to start server:", e)