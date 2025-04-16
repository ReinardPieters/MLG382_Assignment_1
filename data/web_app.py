import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from dash import Dash, html, dcc, Input, Output

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("input_columns.pkl", "rb") as f:
    input_columns = pickle.load(f)

# Load your models
with open("xgboost_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

with open("logistic_regression_model.pkl", "rb") as f:
    logistic_model = pickle.load(f)

with open("random_forest_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

dl_model = load_model("deep_learning_model.h5")

# Sample input layout
app = Dash(__name__)
app.layout = html.Div([
    html.H1("Student Grade Predictor"),
    html.P("Enter values below to predict the grade class"),
    dcc.Input(id='age', type='number', placeholder='Age'),
    dcc.Dropdown(
            id='gender', 
            options=[{'label': 'Male', 'value': 0}, {'label': 'Female', 'value': 1}],
            placeholder="Select Gender"
                ),
    dcc.Input(id='study_time', type='number', placeholder='Study Time Weekly'),
    dcc.Input(id='absences', type='number', placeholder='Absences'),
    dcc.Input(id='gpa', type='number', placeholder='GPA'),
    dcc.Dropdown(
        id='ethnicity',
        options=[{'label': f'Ethnicity_{i}', 'value': i} for i in range(4)],
        placeholder="Select Ethnicity"
    ),
    dcc.Dropdown(
        id='parental_education',
        options=[{'label': f'Parental Education {i}', 'value': i} for i in range(5)],
        placeholder="Select Parental Education"
    ),
    dcc.Dropdown(
        id='parental_support',
        options=[{'label': f'Parental Support {i}', 'value': i} for i in range(5)],
        placeholder="Select Parental Support"
    ),
    dcc.Checklist(
        id='sports',
        options=[{'label': 'Sports', 'value': 1}],
        value=[],
        labelStyle={'display': 'inline-block'}
    ),
    dcc.Checklist(
        id='music',
        options=[{'label': 'Music', 'value': 1}],
        value=[],
        labelStyle={'display': 'inline-block'}
    ),
     dcc.Checklist(
        id='tutoring',
        options=[{'label': 'Tutoring', 'value': 1}],
        value=[],
        labelStyle={'display': 'inline-block'}
    ),
     dcc.Checklist(
        id='extracurricular',
        options=[{'label': 'Extracurricular', 'value': 1}],
        value=[],
        labelStyle={'display': 'inline-block'}
    ),
     dcc.Checklist(
        id='volunteering',
        options=[{'label': 'Volunteering', 'value': 1}],
        value=[],
        labelStyle={'display': 'inline-block'}
    ),
    html.Button("Predict", id='predict_button', n_clicks=0),
    html.Div(id='prediction-output')
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
     Input('gpa', 'value'),
     Input('ethnicity', 'value'),
     Input('parental_education', 'value'),
     Input('parental_support', 'value')]
)
def predict_grade(n_clicks, age, gender,study_time, absences, tutoring,extracurricular, sports, music, volunteering, gpa, ethnicity, parental_education, parental_support):
    if n_clicks > 0 and None not in (age, gender, study_time, absences, gpa, ethnicity, parental_education, parental_support):
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
            'GPA': [gpa],
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

         for col in input_columns:
             if col not in input_df.columns:
                 input_df[col] = [0]

         input_df = input_df[input_columns]

         num_features = ['Age', 'StudyTimeWeekly', 'Absences', 'GPA']
         input_df[num_features] = scaler.transform(input_df[num_features])

         log_prediction = logistic_model.predict(input_df[input_columns])
         xgb_prediction = xgb_model.predict(input_df[input_columns]) 
         dl_prediction = dl_model.predict(input_df[input_columns])
         rf_prediction = rf_model.predict(input_df[input_columns])

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