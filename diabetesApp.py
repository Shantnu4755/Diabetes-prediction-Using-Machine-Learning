import joblib
import sklearn
from flask import Flask, render_template, request, redirect
app = Flask(__name__, template_folder='template')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/Predict')
def prediction():
    return render_template('index.html')

@app.route('/form', methods=["POST"])
def brain():
    Pregnancies=float(request.form['Pregnancies'])
    Glucose=float(request.form['Glucose'])
    BloodPressure=float(request.form['BloodPressure'])
    SkinThickness=float(request.form['SkinThickness'])
    Insulin=float(request.form['Insulin'])
    BMI=float(request.form['BMI'])
    DiabetesPedigreeFunction=float(request.form['DiabetesPedigreeFunction'])
    Age=float(request.form['Age'])

     
    values=[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,DiabetesPedigreeFunction, Age]

    joblib.load('DiabtesPrediction_App/Diabetes_app', 'r')
    model = joblib.load(open('DiabtesPrediction_App/Diabetes_app', 'rb'))
    arr = [values]
    acc = model.predict(arr)
    print(acc)
    return render_template('prediction.html', prediction=str(acc))




if __name__ == '__main__':
    app.run(debug=True)















