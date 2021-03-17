from flask import Flask,render_template,request
from sklearn.externals import joblib
import pandas as pd
import numpy as np


app=Flask(__name__)
# @app.route('/test')
#
# def test():
#     return "Flask is being used by charu"
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        try:
            NewYork = float(request.form['NewYork'])
            California = float(request.form['California'])
            Florida = float(request.form['Florida'])
            RnDSpend = float(request.form['RnDSpend'])
            AdminSpend = float(request.form['AdminSpend'])
            MarketSpend = float(request.form['MarketSpend'])
            pred_args =[NewYork,California,Florida,RnDSpend,AdminSpend,MarketSpend]
            pred_args_arr=np.array(pred_args)
            pred_args_arr=pred_args_arr.reshape(1,-1)
            mul_reg =open('multiple_regression_model.pkl',"rb")
            ml_model = joblib.load(mul_reg)
            model_prediction = ml_model.predict(pred_args_arr)
            model_prediction = round(float(model_prediction),2)

        except valueError:
            return "Please Check if the values are entered correctly!"
    return render_template('prediction.html',prediction=model_prediction)
if __name__=="__main__":
    app.run()
