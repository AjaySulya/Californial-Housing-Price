import pickle 
from flask import Flask, request,app,jsonify,url_for,render_template
import numpy as np 
import pandas as pd



app = Flask(__name__)
regmodel = pickle.load(open('regmodel.pkl','rb'))

### this is home page 
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',method=['POST'])
def predict_api():
    data = request.josn['data']
    print(data)
