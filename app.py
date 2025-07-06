import pickle 
from flask import Flask, request,app,jsonify,url_for,render_template
import numpy as np 
import pandas as pd



app = Flask(__name__)
##3 load the scaler 
scaler = pickle.load(open('scaler.pkl','rb'))
### load the model 
regmodel = pickle.load(open('regmodel.pkl','rb'))

### this is home page 
@app.route('/')
def home():
    return render_template('home.html')


### 
@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    ## prepairing the data for standardization
    print(np.array(list(data.values())).reshape(1,-1)) ## data.values() in dict format so, 
    ## making into list then reshape that dict values into one row and required columns 
    
    ### new scaled data 
    new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])


### predict 
@app.route("/predict",methods=["POST"])
def predict():
    data  = [float(x) for x in request.form.values()]
    final_input = scaler.tranform(np.array(data).reshape(1,-1))
    print(final_input)
    output = regmodel.predict(final_input)
    return render_template("home.html",prediction_text = "the Hose Price is {}".format(output))

if __name__ == "__main__":
    app.run(debug=True)
    