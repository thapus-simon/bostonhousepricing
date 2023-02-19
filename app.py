import pickle
from flask import Flask,request,app,jsonify,url_for,render_template


import numpy as np
import pandas as pd

app=Flask(__name__)
#Load the model
regmodel=pickle.load(open('regmodel.pkl','rb'))
scaler=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    # render_template() in flask will look in the templates folder
    # so I have created template dir
    # inside the template dir -> create a file home.html
    
    return render_template('home.html')

# make a predict api
# make a request using postman tool
app.route('/predict_api',methods=["POST"])
def predictapi():
    #The i/p that we are going to give -> make sure that its in json format 
    # which will captured in the data key, which will be stored in the data variable
    data=request.json['data']
    print(data)
    #AFter converting this to a list we need to reshape the file as per the pickling
    # (1,-1) -> for signle data point record
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_data)
    #output will be in a 2D array, will be taking the first value
    print(output[0])
    return jsonify(output[0])

if __name__=='__main__':
    app.run(debug=True)





