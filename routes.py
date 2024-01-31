# routes.py
# -*- coding: utf-8 -*-
from flask import Flask,request,jsonify
from joblib import load
import numpy as np
import csv

app = Flask(__name__)

@app.route('/', methods= ['GET'])
def helloworld():
     return 'hello world to flask 5000'

#model name 
model = load('models/gb_model_comp.joblib')

#file name 
csv_file = 'AIRCON_9_0130.csv'

data = []
target = []
current = []
pred = []

with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    next(reader)  
    for row in reader:
        data.append([row[1:]])  
        current.append(row[1])
        target.append(row[0]) 
x = np.array(data)
y = np.array(target)  
index = 0    
print(x)

@app.route('/dashboard/1', methods = ['GET']) # load
def dashboard_1():
    global index
    jdata ={
        "current": current[:index],
        "pred"   : pred,
        "target" : target[:index]
    }
    print('GET SUCCESS')

    return jsonify(jdata)

@app.route('/dashboard/2',methods = ['GET'])  #prediction / adding
def dashboard_2():
    global index
    ##model 
    prediction = model.predict(data[index])
    pred.append(round(float(prediction),1)) 
    
    jdata = {
        'pred'   : pred[-1],
        'current': float(target[index]),
        'target' : float(current[index])
    }
    index+=1
    print('GET SUCCESS')

    return jsonify(jdata)

if __name__ == '__main__':
	app.run(debug = True, port=5000)
 