# routes.py
# -*- coding: utf-8 -*-
from flask import Flask,request,jsonify
#from joblib import load
import numpy as np
import csv
app = Flask(__name__)

#model name 
model = load('svm_model.joblib')

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
        data.append([row[1:-1]])  
        current.append(row[1])
        target.append(row[0]) 
x = np.array(data)
y = np.array(target)  
index = 0    
print(x)

@app.route('/dashboard/1', methods = ['GET']) # load
def hello():
    if request.method == 'GET':
        data ={
            "current": current[:index],
            "pred"   : pred,
            "target" : target[:index]
        }
    return jsonify(data)

@app.route('/dashboard/2',methods = ['GET'])  #prediction / adding
def hello_world():
    if request.method == 'GET':
        ##model 
        prediction = model.predict(data[index])
        pred.append(prediction) 
        data = {
            'pred'   : pred[-1],
            'current': target[-1],
            'target' : current[-1]
        }
        global index
        index+=1
    return jsonify(data)

if __name__ == '__main__':
	app.run(port=5000, debug=True)
 