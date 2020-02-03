import keras
from keras import metrics
from keras import regularizers
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
from keras.models import load_model
import os
import numpy as np
import pandas as pd
import json
from flask import Flask, render_template, request, redirect, Response, jsonify
from flask_wtf import FlaskForm
from pprint import pprint

json_file = open('houseprice.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights('houseprice.h5')
# print(loaded_model)
x = np.asarray([[2018,11,4]])
a = model.predict(x)

app = Flask(__name__)

domindata = pd.read_csv('domins.csv').to_json()

statenames = ['Alabama','Alaska','Arizona','Arkansas','California','Colorado','Connecticut','Delaware','Florida','Georgia','Hawaii','Idaho', 'Illinois','Indiana','Iowa','Kansas','Kentucky','Louisiana','Maine','Maryland','Massachusetts','Michigan','Minnesota','Mississippi','Missouri','Montana','Nebraska','Nevada','NewHampshire','NewJersey','NewMexico','NewYork','NorthCarolina','NorthDakota','Ohio','Oklahoma','Oregon','Pennsylvania','RhodeIsland','SouthCarolina','SouthDakota','Tennessee','Texas','Utah','Vermont','Virginia','Washington','WestVirginia','Wisconsin','Wyoming']
state_pca_name = []
state_mat_name=[]
for i in statenames:
    state_pca_name.append(i + "_pca")
    state_mat_name.append(i + "_corrMatrix")


state_pca_data = {}
for i in range(0,len(state_pca_name)):
    state_pca_data[statenames[i]] = []
    csv_data = pd.read_csv('pca/'+state_pca_name[i]).to_dict(orient='records')
    for j in range(0,len(csv_data)):
        csv_data[j].pop('Unnamed: 0')
        state_pca_data[statenames[i]].append(csv_data[j])
# print(state_pca_data)

state_mat_data = {}
for k in range(0,len(state_mat_name)):
    state_mat_data[statenames[k]] = []
    corrdata = pd.read_csv('corr/'+state_mat_name[k])
    # print(corrdata)

    for i in corrdata:
        if i != 'Unnamed: 0':
            for j in range(0,len(corrdata[i])):
                dict = {}
                dict['x'] = corrdata['Unnamed: 0'][j]
                dict['y'] = i
                dict['value'] = corrdata[i][j]
                state_mat_data[statenames[k]].append(dict)
# pprint(state_mat_data)


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST' or request.method =='GET':
        predata = request.form.to_dict()
        if not predata:
             presult = 'Please choose a time'
        else:
            # print(predata)
            prearray = np.asarray([[int(predata['Year']), int(predata['Month']), int(predata['States'])]])
            # print(prearray)
            if -1 in prearray:
                presult = 'Please choose a time'
            else:
                predictresult = model.predict(prearray)
                presult ="$" + str(predictresult[0][0]) + "/Sqft"
    return render_template("index.html", presult=presult,domindata=domindata,pca_data=state_pca_data, martirdata = state_mat_data)



if __name__ == "__main__":
    app.run(debug=True)


