from __future__ import print_function
import keras
from keras import metrics
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
from keras.models import load_model
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
data_org = pd.read_csv('StateHousingPrice.csv')

# convert the region name to number
statemap = {}
for i in range(0,len(data_org['RegionName'].unique())):
    statemap[data_org['RegionName'].unique()[i]] = i
data_org['RegionName'] = data_org['RegionName'].map(statemap)

# split the time data to year and month
data_org['year'] = pd.to_numeric(data_org.Date.str.slice(0, 4))
data_org['month'] = pd.to_numeric(data_org.Date.str.slice(5, 7))


data = pd.DataFrame(data_org,columns=['year','month','RegionName','MedianListingPricePerSqft_AllHomes'])
label_col = 'MedianListingPricePerSqft_AllHomes'
data = data.dropna()
# print(data)

# split data frame indices into train and valid part, here we choose the percent of 70/30
np.random.seed(1)
data_train = np.random.permutation(data.index)[:int(0.7*len(data))]
data_valid = np.random.permutation(data.index)[int(0.7*len(data)):]

y_train = data.loc[data_train, [label_col]]
x_train = data.loc[data_train, :].drop(label_col, axis=1)
y_valid = data.loc[data_valid, [label_col]]
x_valid = data.loc[data_valid, :].drop(label_col, axis=1)


# Z-normalise the entire data frame and then convert them into numpy arrays to be used by Keras.
def z_score(col, df):
    mu = np.mean(df)
    s = np.std(df)
    newdf = pd.DataFrame()
    for c in col.columns:
        newdf[c] = (col[c]-mu[c])/s[c]
    return newdf

arr_x_train = np.array(z_score(x_train, data))
arr_y_train = np.array(y_train)
arr_x_valid = np.array(z_score(x_valid, data))
arr_y_valid = np.array(y_valid)


# print('Training shape:', arr_x_train.shape)
# print('Training samples: ', arr_x_train.shape[0])
# print('Validation samples: ', arr_x_valid.shape[0])

# creat a keras model with 3 layers and adam optimizer
def bulidmodel(x_size, y_size):
    t_model = Sequential()
    t_model.add(Dense(100, activation="tanh", input_shape=(x_size,)))
    t_model.add(Dense(50, activation="relu"))
    t_model.add(Dense(y_size))
    print(t_model.summary())
    t_model.compile(loss='mean_squared_error',optimizer=Adam(),metrics=[metrics.mae])
    return(t_model)


model = bulidmodel(arr_x_train.shape[1], arr_y_train.shape[1])

model.summary()


# fit and train keras model, record the history of training and validation
keras_callbacks = [EarlyStopping(monitor='val_mean_absolute_error', patience=20, verbose=0)]

history = model.fit(arr_x_train, arr_y_train,
    batch_size=128,
    epochs=500,
    shuffle=True,
    verbose=0,
    validation_data=(arr_x_valid, arr_y_valid),
    callbacks=keras_callbacks)

train_score = model.evaluate(arr_x_train, arr_y_train, verbose=0)
valid_score = model.evaluate(arr_x_valid, arr_y_valid, verbose=0)

print('Train MAE: ', round(train_score[1], 4), ', Train Loss: ', round(train_score[0], 4))
print('Val MAE: ', round(valid_score[1], 4), ', Val Loss: ', round(valid_score[0], 4))


def plot_hist(h, xsize=6, ysize=10):

    fig_size = plt.rcParams["figure.figsize"]
    plt.rcParams["figure.figsize"] = [xsize, ysize]
    fig, axes = plt.subplots(nrows=4, ncols=4, sharex=True)

    # summarize history for MAE
    plt.subplot(211)
    plt.plot(h['mean_absolute_error'])
    plt.plot(h['val_mean_absolute_error'])
    plt.title('Training vs Validation MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # summarize history for loss
    plt.subplot(212)
    plt.plot(h['loss'])
    plt.plot(h['val_loss'])
    plt.title('Training vs Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.draw()
    plt.show()
    return

plot_hist(history.history, xsize=8, ysize=12)

# save the mode and do predict
pprint(statemap)
model_json = model.to_json()
with open("houseprice.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights('houseprice.h5')
x = np.asarray([[2018,10,4]])
a = model.predict(x)
print(a)