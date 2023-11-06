# https://keras.io/
import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd

dataset = pd.read_csv('concrete.csv')

#split the dataset into featuers and target variable
columns = ['Cement',
'Blast Furnace Slag',
'Fly Ash', 
'Water', 
'Superplasticizer', 
'Coarse Aggregate', 
'Fine Aggregate', 
'Age']

predictors = np.asarray(dataset[columns])
target = np.asarray(dataset['Strength'])

model = Sequential()

#number of colums/predictors in dataset
n_cols = len(columns)

# Hidden layer #1
num_neurons = 5
model.add(Dense(num_neurons, activation='relu', input_shape=(n_cols,)))
# Hidden layer #2
model.add(Dense(num_neurons, activation='relu'))

# Output layer one neuron
model.add(Dense(1))

# Optimization
# 'adam' optimizer, no need to specify the learning rate for the model
model.compile(optimizer='adam', loss='mean_squared_error')

#training model with fit() method: https://keras.io/api/models/model_training_apis/#fit-method
model.fit(x=predictors, y=target, validation_split=0.25)

test_dataset = pd.read_csv('concrete_test_data.csv')
test_data = np.asarray(test_dataset[columns])
predictions = model.predict(x=test_data)

