########################## Neural Network for predicting continuous values ###############################
import numpy as np
import pandas as pd

# Importing necessary models for implementation of ANN
from keras.models import Sequential
from keras.layers import Dense #, Activation,Layer,Lambda
a=100

# Reading data 
Concrete = pd.read_csv("C:\\Users\\armifaizal\\Google Drive\\360DIGITMG DATA SCIENCE COURSE\\DATA SCIENCE CERIFICATION\\MODUL 14\\concrete.csv")
Concrete.head()

from sklearn.model_selection import train_test_split

X = Concrete.drop(["strength"],axis=1) # X is input, so have to drop Y which is strength 
Y = Concrete["strength"] # define out put Y

#split the dataset for test and train
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

#NN model
#input dim is inputs in 1st layer
# 50 layer is hidden layer 1
# next layer 250 hidden layer 2
# error calc using mean square err 
cont_model = Sequential()
cont_model.add(Dense(50, input_dim=8, activation="relu"))
cont_model.add(Dense(250, activation="relu"))
cont_model.add(Dense(100, activation="relu"))
cont_model.add(Dense(1, kernel_initializer="normal"))
cont_model.compile(loss="mean_squared_error", optimizer = "adam", metrics = ["mse"])

model = cont_model

#use the model and perform model fitting and training
model.fit(np.array(X_train), np.array(y_train), epochs=50)
# play around with epochs to find higher accuracy
# epocs is the no of times the model is being trained

# On Test dataset - test model on test dataset
pred = model.predict(np.array(X_test))
pred = pd.Series([i[0] for i in pred])

# Accuracy
np.corrcoef(pred, y_test)

layerCount = len(model.layers)
layerCount

# On Train dataset
pred_train = model.predict(np.array(X_train))
pred_train = pd.Series([i[0] for i in pred_train])

np.corrcoef(pred_train, y_train) #this is just because some model's count the input layer and others don't


#getting the weights:
#help (model.layers)
#hiddenWeights = model.layers[hiddenLayer].get_weights()
#lastWeights = model.layers[lastLayer].get_weights()
print(model.get_weights())

#accuracy calculation
accuracy=np.corrcoef(pred_train, y_train) 
print(accuracy)
