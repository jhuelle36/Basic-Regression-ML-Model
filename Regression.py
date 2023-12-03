import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

#Precision: amount of decimal places, suppress avoids scientific notation for large and small numbers
np.set_printoptions(precision=3, suppress=True)


import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)


#Get the data
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

#Read from the file and store in a Pandas DataFrame
raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)

#Assign copy of raw_daatset to "dataset" and then take the last few rows of data
dataset = raw_dataset.copy()
print(dataset.tail())


#Clean Data
dataset.isna().sum() #identifies and sums up NaN/missing values
dataset = dataset.dropna() #Removes rows with missing/NaN values


#Our Orgin column is categorical so we are 
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'}) #Replaces numeric values to country names
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='') #Performs one-hot encoding - Creates new binary columns for each country
print(dataset.tail()) # Dsiplay the last few rows


#Split data into training and testing data sets
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)


#Graph the data, diagonal graphs are kernel density estimates instead of histograms.
sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
plt.show()

# .describe() generates summary statistics and .tranpose() flips the columns and rows
print(train_dataset.describe().transpose())


#We are creating copies of data so we don't alter original data
train_features = train_dataset.copy()
test_features = test_dataset.copy()

#Now we are removing the MPG column form data and storing in variales. We will use this to train/test our data because we want to predict MPG
train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

#print(train_features.dtype)
#Shows only the mean and standard deviation of each column and we can see the large differences between column means, therefore we need to normalize data
#Normalizing helps because features will be multiplied by model weights, and with normalized data, training will be more smoother.
#Different models perform better with normalization: Larger scaled features may dominate other features, gradient descent works faster on normalized data, 
# better for interpertating data, and helps distance algorithims compare features based on the actual relationships between data and not the scale of the data.
print(train_dataset.describe().transpose()[['mean', 'std']])


#Needed to add this to get rid of errors
train_features = train_features.astype(np.float32)


#Creating a normalization layer that we can call when creating our model
normalizer = tf.keras.layers.Normalization(axis=-1)

#Used to compute and set the mean and variance of our normalization layer based on the train_features data provided
normalizer.adapt(np.array(train_features))





#Prints first example of data then the normalized version of it.
first = np.array(train_features[:1])

with np.printoptions(precision=2, suppress=True):
  print('First example:', first)
  print()
  print('Normalized:', normalizer(first).numpy())
  

#Single variable linear regression model for predicting MPG from Horespower
#Separate horsepower data from the whole training data
horsepower = np.array(train_features['Horsepower'])

#Create a layer for normalizing data
horsepower_normalizer = layers.Normalization(input_shape=[1,], axis=None)
#Normalize the horsepower data
horsepower_normalizer.adapt(horsepower)



#Creating our model with our normalization layer and a dense layer with one unit
#Dense Layer - FULLY connected layer - each neuron layer is connected to every neuron in the previous layer. It is common for the dense layer to have an
# activation function for an example: relu, sigmoid, softmax... these help model learn complex, non-linear relationships, by default though, Dense is set
# to have a linear activation function. Since we are checking a continous variable: MPG, this is valid.
horsepower_model = tf.keras.Sequential([
    horsepower_normalizer,
    layers.Dense(units=1),#Units is number of layers, if we said units=10, then 10 neurons in our dense layer.
    
])

horsepower_model.summary()


# Make predictions on the first 10 horespower values
horsepower_model.predict(horsepower[:10])

#Using the Adam optimizer, configure model.
horsepower_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

#Loss Function: Measures the models performance based on parameters: weights and biases between neurons. Examples: mse, rmse, mae
#Optimizer: Is an algorithim that determines how to adjust these parameters to minimalize the loss function. Examples: Adam, SGD(Stochastic Gradient Descent)
# To effectively train a model you need both an optimizer and a loss funmction defined, there are no defaults.



#Train the model
history = horsepower_model.fit(
    train_features['Horsepower'], #Data for HP
    train_labels, #Data for MPG
    epochs=100, #Go through dataset 100 times
    verbose=0, #Displays the information it logs throughout training: 0 - silent, 1 - Shows a progress bar, 2 - One line per epoch
    validation_split = 0.2 #20 percent of data will be used for validation
    )
#Validation data helps monitor overfitting when training a model, it uses data the model has not seen yet to test as it trains. It also can stop model early
# if it is no longer improving.

#Turns history into a pandas DataFrame
hist = pd.DataFrame(history.history)
#Adds a epoch column that now holds the epoch numbers
hist['epoch'] = history.epoch
print(hist.tail())


#Plot the history of your model
def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)
  

plot_loss(history)



test_results = {}

#evaluating performance of the mdoel with test data
test_results['horsepower_model'] = horsepower_model.evaluate(
    test_features['Horsepower'],
    test_labels,
    verbose=0)



#Creates a linearly spaced vector with 251 values from 0 to 250
x = tf.linspace(0.0, 250, 251)
#Obtains predictions form the model for each value of x
y = horsepower_model.predict(x)


#Function that plots out the x values and the predicitions
def plot_horsepower(x, y):
  plt.scatter(train_features['Horsepower'], train_labels, label='Data')
  plt.plot(x, y, color='k', label='Predictions')
  plt.xlabel('Horsepower')
  plt.ylabel('MPG')
  plt.legend()
  


plot_horsepower(x, y)
plt.show()


#Now creating a  model that will take in multiple inputs
linear_model = tf.keras.Sequential([
    normalizer, #Our normalizer layer that will normalize the multiple parameters
    layers.Dense(units=1) #Dense layer with 1 neuron
])

#Make predictions with the first 10 examples from the traiing 
linear_model.predict(train_features[:10])

#Kernels - weights associated with the connections between neurons in a layer.
#Accesses the weights(kerenl) of the second layer of our model(dense layer) in the linear_model and prints them 
linear_model.layers[1].kernel
print(linear_model.layers[1].kernel.numpy())



#Compile our model using the Adam optimizer and mae as our loss function
#The learning rate 0.1 is telling how much the optimizer should skip by when testing new weights and biases
linear_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')


history = linear_model.fit(
    train_features,#x
    train_labels,#y
    epochs=100,#Run through data 100 times
    verbose=0,#Display nothing when training
    validation_split = 0.2 #20 percent of the training data will be used for validation - testing model on data it has not yet seen during training which reduces overfitting
    )

#Plot the loss of the multople parameter linear model
plot_loss(history)

# Added this to get rid of errors
test_features = test_features.astype(np.float32)

test_results['linear_model'] = linear_model.evaluate(
    test_features, test_labels, verbose=0)


plt.show()




#REGRESSION WITH A DEEP LEARNING NEURAL NETWORK(DNN)
#We are gonna do the same exact thing as before but this time with a DNN

#Function that builds a model, compiles it and then returns the model
def build_and_compile_model(norm):
  #
  model = keras.Sequential([
      norm, #Pass in the normalized layer, this will be specific to the type of data being added
      layers.Dense(64, activation='relu'), #Relu adds non-linearity to the model, Dense is still a fully connected layer, but this time with 64 neurons(hidden)
      layers.Dense(64, activation='relu'), #(hidden) relu - Rectified Learning Unit - sets negative values to 0 and positive values stay the sameo - I am not sure how this would help
      layers.Dense(1) #Output layer of model, 1 neuron with linear activation, typical for regression due to the output being a continous range of values
  ])
  #Compile model with the mae losss finction and the Adam optimizer
  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model



#Creating a model with the single parameter normalizer
dnn_horsepower_model = build_and_compile_model(horsepower_normalizer)

#print the summary stats
dnn_horsepower_model.summary()

#Train the data with the horsepower column from training features.
history = dnn_horsepower_model.fit(
    train_features['Horsepower'],
    train_labels,
    validation_split=0.2, #20 percent goes towards validation data
    verbose=0, #No display during training
    epochs=100 #Run through data 100 time
    )


#plot the data
plot_loss(history)



#Create a vector space from 0 to 250 to make predictions on DNN
x = tf.linspace(0.0, 250, 251)
y = dnn_horsepower_model.predict(x)

plot_horsepower(x, y)
plt.show()

#Add the results fromt he model to our test results dictionar for later
test_results['dnn_horsepower_model'] = dnn_horsepower_model.evaluate(
    test_features['Horsepower'], test_labels,
    verbose=0)



#Regression using a DNN with multiple inputs

#Build and compile the model with the multivariable normalizer and display the summary
dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()


#Train the model 
history = dnn_model.fit(
    train_features,
    train_labels,
    validation_split=0.2, #20 percent goes towards validation data
    verbose=0, #Nothing is displayed during training
    epochs=100 #Runs through the training dat 100 times
    )


plot_loss(history)
plt.show()


#Add to the dictionary of model results
test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)

#Using pandas DaatFrame to dsiplay the test results(tranpose of results)
print(pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T)


#                      Mean absolute error [MPG] - On average the model is this far off the real cars MPG(MAE is an absolute measure so this can be in positve or negative direction)
#horsepower_model                       3.646586
#linear_model                           2.505629
#dnn_horsepower_model                   2.938782
#dnn_model                              1.731787


#Make predictions using the test data
test_predictions = dnn_model.predict(test_features).flatten()

#
a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)



error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [MPG]')
_ = plt.ylabel('Count')


plt.show()




#You can save and reload the model
#dnn_model.save('dnn_model.keras')


##reloaded = tf.keras.models.load_model('dnn_model.keras')

#test_results['reloaded'] = reloaded.evaluate(
#    test_features, test_labels, verbose=0)


#pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T


