import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

melb_dataset=pd.read_csv("/home/hassan/Desktop/ML_Assignment-0/melb_data.csv")
melb=pd.DataFrame(melb_dataset)

obj_cats = ['Suburb', 'Address', 'Type', 'Method', 'SellerG', 'CouncilArea','Regionname','Postcode']

for colname in obj_cats:
    melb[colname] = melb[colname].astype('category')

melb['Date'] = pd.to_datetime(melb['Date'])
melb = melb.drop(['Bedroom2'],1)
melb['Age'] = 2019 - melb['YearBuilt']
# Remove rows missing data
melb = melb.dropna()
# Remove outlier
melb = melb[melb['BuildingArea']!=0]
features_df = pd.DataFrame(melb, columns=["Rooms", "Distance", "Bathroom", "Car", "Landsize",
            "BuildingArea", "Propertycount","Age"])
# Labels
labels_df = pd.DataFrame(melb, columns=["Price"])
labels_df.head()

# Train Test Split
from sklearn.model_selection import train_test_split
# Train Test Split
# Training Data = 80% of Dataset
# Test Data = 20% of Dataset
X_train, X_test, y_train, y_test = train_test_split(features_df, labels_df, test_size=0.2, random_state=101)
# Normalize Data
from sklearn.preprocessing import StandardScaler
# Define the Preprocessing Method and Fit Training Data to it
scaler = StandardScaler()
scaler.fit(X_train)
# Make X_train to be the Scaled Version of Data
# This process scales all the values in all 6 columns and replaces them with the new values
X_train = pd.DataFrame(data=scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
# Converting from Pandas Dataframe to Numpy Arrays
X_train = np.array(X_train)
y_train = np.array(y_train)
# Get the Type of Training Data
type(X_train), type(y_train)
# Apply same Normalization for Test Features
scal = StandardScaler()
scal.fit(X_test)
# Make X_test to be the Scaled Version of Data
# This process scales all the values in all columns and replaces them with the new values
X_test = pd.DataFrame(data=scal.transform(X_test), columns=X_test.columns, index=X_test.index)
# Convert test features and Labels to Numpy Arrays
X_test = np.array(X_test)
y_test = np.array(y_test)
# Get the Type of Test Data
type(X_test), type(y_test)
# Define Training Parameters

# Learning Rate
lr = 0.1

# Number of epochs for which the model will run
epochs = 1000
# Define Features and Label Placeholders

# Features
X = tf.placeholder(tf.float32,[None,X_train.shape[1]])

# Labels
y = tf.placeholder(tf.float32,[None,1])
# Define Hyperparameters

# Weight
W = tf.Variable(tf.ones([X_train.shape[1], 1]))

# Bias
b = tf.Variable(tf.ones(X_train.shape[1]))
# Initiaize all Variables
init = tf.global_variables_initializer()
# Define Cost Function, Optimizer and the Output Predicitons Function

# Predictions
# y_hat = (W*X + b)
y_hat = tf.add(tf.matmul(X, W), b)

# Cost Function
# MSE
cost = tf.reduce_mean(tf.square(y - y_hat))

# Gradient Descent Optimizer to Minimize the Cost
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)
# Tensor to store the cost after every Epoch
# Comes in handy while plotting the cost vs epochs
cost_history = np.empty(shape=[1],dtype=float)
with tf.Session() as sess:
    # Initialize all Variables
    sess.run(init)

    for epoch in range(0, epochs):
        # Run the optimizer and the cost functions
        result, err = sess.run([optimizer, cost], feed_dict={X: X_train, y: y_train})

        # Add the calculated cost to the array
        cost_history = np.append(cost_history, err)

        # Print the Loss/Error after every 100 epochs
        if epoch % 100 == 0:
            print('Epoch: {0}, Error: {1}'.format(epoch, err))

    print('Epoch: {0}, Error: {1}'.format(epoch + 1, err))

    # Values of Weight & Bias after Training
    new_W = sess.run(W)
    new_b = sess.run(b)

    # Predicted Labels
    y_pred = sess.run(y_hat, feed_dict={X: X_test})

    # Mean Squared Error
    mse = sess.run(tf.reduce_mean(tf.square(y_pred - y_test)))

    # New Value of Biases
    print('Trained Bias: \n', new_b)
    # Predicted Values
    print('Predicted Values: \n', y_pred)
    # Mean Squared Error
    print('Mean Squared Error [TF Session]: ', mse)