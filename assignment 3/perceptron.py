#-------------------------------------------------------------------------
# AUTHOR: Joshua Gomez
# FILENAME: perceptron.py
# SPECIFICATION: runs a single-layer and multi=layer perceptron using shuffling to potentially randomize the order of input. The expectation is that MLP will
# perform better than Single-layer neural network, and yes, but remember that it is a plateau, not a clear cut winner. As n and r are traversed, it generates
# the specs for the neural network, and compares the performance of both MLP and single-layer by how accurate their predictions were. Recall that accuracy
# is the rate of correctly-labeled classifciations over total classifications
# FOR: CS 4210- Assignment #3
# TIME SPENT: 25 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier #pip install scikit-learn==0.18.rc2 if needed
import numpy as np
import pandas as pd

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to form the feature data for training
y_training = np.array(df.values)[:,-1]  #getting the last field to form the class label for training

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the data by using Pandas library

X_test = np.array(df.values)[:,:64]    #getting the first 64 fields to form the feature data for test
y_test = np.array(df.values)[:,-1]     #getting the last field to form the class label for test

highest_perceptron_accuracy = 0
highest_mlp_accuracy = 0

for learning_rate in n: #iterates over n

    for shuffling in r: #iterates over r

        #iterates over both algorithms
        #-->add your Pyhton code here

        for algorithm in ["MLP", "Perceptron"]: #iterates over the algorithms

            #Create a Neural Network classifier
            #if Perceptron then
            #   clf = Perceptron()    #use those hyperparameters: eta0 = learning rate, shuffle = shuffle the training data, max_iter=1000
            #else:
            #   clf = MLPClassifier() #use those hyperparameters: activation='logistic', learning_rate_init = learning rate,
            #                          hidden_layer_sizes = number of neurons in the ith hidden layer - use 1 hidden layer with 25 neurons,
            #                          shuffle = shuffle the training data, max_iter=1000
            #-->add your Pyhton code here
            if algorithm == "Perceptron":
                clf = Perceptron(eta0= learning_rate, shuffle= shuffling, max_iter= 1000) # eta0 stands for initial eta (greek symbol, looks like lowercase n)
            else:
                clf = MLPClassifier(
                    activation= 'logistic',    # activation= 'logistic' specifies activation function to be logistic sigmoid function
                    learning_rate_init= learning_rate,
                    hidden_layer_sizes= (25,), # accepts a tuple, defined by both parentheses but also by the comma for single-value
                    shuffle= shuffling,        # shuffle alters the order of the training data, introduces randomness
                    max_iter= 1000)

            #Fit the Neural Network to the training data
            clf.fit(X_training, y_training)

            #make the classifier prediction for each test sample and start computing its accuracy
            #hint: to iterate over two collections simultaneously with zip() Example:
            #for (x_testSample, y_testSample) in zip(X_test, y_test):
            #to make a prediction do: clf.predict([x_testSample])
            #--> add your Python code here
            correct_predictions = 0
            for (X_testSample, y_testSample) in zip(X_test, y_test): # RECALL, zip() function outputs tuples
                o = clf.predict([X_testSample])                      # labeled o to be consistent with learning material

                if o[0] == y_testSample:
                    correct_predictions += 1

            #check if the calculated accuracy is higher than the previously one calculated for each classifier. If so, update the highest accuracy
            #and print it together with the network hyperparameters
            #Example: "Highest Perceptron accuracy so far: 0.88, Parameters: learning rate=0.01, shuffle=True"
            #Example: "Highest MLP accuracy so far: 0.90, Parameters: learning rate=0.02, shuffle=False"
            #--> add your Python code here
            accuracy = correct_predictions / len(y_test)

            if algorithm == "Perceptron":
                if accuracy > highest_perceptron_accuracy:
                    highest_perceptron_accuracy = accuracy
                    print(f"Highest Perceptron accuracy so far: {highest_perceptron_accuracy}, Parameters: learning rate= {learning_rate}, shuffle={shuffling}")
            else:
                if accuracy > highest_mlp_accuracy:
                    highest_mlp_accuracy = accuracy
                    print(f"Highest MLP accuracy so far: {highest_mlp_accuracy}, Parameters: learning rate= {learning_rate}, shuffle={shuffling}")

