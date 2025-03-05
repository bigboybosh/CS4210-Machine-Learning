#-------------------------------------------------------------------------
# AUTHOR: Joshua Gomez
# FILENAME: knn.py
# SPECIFICATION: Uses k-nearest neighbor algorithm to approximate classifications of data samples
# FOR: CS 4210- Assignment #2
# TIME SPENT: 20-25 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

#Reading the data in a csv file
with open('email_classification.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

error_count = 0

#Loop your data to allow each instance to be your test set
for i in db:

    #Add the training features to the 20D array X removing the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 2, 3, 4, 5, ..., 20]].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here

    X = []  # Training features
    Y = []  # Training labels
    
    for j in range(len(db)):
        if j != db.index(i):
            X.append([float(value) for value in db[j][:-1]])

            #Transform the original training classes to numbers and add them to the vector Y.
            #Do not forget to remove the instance that will be used for testing in this iteration.
            #For instance, Y = [1, 2, ,...].
            #Convert each feature value to float to avoid warning messages
            #--> add your Python code here
            Y.append(1 if db[j][-1] == 'spam' else 0)

    #Store the test sample of this iteration in the vector testSample
    #--> add your Python code here
    testSample = [float(value) for value in i[:-1]]
    true_label = 1 if i[-1] == 'spam' else 0

    #Fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2) #NOTE: p represents the power for Minkowski formula (general form of Euclidean distance) it's takes square of the attribute diff
    clf = clf.fit(X, Y)

    #Use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2, 3, 4, 5, ..., 20]])[0]
    #--> add your Python code here
    class_predicted = clf.predict([testSample])[0]

    #Compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
    if class_predicted != true_label:
        error_count += 1

#Print the error rate
#--> add your Python code here
error_rate = error_count / len(db)
print(f"Error rate: {error_rate:.2f}")






