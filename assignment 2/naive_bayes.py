#-------------------------------------------------------------------------
# AUTHOR: Joshua Gomez
# FILENAME: naive_bayes.py
# SPECIFICATION: Uses naive bayes algorithm to predict probabilies of classifiers given training and testing data
# FOR: CS 4210- Assignment #2
# TIME SPENT: Around 20 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

#Reading the training data in a csv file
#--> add your Python code here
dbTraining = []

with open('weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            dbTraining.append (row)

X = []
Y = []

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
outlook_map = {"Sunny": 1, "Overcast": 2, "Rain": 3}
temperature_map = {"Hot": 1, "Mild": 2, "Cool": 3}
humidity_map = {"High": 1, "Normal": 2}
wind_map = {"Weak": 1, "Strong": 2}
class_map = {"Yes": 1, "No": 2}

for i in dbTraining:
    X.append([
        outlook_map[i[1]],
        temperature_map[i[2]],
        humidity_map[i[3]],
        wind_map[i[4]]
    ])

    #Transform the original training classes to numbers and add them to the vector Y.
    #For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    Y.append(class_map[i[5]])


#Fitting the naive bayes to the data
clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X, Y)

#Reading the test data in a csv file
#--> add your Python code here
dbTest = []

with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            dbTest.append (row)

X_test = []
for i in dbTest:
    X_test.append([
        outlook_map[i[1]],
        temperature_map[i[2]],
        humidity_map[i[3]],
        wind_map[i[4]]
    ])

#Printing the header os the solution
#--> add your Python code here
print("Day        Outlook     Temperature   Humidity   Wind    PlayTennis   Confidence")

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)

for i in range(len(dbTest)):
    confidence = max(probabilities[i])

    if confidence >= 0.75:  # Only print results with confidence >= 0.75
        predicted_label = "Yes" if predictions[i] == 1 else "No"

        print(f"{dbTest[i][0]:<10} {dbTest[i][1]:<12} {dbTest[i][2]:<12} "
              f"{dbTest[i][3]:<10} {dbTest[i][4]:<8} {predicted_label:<12} {confidence:.2f}")
