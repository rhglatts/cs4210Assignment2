#-------------------------------------------------------------------------
# AUTHOR: Rebecca Glatts
# FILENAME: naive_bayes.py
# SPECIFICATION: Trains a naive bayes algorithm on weather_training.csv and 
# calculates the probability of playing tennis using the test set. Outputs result 
# if confidence >= 0.75
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 hour
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

#reading the training data in a csv file
training = []
with open('weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            training.append (row)

#transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
feature_to_value = {"Sunny" : 1, "Overcast" : 2, "Rain" : 3, "Hot" : 4, "Mild" : 5, 
                    "Cool" : 6, "Normal": 7, "High" : 8, "Weak": 9, "Strong": 10}
X = []
for row in training:
    temp = []
    for i in range (1, 5):
        temp.append(feature_to_value.get(row[i]))
    X.append(temp)

#transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
class_to_value = {"Yes" : 1, "No" : 2}
Y = []
for row in training:
    Y.append(class_to_value.get(row[5]))

#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

#reading the test data in a csv file
test_original = []
with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            test_original.append (row)

test = []
for row in training:
    temp = []
    for i in range (1, 5):
        temp.append(feature_to_value.get(row[i]))
    test.append(temp)

#printing the header of the solution
print("Day  Outlook   Temperature  Humidity   Wind    PlayTennis  Confidence")

#use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
for row, original in zip(test, test_original):
    prediction = clf.predict_proba([row])[0]
    if prediction[0] >= 0.75:
        print("{:<5} {:<10} {:<10} {:<10} {:<10}".format(original[0], original[1], original[2], original[3], original[4]), end="")
        print(f"Yes         {prediction[0]:.2f}\n")
    elif prediction[1] >= 0.75:
        print("{:<5} {:<10} {:<10} {:<10} {:<10}".format(original[0], original[1], original[2], original[3], original[4]), end="")
        print(f"No         {prediction[1]:.2f}\n")


