#-------------------------------------------------------------------------
# AUTHOR: Rebecca Glatts
# FILENAME: knn.py
# SPECIFICATION: Calculates the LOO-CV error rate for 1NN using the data in binary_points.csv
# FOR: CS 4210- Assignment #2
# TIME SPENT: 2 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

#reading the data in a csv file
with open('binary_points.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

#loop your data to allow each instance to be your test set
error = 0
for row in (db):

    #add the training features to the 2D array X removing the instance that will be used for testing in this iteration. For instance, X = [[1, 3], [2, 1,], ...]]. Convert each feature value to
    # float to avoid warning messages
    #--> add your Python code here
    X = []
    for row2 in db:
       if row2 != row:
            X.append([float(row2[0]), float(row2[1])])
    
    
    #transform the original training classes to numbers and add to the vector Y removing the instance that will be used for testing in this iteration. For instance, Y = [1, 2, ,...]. Convert each
    #  feature value to float to avoid warning messages
    #--> add your Python code here
    Y = []
    for row2 in db:
        if row2 != row:
            Y.append(row2[2])
    

    #store the test sample of this iteration in the vector testSample
    testSample = [float(row[0]), float(row[1])]
    #fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #use your test sample in this iteration to make the class prediction. For instance
    class_predicted = clf.predict([testSample])[0]

    #compare the prediction with the true label of the test instance to start calculating the error rate.
    if class_predicted != row[2]:
        error += 1

#print the error rate
print(f"Error rate: {error/10} or {error*10}%")
