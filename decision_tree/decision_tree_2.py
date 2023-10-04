#-------------------------------------------------------------------------
# AUTHOR: Rebecca Glatts
# FILENAME: decision_tree_2.py
# SPECIFICATION: Creates decision trees from csv data and tests them 10 times each to find the 
# average accuracy of each of them
# FOR: CS 4210- Assignment #2
# TIME SPENT: 3 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv',
            'contact_lens_training_2.csv',
            'contact_lens_training_3.csv']

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append (row)
                
    #transform the original categorical training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
    # so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    feature_to_value = { 'Young' : 1, 'Prepresbyopic' : 2, 'Presbyopic' : 3, 
                     'Myope' : 4, 'Hypermetrope' : 5,
                     'No' : 6, 'Yes' : 7, 
                     'Normal': 8, 'Reduced' : 9}

    for row in dbTraining:
        a = []
        for i in range (4):
            a.append(feature_to_value.get(row[i]))
        X.append(a)

    #transform the original categorical training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    class_to_value = {'Yes' : 1, 'No' : 2}
    for row in dbTraining:
        Y.append(class_to_value.get(row[4]))

    
    accuracy_list = [0]*10
    #loop your training and test tasks 10 times here
    for i in range (10):

        #fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
        clf = clf.fit(X, Y)

        #read the test data and add this data to dbTest
        #--> add your Python code here
        dbTest = []
        with open('contact_lens_test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for j, row in enumerate(reader):
                if j > 0: #skipping the header
                    dbTest.append(row)

        #for data in dbTest:
           #transform the features of the test instances to numbers following the same strategy done during training,
           #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
           #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
        
        correct = 0
        for data in (dbTest):
            class_predicted = clf.predict([[feature_to_value[item] for item in data[:4]]])[0]

           #compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            if class_to_value.get(data[4]) == class_predicted:
                correct += 1
        accuracy = correct / 8
        accuracy_list[i] = accuracy
    #find the average of this model during the 10 runs (training and test set)
    average_accuracy = sum(accuracy_list)/10
    #print the average accuracy of this model during the 10 runs (training and test set).
    #your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    print(f"Final accuracy when training on {ds}: {average_accuracy}")

