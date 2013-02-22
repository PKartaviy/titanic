""" This simple code is desinged to teach a basic user to read in the files in python, simply find what proportion of males and females survived and make a predictive model based on this
Author : AstroDave
Date : 18th September, 2012

"""


import csv as csv
import numpy as np
import operator
import random as rn 

id = { 'pclass': 1, 'sex': 3, 'age': 4, 'sibsp':  5, 'parch': 6, 'embarked': 10}
max = {'pclass': 3, 'sex': 2, 'age': 8, 'sibsp':  9, 'parch': 10, 'embarked': 3}

def splitData():
    csv_file_object = csv.reader(open('csv/train.csv', 'rb')) #Load in the csv file
    header = csv_file_object.next() #Skip the fist line as it is a header
    data=[] #Creat a variable called 'data'
    test_data = []
    for row in csv_file_object: #Skip through each row in the csv file
        if rn.random() < 0.5:
            data.append(row) #adding each row to the data variable
        else:
            test_data.append(row)
            
    data = np.array(data) #Then convert from a list to an array
    test_data = np.array(test_data)
    bakeData(data)
    bakeData(test_data)

    return [data, test_data]

def getSubmitData():
    test_file_obect = csv.reader(open('csv/test.csv', 'rb'))
    header = test_file_obect.next()

    submit_data = []
    for row in test_file_obect:
        row.insert(0,'7')
        submit_data.append(row)
    submit_data = np.array(submit_data)
    bakeData(submit_data)
    
    return submit_data
    
def bakeData(data):
    rows = np.size(data[0::, 0])
    columns = np.size(data[0, 0::])
    #transform data to equal form
    data[data[0::, id['pclass']]=='1', id['pclass']] = 0
    data[data[0::, id['pclass']]=='2', id['pclass']] = 1
    data[data[0::, id['pclass']]=='3', id['pclass']] = 2
    
    data[data[0::, id['sex']]=='male', id['sex'] ] = 1
    data[data[0::, id['sex']]=='female', id['sex'] ] = 0
    
    data[ data[0::, id['embarked']]=='S', id['embarked'] ] = 0
    data[ data[0::, id['embarked']]=='C', id['embarked'] ] = 1
    data[ data[0::, id['embarked']]=='Q', id['embarked'] ] = 2
    data[ data[0::, id['embarked']]=='', id['embarked'] ] = 0
    
    incorAge = data[0::, id['age']]==''
    corAge = data[0::, id['age']]!=''
    data[incorAge, id['age']] = np.median(data[corAge, id['age']].astype(np.float) )
    for k in range(rows):
        data[k, id['age']] = int(float(data[k, id['age']]))/12
    
def calcEntropy(data):    
    weights = {}
    number_passengers = np.size(data[0::,0])
    for key in id :
        # cast all arrays to float
        dataColumn = data[0::, id[key]].astype(np.float)
        weights[key] = []
        for value in range(max[key]):
            passengersWithValue = dataColumn == value
            numPassengersWithValue = np.size(data[passengersWithValue, id[key]])
            survivedPassengersWithValue = np.sum(data[passengersWithValue, 0 ].astype(np.float)) 
            survivingRate = 0.5
            # Ignore statistic which has not enough data
            if(numPassengersWithValue>5):
                survivingRate = float(survivedPassengersWithValue)/numPassengersWithValue
            weights[key].append( survivingRate)
    
    return weights

def predict(row, weights):
    #just calc as average sum
    probability = 0.0
    n = 0
    for key in id :
        n += 1
        col = id[key]
        probability += weights[key][ int(row[col]) ]
    probability = probability / n
    
    if probability > 0.5:
        return 1
    else:
        return 0

def checkPrediction(weights, test_data):
    n = 0.0
    correct = 0.0
    for row in test_data:
        n += 1.0
        if predict(row, weights) == int(row[0]) :
            correct += 1.0
    
    return correct/n

def visualizeWeights(weights):
    for key in weights.keys():
        print key, weights[key]
        
attemptNum = 0
rateSum = 0.0
for i in range(10):
    [data, test_data] = splitData()
    weights = calcEntropy(data)

    rateSum += checkPrediction(weights, test_data)
    attemptNum += 1
print "Prediction rate on test_data is ", rateSum / attemptNum

visualizeWeights(weights)
#Now also open the a new file so we can write to it call it something
#descriptive

submit_data = getSubmitData()
open_file_object = csv.writer(open("csv/genderbasedmodelpy.csv", "wb"))
for row in submit_data:
    row[0] = predict(row, weights) #Insert the prediciton at the start of the row
    open_file_object.writerow(row) #Write the row to the file
    

