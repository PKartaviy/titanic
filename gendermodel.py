""" This simple code is desinged to teach a basic user to read in the files in python, simply find what proportion of males and females survived and make a predictive model based on this
Author : AstroDave
Date : 18th September, 2012

"""


import csv as csv
import numpy as np
import operator
import random as rn 

class Predictor:
    def __init__(self, trainFile, testFile, submitFile):
        self.id = { 'pclass': 1, 'sex': 3, 'age': 4, 'sibsp':  5, 'parch': 6, 'embarked': 10}
        self.max = {'pclass': 3, 'sex': 2, 'age': 8, 'sibsp':  9, 'parch': 10, 'embarked': 3}
            
        self.trainFile = trainFile
        self.testFile = testFile
        self.submitFile = submitFile
            
        self.allTrainData = self.bakeData(np.array( self.readFile(trainFile) ))
            
        self.submitData = self.readFile(testFile)
        for row in self.submitData:
            row.insert(0, 7)
        self.submitData = self.bakeData(np.array( self.submitData ) )
    
    @staticmethod
    def readFile(fileName):
        file_obect = csv.reader(open(fileName, 'rb'))
        header = file_obect.next()

        data = []
        for row in file_obect:
            data.append(row)
        return data
    
    def splitData(self):
        trainData = []
        testData = []
        for row in self.allTrainData:
            if rn.random() < 0.5:
                trainData.append(row)
            else:
                testData.append(row)
        trainData = np.array(trainData)
        testData = np.array(testData)
        return [trainData, testData]
    
    
    def bakeData(self, data):
        rows = np.size(data[0::, 0])
        columns = np.size(data[0, 0::])
        #transform data to equal form
        data[data[0::, self.id['pclass']]=='1', self.id['pclass']] = 0
        data[data[0::, self.id['pclass']]=='2', self.id['pclass']] = 1
        data[data[0::, self.id['pclass']]=='3', self.id['pclass']] = 2
        
        data[data[0::, self.id['sex']]=='male', self.id['sex'] ] = 1
        data[data[0::, self.id['sex']]=='female', self.id['sex'] ] = 0
        
        data[ data[0::, self.id['embarked']]=='S', self.id['embarked'] ] = 0
        data[ data[0::, self.id['embarked']]=='C', self.id['embarked'] ] = 1
        data[ data[0::, self.id['embarked']]=='Q', self.id['embarked'] ] = 2
        data[ data[0::, self.id['embarked']]=='', self.id['embarked'] ] = 0
        
        incorAge = data[0::, self.id['age']]==''
        corAge = data[0::, self.id['age']]!=''
        data[incorAge, self.id['age']] = np.median(data[corAge, self.id['age']].astype(np.float) )
        for k in range(rows):
            data[k, self.id['age']] = int(float(data[k, self.id['age']]))/12
        
        return data
        
    def learnIteration(self, data):    
        self.weights = {}
        number_passengers = np.size(data[0::,0])
        for key in self.id :
            # cast all arrays to float
            dataColumn = data[0::, self.id[key]].astype(np.float)
            self.weights[key] = []
            for value in range(self.max[key]):
                passengersWithValue = dataColumn == value
                numPassengersWithValue = np.size(data[passengersWithValue, self.id[key]])
                survivedPassengersWithValue = np.sum(data[passengersWithValue, 0 ].astype(np.float)) 
                survivingRate = 0.5
                # Ignore statistic which has not enough data
                if(numPassengersWithValue>5):
                    survivingRate = float(survivedPassengersWithValue)/numPassengersWithValue
                self.weights[key].append( survivingRate)

    def predict(self, row):
        #just calc as average sum
        probability = 0.0
        n = 0
        for key in self.id :
            n += 1
            col = self.id[key]
            probability += self.weights[key][ int(row[col]) ]
        probability = probability / n
        
        if probability > 0.5:
            return 1
        else:
            return 0

    def checkPrediction(self, test_data):
        n = 0.0
        correct = 0.0
        for row in test_data:
            n += 1.0
            if self.predict(row) == int(row[0]) :
                correct += 1.0
        
        return correct/n

    def visualize(self):
        for key in self.weights.keys():
            print key, self.weights[key]
        
        print "Prediction rate is", self.predictionRate
    
    def learn(self):
        maxAttempsNum = 10
        attemptNum = 0
        rateSum = 0.0
        for i in range(10):
            [data, test_data] = self.splitData()
            self.learnIteration(data)
        
            rateSum += self.checkPrediction(test_data)
            attemptNum += 1
            
        
        self.predictionRate = rateSum / attemptNum

    def submit(self):
        open_file_object = csv.writer(open(self.submitFile, "wb"))
        for row in self.submitData:
            row[0] = self.predict(row) #Insert the prediciton at the start of the row
        open_file_object.writerow(row) #Write the row to the file

predictor = Predictor("csv/train.csv", "csv/test.csv", "csv/genderbasedmodelpy.csv")
predictor.learn()
predictor.visualize()

predictor.submit()
