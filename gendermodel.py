""" This simple code is desinged to teach a basic user to read in the files in python, simply find what proportion of males and females survived and make a predictive model based on this
Author : AstroDave
Date : 18th September, 2012

"""


import csv as csv
import numpy as np
import operator

id = { 'pclass': 1, 'sex': 3, 'age': 4, 'sibsp':  5, 'parch': 6, 'embarked': 10}
max = {'pclass': 3, 'sex': 2, 'age': 8, 'sibsp':  9, 'parch': 10, 'embarked': 3}

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

    for key in weights.keys():
        print key, weights[key]
    
    return weights

def calcProbability(row, weights):
    #just calc as average sum
    probability = 0.0
    n = 0
    for key in id :
        n += 1
        col = id[key]
        probability += weights[key][ int(row[col]) ]
    probability = probability / n
    
    return probability

csv_file_object = csv.reader(open('csv/train.csv', 'rb')) #Load in the csv file
header = csv_file_object.next() #Skip the fist line as it is a header
data=[] #Creat a variable called 'data'
for row in csv_file_object: #Skip through each row in the csv file
    data.append(row) #adding each row to the data variable
data = np.array(data) #Then convert from a list to an array

#Now I have an array of 11 columns and 891 rows
#I can access any element I want so the entire first column would
#be data[0::,0].astype(np.flaot) This means all of the columen and column 0
#I have to add the astype command
#as when reading in it thought it was  a string so needed to convert

bakeData(data)
weights = calcEntropy(data)

number_passengers = np.size(data[0::,0].astype(np.float))
number_survived = np.sum(data[0::,0].astype(np.float))
proportion_survivors = number_passengers / number_survived

# I can now find the stats of all the women on board
women_only_stats = data[0::,3] == '0' #This finds where all the women are
men_only_stats = data[0::,3] != '0' #This finds where all the men are
                                         # != means not equal

#I can now find for example the ages of all the women by just placing
#women_only_stats in the '0::' part of the array index. You can test it by
#placing it in the 4 column and it should all read 'female'

women_onboard = data[women_only_stats,0].astype(np.float)
men_onboard = data[men_only_stats,0].astype(np.float)

proportion_women_survived = np.sum(women_onboard) / np.size(women_onboard)
proportion_men_survived = np.sum(men_onboard) / np.size(men_onboard)

print 'Proportion of women who survived is %s' % proportion_women_survived
print 'Proportion of men who survived is %s' % proportion_men_survived

#Now I have my indicator I can read in the test file and write out
#if a women then survived(1) if a man then did not survived (0)
#1st Read in test
test_file_obect = csv.reader(open('csv/test.csv', 'rb'))
header = test_file_obect.next()

#Now also open the a new file so we can write to it call it something
#descriptive

open_file_object = csv.writer(open("csv/genderbasedmodelpy.csv", "wb"))

test_data = []
for row in test_file_obect:
    row.insert(0,'7')
    test_data.append(row)
test_data = np.array(test_data)
bakeData(test_data)

for row in test_data:
    if calcProbability(row, weights) > 0.5 :
        row[0] = '1' #Insert the prediciton at the start of the row
        open_file_object.writerow(row) #Write the row to the file
    else:
        row[0] = '0'
        open_file_object.writerow(row)

