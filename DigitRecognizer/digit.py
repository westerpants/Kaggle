import numpy as np
from sklearn import svm
from sklearn import linear_model
import csv

def myInt(myList):
    return map(int, myList)

traindata=[]
testdata=[]

with open('train.csv', 'rb') as csvfile:
  data_tmp = csv.reader(csvfile, delimiter=',', quotechar='"')
  for row in data_tmp:
    traindata.append(row)

with open('test.csv', 'rb') as csvfile:
  data_tmp = csv.reader(csvfile, delimiter=',', quotechar='"')
  for row in data_tmp:
    testdata.append(row)

del traindata[0]
del testdata[0]

traindata=map(myInt, traindata)
testdata=map(myInt, testdata)

traindata=np.array(traindata)
testdata=np.array(testdata)

train_y=traindata[:,0]
train_x=traindata[:,1:785]

clf = linear_model.SGDClassifier(loss='log')
clf.fit(train_x, train_y)  

result=clf.predict(testdata)
resultFile = open('result.csv', 'w')
resultList = result.tolist()
resultFile.write('ImageId,Label\n')
for idx, val in enumerate(resultList):
  resultFile.write(str(idx+1)+','+str(val)+'\n')