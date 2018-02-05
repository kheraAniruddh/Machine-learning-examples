import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

import matplotlib.pyplot as plt


#Read dataset file using dataframe
dataset = pd.read_csv("spam.csv",header=None)


#Normalize data
scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(dataset)
dataset.loc[:,:] = scaled_values


#Partition training & test
trainingdata =  dataset.head(3000)
testdata = dataset.tail(1601)


# Apply SVM poly kernel model
# Xvalidate 10 folds with varying k and d = 1,2,3,4
for d in range(1,5):
 arr = []
 for k in range(0,16):
    clf = SVC(kernel='poly', degree=d, C=pow(2,k))
    scores = cross_val_score(clf, trainingdata[trainingdata.columns[:57]],trainingdata[trainingdata.columns[57]], cv=10)
    tup = (round(1-scores.mean(),3), round(scores.std(),3))
    arr.append(tup)
 df = pd.DataFrame(arr)
 df.to_csv("plot_"+ str(d) +".csv")

#Plot validation erros vs. log C (base 2)
d1 =  pd.read_csv("plot_1.csv",skiprows=[0], header=None)
plt.errorbar(d1[int(d1.columns[0])],d1[int(d1.columns[1])],d1[int(d1.columns[2])],label="d=1")

d2 =  pd.read_csv("plot_2.csv",skiprows=[0], header=None)
plt.errorbar(d2[int(d2.columns[0])],d2[int(d2.columns[1])],d2[int(d2.columns[2])],label="d=2")

d3 =  pd.read_csv("plot_3.csv",skiprows=[0], header=None)
plt.errorbar(d3[int(d3.columns[0])],d3[int(d3.columns[1])],d3[int(d3.columns[2])], label="d=3")

d4 =  pd.read_csv("plot_4.csv",skiprows=[0], header=None)
plt.errorbar(d4[int(d4.columns[0])],d4[int(d4.columns[1])],d4[int(d4.columns[2])], label="d=4")

plt.show()


# Store validation error for C*=11 as a function of degree 1 to 4 & get no of support vectors
valarr=[]
for d in range(1,5):
    fixedclf = SVC(kernel='poly', degree=d, C=pow(2,11))
    scores = cross_val_score(fixedclf, trainingdata[trainingdata.columns[:57]],trainingdata[trainingdata.columns[57]], cv=10)
    tup = (d, round(1-scores.mean(),3,))
    valarr.append(tup)
valdf = pd.DataFrame(valarr)


# store test error for C*=11 as a function of degree 1 to 4 & get no of support vectors
testarr=[]
for d in range(1,5):
    fixedclf = SVC(kernel='poly', degree=d, C=pow(2,11))
    fixedclf.fit(trainingdata[trainingdata.columns[:57]],trainingdata[trainingdata.columns[57]])
    predicts =fixedclf.predict(testdata[testdata.columns[:57]])
    scores = accuracy_score(testdata[testdata.columns[57]], predicts)
    tup = (d, round(1-scores.mean(),3))
    testarr.append(tup)
testdf = pd.DataFrame(testarr)

# for d in range(1,5):
#     fixedclf = SVC(kernel='poly', degree=d, C=11)
#     fixedclf.fit(trainingdata[trainingdata.columns[:57]],trainingdata[trainingdata.columns[57]])
#     print(fixedclf.support_vectors_)

#Plot the test errors & validation errors as function of degree of polynomial kernel
# plt.plot(testdf[int(testdf.columns[0])], testdf[int(testdf.columns[1])], label="Test errors")
# plt.plot(valdf[int(valdf.columns[0])], valdf[int(valdf.columns[1])], label="Validation errors")
# plt.show()
