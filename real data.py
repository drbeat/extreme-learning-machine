import pickle
import time
import ELM
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

print("load data")
Xtrain, Xtrain_sparse, Xtest, Xtest_sparse, ytrain, ytest = pickle.load(open("data.p", "rb"))

print("fit elm")
elm = ELM.ELM()
elm.fit(12, 20, 0)
print("start")
#----------------------------------reg 
transform=['reg','sig','hlf','rbf','mqf']

for act in transform:
    print("Zeit:" + str(time.clock()))
    
    Xtrans = elm.transform(Xtrain, act)
    logreg = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6).fit(Xtrans, ytrain)
    
    ypred = logreg.predict_proba(elm.transform(Xtest, act))[:,1]
    rmse = mean_squared_error(ytest, ypred)
    print(act + ': ' + str(rmse))