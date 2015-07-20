import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import sklearn.datasets
import scipy.stats

import ELM

from sklearn import linear_model

#Datensatz erstellen
#X, y = sklearn.datasets.make_classification(n_samples=100000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=2)
X, y = sklearn.datasets.make_gaussian_quantiles(n_samples=100000, n_classes=2)
#X, y = sklearn.datasets.make_circles(n_samples=100000, noise=0.05)
#X, y = sklearn.datasets.make_moons(n_samples=100000, noise=0.05)

Xtrain, ytrain = X[:99000], y[:99000]
#Xtrain2, ytrain2 = csr_matrix(Xtrain), csr_matrix(ytrain)
Xtest, ytest = X[99000:], y[99000:]
#Xtrain, ytrain = X[:10], y[:10]
#Xtest, ytest = X[:10], y[:10]

#Fitte ELM and transformiere Inputdaten
elm = ELM.ELM()
elm.fit(2, 30, 0)

#Erstelle Netz (fuer heatmaps)
h = .02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

#----------------------------------Zeichne Heat Map-----------------------------------------

#Erstelle und trainiere logistische Regression auf Originaldaten
logreg = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6).fit(Xtrain, ytrain)

grid_data = np.c_[xx.ravel(), yy.ravel()]
Z = logreg.predict_proba(grid_data)[:,1]
Z = Z.reshape(xx.shape)

#Erstelle Prognosen auf Trainingsset
yhat = logreg.predict_proba(Xtest)[:,1]
r = scipy.stats.pearsonr(yhat, ytest)[0]

plt.imshow(Z, extent=[xx.min(), xx.max(), yy.max(), yy.min()])

#Plotte Trainingspunkte
plt.plot(Xtest[ytest==0, 0], Xtest[ytest==0, 1], 'co')
plt.plot(Xtest[ytest==1, 0], Xtest[ytest==1, 1], 'ro')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title('Logistische Regression, heat map, r=' + str(r))

plt.show()

#-------------------------------------Sigmoid

#Erstelle und trainiere logistische Regression auf transformierte Daten

Xtrans = elm.transform(Xtrain, 'sig')
logreg = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6).fit(Xtrans, ytrain)

grid_data = elm.transform(np.c_[xx.ravel(), yy.ravel()], 'sig')
Z = logreg.predict_proba(grid_data)[:,1]
Z = Z.reshape(xx.shape)

#Erstelle Prognosen auf Trainingsset
yhat = logreg.predict_proba(elm.transform(Xtest, 'sig'))[:,1]
r = scipy.stats.pearsonr(yhat, ytest)[0]

plt.imshow(Z, extent=[xx.min(), xx.max(), yy.max(), yy.min()])

#Plotte Trainingspunkte
plt.plot(Xtest[ytest==0, 0], Xtest[ytest==0, 1], 'co')
plt.plot(Xtest[ytest==1, 0], Xtest[ytest==1, 1], 'ro')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title('ELM, heat map, r=' + str(r))

plt.show()

#-------------------------------------RBF

Xtrans = elm.transform(Xtrain, 'rbf')
logreg = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6).fit(Xtrans, ytrain)

grid_data = elm.transform(np.c_[xx.ravel(), yy.ravel()], 'rbf')
Z = logreg.predict_proba(grid_data)[:,1]
Z = Z.reshape(xx.shape)

#Erstelle Prognosen auf Trainingsset
yhat = logreg.predict_proba(elm.transform(Xtest, 'rbf'))[:,1]
r = scipy.stats.pearsonr(yhat, ytest)[0]

plt.imshow(Z, extent=[xx.min(), xx.max(), yy.max(), yy.min()])

#Plotte Trainingspunkte
plt.plot(Xtest[ytest==0, 0], Xtest[ytest==0, 1], 'co')
plt.plot(Xtest[ytest==1, 0], Xtest[ytest==1, 1], 'ro')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title('ELM, heat map, r=' + str(r))

plt.show()

#--------------------------------HLF

Xtrans = elm.transform(Xtrain, 'hlf')
logreg = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6).fit(Xtrans, ytrain)

grid_data = elm.transform(np.c_[xx.ravel(), yy.ravel()], 'hlf')
Z = logreg.predict_proba(grid_data)[:,1]
Z = Z.reshape(xx.shape)

#Erstelle Prognosen auf Trainingsset
yhat = logreg.predict_proba(elm.transform(Xtest, 'hlf'))[:,1]
r = scipy.stats.pearsonr(yhat, ytest)[0]

plt.imshow(Z, extent=[xx.min(), xx.max(), yy.max(), yy.min()])

#Plotte Trainingspunkte
plt.plot(Xtest[ytest==0, 0], Xtest[ytest==0, 1], 'co')
plt.plot(Xtest[ytest==1, 0], Xtest[ytest==1, 1], 'ro')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title('ELM, heat map, r=' + str(r))

plt.show()

#--------------------------------MQF

Xtrans = elm.transform(Xtrain, 'mqf')
logreg = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6).fit(Xtrans, ytrain)

grid_data = elm.transform(np.c_[xx.ravel(), yy.ravel()], 'mqf')
Z = logreg.predict_proba(grid_data)[:,1]
Z = Z.reshape(xx.shape)

#Erstelle Prognosen auf Trainingsset
yhat = logreg.predict_proba(elm.transform(Xtest, 'mqf'))[:,1]
r = scipy.stats.pearsonr(yhat, ytest)[0]

plt.imshow(Z, extent=[xx.min(), xx.max(), yy.max(), yy.min()])

#Plotte Trainingspunkte
plt.plot(Xtest[ytest==0, 0], Xtest[ytest==0, 1], 'co')
plt.plot(Xtest[ytest==1, 0], Xtest[ytest==1, 1], 'ro')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title('ELM, heat map, r=' + str(r))

plt.show()