import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import ELM
import scipy.stats
from sklearn import linear_model

X, y = sklearn.datasets.make_classification(n_samples=100000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=2)
#X, y = sklearn.datasets.make_gaussian_quantiles(n_samples=100000, n_classes=2)
#X, y = sklearn.datasets.make_circles(n_samples=100000, noise=0.05)
#X, y = sklearn.datasets.make_moons(n_samples=100000, noise=0.05)

Xtrain, ytrain = X[:99000], y[:99000]
Xtest, ytest = X[99000:], y[99000:]

#fit elm
elm = ELM.ELM()
elm.fit(2, 20, 0)

#grid
h = .02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

#-------------------------------------REG

transform=['reg','sig','hlf','rbf','mqf']

for act in transform:
    Xtrans = elm.transform(Xtrain, act)
    logreg = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6).fit(Xtrans, ytrain)
    
    grid_data = elm.transform(np.c_[xx.ravel(), yy.ravel()], act)
    Z = logreg.predict_proba(grid_data)[:,1]
    Z = Z.reshape(xx.shape)
    
    yhat = logreg.predict_proba(elm.transform(Xtest, act))[:,1]
    r = scipy.stats.pearsonr(yhat, ytest)[0]
    
    plt.imshow(Z, extent=[xx.min(), xx.max(), yy.max(), yy.min()])
    
    plt.plot(Xtest[ytest==0, 0], Xtest[ytest==0, 1], 'co')
    plt.plot(Xtest[ytest==1, 0], Xtest[ytest==1, 1], 'ro')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title('ELM, '+ act +' function, r=' + str(r))
    
    plt.show()