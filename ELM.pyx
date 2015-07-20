# distutils: language = c++
# distutils: sources = cELM.cpp

from cython.view cimport array
cimport numpy as np
import numpy as np
from scipy.sparse import csr_matrix

cdef extern from "extras.hpp":
    cdef cppclass cELM:
        cELM()
        int cfit(int, int, int)
        double* normaltransform(double*, int, int)
        double* sparsetransform(double*, int*, int*, int, int)

cdef class ELM:
    cdef cELM *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self):
        self.thisptr = new cELM()
    def __dealloc__(self):
        del self.thisptr
    def fit(self, J, hiddenNeurons=20, seed=0):
        global hn
        hn = hiddenNeurons
        self.thisptr.cfit(J, hiddenNeurons, seed)
    def transform(self, X, kernel='rbf'):
        r = X.shape[0]
        
        if kernel is 'rbf':
            typ = 0
        elif kernel is 'sig':
            typ = 1
        elif kernel is 'hlf':
            typ = 2
        elif kernel is 'mqf':
            typ = 3
        else:
            typ = 4
            
        if isinstance(X, csr_matrix):
            dataX = np.ascontiguousarray(X.data)
            indptrX = np.ascontiguousarray(X.indptr)
            indX = np.ascontiguousarray(X.indices)
            a = np.asarray(<double[:r,:hn]>self.thisptr.sparsetransform(<double*> np.PyArray_DATA(dataX), <int*> np.PyArray_DATA(indptrX), <int*> np.PyArray_DATA(indX), r, typ))
        else:  
            a = np.asarray(<double[:r,:hn]>self.thisptr.normaltransform(<double*> np.PyArray_DATA(X), r, typ))
        return a