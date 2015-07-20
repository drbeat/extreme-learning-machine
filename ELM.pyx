# distutils: language = c++
# distutils: sources = cELM.cpp

from cython.view cimport array
cimport numpy as np
import numpy as np
from scipy.sparse import csr_matrix

cdef extern from "extras.h":
    cdef cppclass cELM:
        cELM()
        void cfit(int, int, int)
        double* normaltransform(double*, int, int)
        double* sparsetransform(double*, int*, int*, int, int)

cdef class ELM:
    cdef cELM *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self):
        self.thisptr = new cELM()
    def __dealloc__(self):
        del self.thisptr
    def fit(self, columns, numTransformation=20, seed=0):
        global numT
        numT = numTransformation
        self.thisptr.cfit(columns, numTransformation, seed)
    def transform(self, X, activation='sig'):
        rows = X.shape[0]

        if activation is 'sig':
            typ = 0
        elif activation is 'hlf':
            typ = 1
        elif activation is 'rbf':
            typ = 2
        elif activation is 'mqf':
            typ = 3
        else:
            typ = 4

        if isinstance(X, csr_matrix):
            dataX = np.ascontiguousarray(X.data)
            indptrX = np.ascontiguousarray(X.indptr)
            indX = np.ascontiguousarray(X.indices)
            a = np.asarray(<double[:rows,:numT]>self.thisptr.sparsetransform(<double*> np.PyArray_DATA(dataX), <int*> np.PyArray_DATA(indptrX), <int*> np.PyArray_DATA(indX), rows, typ))
        else:
            a = np.asarray(<double[:rows,:numT]>self.thisptr.normaltransform(<double*> np.PyArray_DATA(X), rows, typ))
        return a
