## Development of a prototype of an       Extreme Learning Machine

This project is an implementation of the Extreme Learning Machine (ELM) in C++ with an interface in Cython for Python.

### Files

* cELM.cpp | C++ source
* extras.hpp | C++ header
* ELM.pyx | Cython wrapper
* setup.py | Cython compilersetup
* test.py | Testfile with data
* ELM.cpp | Compiled Cythonfile
* ELM.pyd | ELM module for Windows
* ELM.so | ELM module for Linux

### Usage

To use the pre-compiled modules in Python it can be imported.
You can find an example implementation in the *test.py*.

> import ELM
> elm = ELM.ELM()

In the next step the ELM has to be fitted to the data.

> elm.fit(inputNeurons, hiddenNeurons)

After the fitting your data can be transformed.

> elm.transform(X, 'transformation')

The ELM comes with four different functions

* rbf - Gaussian
* sig - Sigmoid
* hlf - Hard-limit
* mqf - Multiquadratic

as described by Huang et. al.

### Compiling

To compile the module after changing the code run

> python setup.py build_ext --inplace
