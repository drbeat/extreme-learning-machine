#include <stdlib.h>
#include <math.h>
#include "fastmath.h"
#include "extras.h"

/* transformation functions begin */
void sig(double * input, int rows, int columns, double * output){
    /*
     * sigmoid = 1 / (1 + exp( -x ))
     */
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            output[i * columns + j] = 1 / (1 + exp(-input[i * columns + j]));
        }
    }
}

void hlf(double * input, int rows, int columns, double * output){
    /*
     * hlf =	{ 1:		x >= 0
     *				{ 0:		x < 0
     */
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            if ((input[i * columns + j]) >= 0)
                output[i * columns + j] = 1;
            else
                output[i * columns + j] = 0;
        }
    }
}

void rbf(double * input, int rows, int columns, double * output){
    /*
     * rbf = exp( -x² )
     */
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            output[i * columns + j] = exp(-fastpow(input[i * columns + j], 2));
        }
    }
}

void mqf(double * input, int rows, int columns, double * output){
    for (int i = 0; i < rows; i++) {
        /*
         * mqf = sqrt(1 + x²)
         */
        for (int j = 0; j < columns; j++) {
            output[i * columns + j] = fastsqrt(1 + fastpow(input[i * columns + j], 2));
        }
    }
}
/* transformation functions end */

/* ELM begin */
cELM::cELM(){
}

void cELM::cfit(int columns, int numTransformation, int seed){
    // save parameter programwide
    this->columns = columns;
    this->numTransformation = numTransformation;

    // set randomseed
    srand(seed);

    // weights = (double)random (-1 to 1)
    weights = new double[columns * numTransformation];
    for (int i = 0; i < (columns * numTransformation); i++) {
        weights[i] = -1 + 2 * (double)rand() / RAND_MAX;
    }

    // bias = (double)random (0 to 1)
    bias = new double[numTransformation];
    for (int i = 0; i < numTransformation; i++) {
        bias[i] = (double)rand() / RAND_MAX;
    }
}  // cELM::cfit

double * cELM::normaltransform(double * X, int rows, int activation){
    // create matrix preH[rows * numTransformation]
    double * preH = new double[rows * numTransformation];

    // preH = X * weights
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < numTransformation; j++) {
            preH[j + i * numTransformation] = 0;
            for (int k = 0; k < columns; k++) {
                preH[j + i * numTransformation] += X[k + i * columns] * weights[j + k * numTransformation];
            }
        }
    }

    // preH = preH + bias
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < numTransformation; j++) {
            preH[i * numTransformation + j] += bias[j];
        }
    }

    // H = activate(preH)
    double * H = new double[rows * numTransformation];

    if (activation == 0)
        sig(preH, rows, numTransformation, H);
    else if (activation == 1)
        hlf(preH, rows, numTransformation, H);
    else if (activation == 2)
        rbf(preH, rows, numTransformation, H);
    else if (activation == 3)
        mqf(preH, rows, numTransformation, H);
    else {
        for (int i = 0; i < (rows * numTransformation); i++) {
            H[i] = preH[i];
        }
    }


    delete preH;
    return H;
} // cELM::normaltransform

double * cELM::sparsetransform(double * X, int * Xindptr, int * Xind, int rows, int activation){
    // create Xm matrix [rows * columns] filled with 0
    double * Xm =  new double[rows * columns];

    for (int i = 0; i < (rows * columns); i++) {
        Xm[i] = 0;
    }

    // fill Xm with sparse data
    int count = 0;
    int number = 0;
    for (int i = 0; i < rows; i++) {
        int start = Xindptr[count];
        int end = Xindptr[count + 1];
        while (start < end) {
            Xm[i * columns + Xind[number]] = X[number];
            number++;
            start++;
        }
        count++;
    }

    // preH = Xm * weights
    double * preH = new double[rows * numTransformation];
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < numTransformation; j++) {
            preH[j + i * numTransformation] = 0;
            for (int k = 0; k < columns; k++) {
                preH[j + i * numTransformation] += Xm[k + i * columns] * weights[j + k * numTransformation];
            }
        }
    }
    delete Xm;
    // preH = preH + bias
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < numTransformation; j++) {
            preH[i * numTransformation + j] += bias[j];
        }
    }

    // H = activate(preH)
    double * H = new double[rows * numTransformation];

    if (activation == 0)
        sig(preH, rows, numTransformation, H);
    else if (activation == 1)
        hlf(preH, rows, numTransformation, H);
    else if (activation == 2)
        rbf(preH, rows, numTransformation, H);
    else if (activation == 3)
        mqf(preH, rows, numTransformation, H);
    else {
        for (int i = 0; i < (rows * numTransformation); i++) {
            H[i] = preH[i];
        }
    }

    delete preH;
    return H;
} // cELM::sparsetransform
