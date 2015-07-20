#include <stdlib.h>
#include <math.h>
#include "extras.hpp"

/* matrices helper functions begin */
void matrixmult(double* inputA, int rowsA, int rowcol, double* inputB, int columnsB, double* outputC){
	// inputA[rowsA*rowcol]
	// inputB[rowcol*columnsB]
	// outputC[rowsA*columnsB]
	for(int i = 0; i < rowsA; i++){
		for(int j = 0; j < columnsB; j++){
			outputC[j + i * columnsB] = 0;
			for(int k = 0; k < rowcol; k++){
				outputC[j + i * columnsB] += inputA[k + i * rowcol] * inputB[j + k * columnsB];
			}
		}
	}
}
/* matrices helper functions end */

/* transformation functions begin */
void sig(double* input, int rows, int columns, double* output, double* b){
    for(int i = 0; i < rows; i++){
		for(int j = 0; j < columns; j++)
        	output[i * columns + j] = 1 / (1 + exp(-(input[i * columns + j] + b[j])));
    }
}

void hlf(double* input, int rows, int columns, double* output, double* b){
    for(int i = 0; i < rows; i++){
    	for(int j = 0; j < columns; j++){
	        if((input[i * columns + j] - b[j]) >= 0)
	    		output[i * columns + j] = 1;
	    	else
	    		output[i * columns + j] = 0;
	    }
    }
}

void rbf(double* input, int rows, int columns, double* output, double* b){
	//double center = input[rand()%(rows*columns)]; 
    for(int i = 0; i < rows; i++){
    	for(int j = 0; j < columns; j++)
        	output[i * columns + j] = exp(-b[j] * pow(input[i * columns + j],2));
    }
}

void mqf(double* input, int rows, int columns, double* output, double* b){
    for(int i = 0; i < rows; i++){
    	for(int j = 0; j < columns; j++)
        	output[i * columns + j] = sqrt(pow(input[i * columns + j],2) + pow(b[j],2));
    }
}
/* transformation functions end */

/* fit function begin */
cELM::cELM(){	
}

int cELM::cfit(int columns, int hiddenNeurons, int seed){
	this->columns = columns;
	this->hiddenNeurons = hiddenNeurons;
	
	srand(seed);
	
	// inW = random (-1 to 1)
	inW = new double[columns * hiddenNeurons];
	for(int i = 0; i < (columns * hiddenNeurons); i++){
		inW[i] = -1+2*(double) rand() / RAND_MAX;
	}

	// bias = random (0 to 1)
	bias = new double[hiddenNeurons];
	for(int i = 0; i < hiddenNeurons; i++){
		bias[i] = (double) rand() / RAND_MAX;
	}

	return 0;
}

double* cELM::normaltransform(double* X, int rows, int kernel){
	
	double* preH = new double[rows * hiddenNeurons];
	// preH = X * inW
    matrixmult(X, rows, columns, inW, hiddenNeurons, preH);
	
	// H = activate(preH)
	double* H = new double[rows * hiddenNeurons];
	switch(kernel){
		case 0: 
			rbf(preH, rows, hiddenNeurons, H, bias);
			break;
		case 1:
			sig(preH, rows, hiddenNeurons, H, bias);
			break;
		case 2:
			hlf(preH, rows, hiddenNeurons, H, bias);
			break;
		case 3:
			mqf(preH, rows, hiddenNeurons, H, bias);
			break;
		default:
			for(int i = 0; i < (rows * hiddenNeurons); i++)
				H[i] = preH[i];
			break;
	}
	
	delete preH;
	return H;
}

double* cELM::sparsetransform(double* X, int* Xindptr, int* Xind, int rows, int kernel){
	
	// create Xm array filled with 0
	double* Xm =  new double[rows * columns];
	for(int i = 0; i < (rows * columns); i++){
		Xm[i] = 0;
	}
	
	// fill Xm with sparse data
	int count = 0;
	int number = 0;
	for(int i = 0; i < rows; i++){
		while(Xindptr[count] < Xindptr[count + 1]){
			Xm[i * columns + Xind[number]] = X[number];
			number++;
			Xindptr[count]++;
		}
		count++;
	}
	
	// preH = Xm * inW
	double* preH = new double[rows * hiddenNeurons];
    matrixmult(Xm, rows, columns, inW, hiddenNeurons, preH);
    delete Xm;
	
	// H = activate(preH)
	double* H = new double[rows * hiddenNeurons];
	switch(kernel){
		case 0: 
			rbf(preH, rows, hiddenNeurons, H, bias);
			break;
		case 1:
			sig(preH, rows, hiddenNeurons, H, bias);
			break;
		case 2:
			hlf(preH, rows, hiddenNeurons, H, bias);
			break;
		case 3:
			mqf(preH, rows, hiddenNeurons, H, bias);
			break;
		default:
			for(int i = 0; i < (rows * hiddenNeurons); i++)
				H[i] = preH[i];
			break;
	}
	
	delete preH;
	return H;
}
