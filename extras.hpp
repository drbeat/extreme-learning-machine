class cELM{
	public:
		cELM();
		int cfit(int columns, int hiddenNeurons, int seed);
		double* normaltransform(double* X, int columns, int kernel);
		double* sparsetransform(double* X, int* Xindptr, int* Xind, int rows, int kernel);
	private:
		int columns;
		int hiddenNeurons;
		double* inW;
		double* bias;
};

