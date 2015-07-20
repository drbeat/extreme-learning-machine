class cELM{
	public:
		cELM();
		void cfit(int columns, int numTransformation, int seed);
		double* normaltransform(double* X, int columns, int activation);
		double* sparsetransform(double* X, int* Xindptr, int* Xind, int rows, int activation);
	private:
		int columns;
		int numTransformation;
		double* weights;
		double* bias;
};
