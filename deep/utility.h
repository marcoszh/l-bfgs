#include <armadillo>
#include <math.h>
#include <iostream>

using namespace std;
using namespace arma;

//Initialize parameters randomly based on layer sizes.
colvec initializeParameters(int hiddenSize, int visibleSize)
{
	double r = sqrt((double)6)/sqrt((double)hiddenSize+visibleSize+1);
	mat W1(hiddenSize,visibleSize,fill::randu);
	mat W2(visibleSize,hiddenSize,fill::randu);
	W1=W1*2*r-r;
	W2=W2*2*r-r;

	colvec b1(hiddenSize,fill::zeros);
	colvec b2(visibleSize,fill::zeros);

	//Convert weights and bias gradients to the vector form.
	mat W11 = mat(W1);
	mat W22 = mat(W2);
	W11.reshape(W11.n_rows*W11.n_cols,1);
	W22.reshape(W22.n_rows*W22.n_cols,1);

	colvec theta=join_vert(W11,W22);
	theta = join_vert(theta,b1);
	theta = join_vert(theta,b2);
	return theta;
}

mat sigmoid(mat & a)
{
	colvec g= 1.0 / (1.0+exp(-a));
	return g;
}

colvec sigmoidGradient(colvec z)
{
	colvec g= colvec(z.n_rows,fill::zeros);
	colvec t=colvec(z.n_rows,fill::ones);
	g =sigmoid(z)%(t-sigmoid(z));
	return g;
}