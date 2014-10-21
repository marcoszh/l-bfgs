#include <iostream>

#include <armadillo>
#include "utility.h"
#include "MNIST.h"
#include "lbfgs.h"

using namespace arma;
using namespace std;

const string trainsetName="train-images-idx3-ubyte";
const string trainlabelName="train-labels-idx1-ubyte";
const string testsetName="t10k-images-idx3-ubyte";
const string testlabelName="t10k-labels-idx1-ubyte";

const int trainNum=60000;
const int testNum=10000;

const int inputSize =28*28;
const int numClasses =10;
const int hiddenSizeL1=200;
const int hiddenSizeL2=200;
const int visibleSize=28*28;
const double sparsityParam = 0.1;//desired average activation of the hidden units.
const double lambda=3e-3;//weight decay parameter
const double beta = 3.0; //weight of sparsity penalty term

mat imagesT;
colvec labelsT(trainNum);
int hiddensize;
int visiblesize;
int memsize;
mat* data;
double func2(colvec& x,colvec& g,size_t N)
{
	double dot0=0;
	dot0=dot(x,x);
	g=2*x;
	return dot0;
}

double sparseCost(colvec & theta, colvec & g, size_t N)
{
	mat W1 = reshape(theta.rows(0,hiddensize*visiblesize-1),hiddensize,visiblesize);
	mat W2 = reshape(theta.rows(hiddensize*visiblesize,2*hiddensize*visiblesize-1),visiblesize,hiddensize);
	colvec b1 = theta.rows(2*hiddensize*visiblesize,2*hiddensize*visiblesize+hiddensize-1);
	colvec b2 = theta.rows(2*hiddensize*visiblesize+hiddensize,theta.n_rows-1);

	double cost = 0;

	colvec W1grad = mat(W1.n_rows,W1.n_cols,fill::zeros);
	colvec W2grad = mat(W2.n_rows,W2.n_cols,fill::zeros);
	colvec b1grad = mat(b1.n_rows,b1.n_cols,fill::zeros);
	colvec b2grad = mat(b2.n_rows,b2.n_cols,fill::zeros);

	int m=imagesT.n_cols;

	//forward propagation
	mat Z2 = W1*(*data)+b1;
	mat A2 = sigmoid(Z2);
	mat Z3 = W2*A2+b2;
	mat A3 = sigmoid(Z3);

	mat diff = A3-(*data);
	double p = sparsityParam;
	colvec pHat = 1.0/m * sum(A2,1);

	double squareError = 0.0;
	for(int i=0;i<m;i++)
	{
		squareError = squareError + ((mat)(trans(diff.row(i))*diff.row(i)))(0,0);
	}

	cost = 0.5 / m * squareError + lambda/2 *(sum(sum((W1%W1)))+sum(sum(W2%W2)))
		+beta*sum(p*log(p/pHat)+(1-p)*log((1-p)/(1-pHat)));

	//square error term
	mat D3=-1 * ((*data)-A3)%sigmoidGradient(Z3);
	colvec KL = beta * (-p/pHat+(1-p)/(1-pHat));
	mat D2= (W2.t() * D3 +KL)% sigmoidGradient(Z2);
	mat W2g=D3*A2.t();
	mat W1g=D2*(*data).t();

	W1grad = 1/m * W1g +lambda * W1;
	W2grad = 1/m * W2g + lambda*W2;
	b1grad = 1/m * (D2*ones(m,1));
	b2grad = 1/m *(D3*ones(m,1));

	W1grad.reshape(W1.n_rows*W1.n_cols,1);
	W2grad.reshape(W2.n_rows*W2.n_cols,1);

	g = join_vert(W1grad,W2grad);
	g = join_vert(g, b1grad);
	g= join_vert(g,b2grad);

	return cost;
}


int main(int argc, char** argv)
{
  cout << "Armadillo version: " << arma_version::as_string() << endl;
  
  //load data
  //read train images
  read_Mnist(trainsetName, imagesT);
  cout<<"Image number: "<<imagesT.n_cols<<endl;
  cout<<"Image size: "<<imagesT.n_rows<<endl;
  //read train labels
  read_Mnist_Label(trainlabelName,labelsT);
  cout<<"Label number: "<<labelsT.size()<<endl;
  //cout<<labelsT(0)<<" "<<labelsT(1)<<endl;

  //train the first sparse autocoder
  cout<<"train first sae"<<endl;
  colvec sae1Theta = initializeParameters(hiddenSizeL1,inputSize);
  data = &imagesT;
  hiddensize=hiddenSizeL1;
  visiblesize=inputSize;
  memsize=visiblesize*hiddensize*2+visiblesize+hiddensize;
  LBFGS lbfgs(&func2,sae1Theta,inputSize,memsize);
  lbfgs.set_max_iter(400);
  lbfgs.set_iprint(1);
  lbfgs.run();
  sae1Theta=lbfgs.get_ax();
  cout << "final result:\n";
  cout << "f: " << lbfgs.get_f() << endl;
  cout << "rms: " << lbfgs.get_rms() << endl;
  cout << "nfev: " << lbfgs.get_nfev() <<endl;
  cout << "success: " << lbfgs.success() << endl;


  getchar();
  return 0;
}

