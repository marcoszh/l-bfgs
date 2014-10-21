/*
 * ----------------------------------------------------------------------------
 * "THE BEER-WARE LICENSE" (Revision 42):
 * <js850@cam.ac.uk> wrote this file. As long as you retain this notice you
 * can do whatever you want with this stuff. If we meet some day, and you think
 * this stuff is worth it, you can buy me a beer in return Jacob Stevenson
 * ----------------------------------------------------------------------------
 */

#include "lbfgs.h"
#include <vector>
#include <math.h>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <assert.h>


/**
 * compute the L2 norm of a vector
 */
double vecnorm(colvec v)
{
	return sqrt(dot(v,v));
}

LBFGS::LBFGS(
    double (*func)(colvec &, colvec &, size_t), 
    colvec & x0, 
    size_t N, 
    int M
    //double tol,
    //double maxstep,
    //double max_f_rise,
    //double H0,
    //int maxiter
    )
  :
    afunc_f_grad_(func),
    M_(M),
    tol_(1e-4),
    maxstep_(0.2),
    max_f_rise_(1e-4),
    maxiter_(1000),
    iprint_(-1),
    iter_number_(0),
    nfev_(0),
    H0_(0.1),
    k_(0)
{
  // set the precision of the printing
  cout << std::setprecision(12);

  // allocate arrays
  ax_=colvec(N);
  ag_=colvec(N);

  ay_=mat(M_,N,fill::ones);
  as_=mat(M_,N,fill::ones);
  arho_ =colvec(M_);
  astep_=colvec(N);
  ax_=x0;
  compute_func_gradient(ax_, f_, ag_);
  rms_ = vecnorm(ag_) / sqrt((double)N);
}


/**
 * Do one iteration iteration of the optimization algorithm
 */
void LBFGS::one_iteration()
{
  //std::vector<double> x_old = x_;
  colvec ax_old=ax_;
  //std::vector<double> g_old = g_;
  colvec ag_old=ag_;

  compute_lbfgs_step();

  double stepsize = backtracking_linesearch();

  //update_memory(x_old, g_old, x_, g_);
  update_memory(ax_old,ag_old,ax_,ag_);
  if ((iprint_ > 0) && (iter_number_ % iprint_ == 0)){
    cout << "lbgs: " << iter_number_ 
      << " f " << f_ 
      << " rms " << rms_
      << " stepsize " << stepsize << "\n";
  }
  iter_number_ += 1;
}

void LBFGS::run()
{
  // iterate until the stop criterion is satisfied or maximum number of
  // iterations is reached
  while (iter_number_ < maxiter_)
  {
    if (stop_criterion_satisfied()){
      break;
    }
    one_iteration();
  }
}


void LBFGS::update_memory(
          colvec & axold,
          colvec & agold,
          colvec & axnew,
          colvec & agnew
          )
{
  // update the lbfgs memory
  // This updates s_, y_, rho_, and H0_, and k_
  int klocal = k_ % M_;
  ay_.row(klocal)=trans(agnew-agold);

  //double ys = vecdot(y_[klocal], s_[klocal]);
  double ys = dot(ay_.row(klocal),as_.row(klocal));
  if (ys == 0.) {
    // should print a warning here
    cout << "warning: resetting YS to 1.\n";
    ys = 1.;
  }
  arho_(klocal)=1./ys;

  double yy = dot(ay_.row(klocal),ay_.row(klocal));
  if (yy == 0.) {
    // should print a warning here
    cout << "warning: resetting YY to 1.\n";
    yy = 1.;
  }
  H0_ = ys / yy;
//  cout << "    setting H0 " << H0_ 
//    << " ys " << ys 
//    << " yy " << yy 
//    << " rho[i] " << rho_[klocal] 
//    << "\n";

  // increment k
  k_ += 1;
  
}

void LBFGS::compute_lbfgs_step()
{
  if (k_ == 0){ 
    double gnorm = vecnorm(ag_);
    if (gnorm > 1.) gnorm = 1. / gnorm;
	astep_ = - gnorm * H0_ * ag_;
    return;
  } 

  astep_ = ag_;

  int jmin = std::max(0, k_ - M_);
  int jmax = k_;
  int i;
  double beta;
  
  colvec alpha(M_);

  // loop backwards through the memory
  for(int j=jmax-1;j>=jmin;--j)
  {
	  i=j%M_;
	  alpha(i)=arho_(i)*dot(as_.row(i),astep_);
	  for(size_t j2 = 0; j2<astep_.size();j2++)
	  {
		  astep_(j2)-= alpha(i)*(ay_(i,j2));
	  }
  }

  // scale the step size by H0
  astep_=H0_*astep_;

  // loop forwards through the memory
  for(int j = jmin;j<jmax;j++)
  {
	  i=j%M_;
	  beta = arho_(i)*dot(ay_.row(i),astep_);
	  astep_=astep_+(alpha(i)-beta)*as_.row(i).t();
  }

  // invert the step to point downhill
  astep_=-astep_;
}

double LBFGS::backtracking_linesearch()
{
  double fnew;
  colvec xnew(ax_.size());
  colvec gnew(ax_.size());

  // if the step is pointing uphill, invert it
  if(dot(astep_,ag_)>0.)
  {
	  cout << "warning: step direction was uphill.  inverting\n"<<endl;
	  astep_=-astep_;
  }

  double factor = 1.;
  double stepsize = vecnorm(astep_);

  // make sure the step is no larger than maxstep_
  if (factor * stepsize > maxstep_){
    factor = maxstep_ / stepsize;
  }

  int nred;
  int nred_max = 10;

  for(nred =0;nred<nred_max;nred++)
  {
	  xnew=ax_+factor*astep_;
	  compute_func_gradient(xnew,fnew,gnew);
	  double df = fnew - f_;
	  if (df < max_f_rise_){
         break;
      } else {
         factor /= 10.;
         cout 
          << "function increased: " << df 
          << " reducing step size to " << factor * stepsize 
          << " H0 " << H0_ << "\n";
     }
  }

  if (nred >= nred_max){
    // possibly raise an error here
    cout << "warning: the line search backtracked too many times\n";
  }

  ax_ = xnew;
  ag_ = gnew;
  f_ = fnew;
  rms_ = vecnorm(gnew) / sqrt((double)gnew.size());
  return stepsize * factor;
}

bool LBFGS::stop_criterion_satisfied()
{
  return rms_ <= tol_;
}

void LBFGS::compute_func_gradient(colvec &x, double & func,
      colvec &gradient)
{
  nfev_ += 1;
  func = (*afunc_f_grad_)(x, gradient, x.size());
}

void LBFGS::set_H0(double H0)
{
  if (iter_number_ > 0){
    cout << "warning: setting H0 after the first iteration.\n";
  }
  H0_ = H0;
}
