/*
 * ----------------------------------------------------------------------------
 * "THE BEER-WARE LICENSE" (Revision 42):
 * <js850@cam.ac.uk> wrote this file. As long as you retain this notice you
 * can do whatever you want with this stuff. If we meet some day, and you think
 * this stuff is worth it, you can buy me a beer in return Jacob Stevenson
 * ----------------------------------------------------------------------------
 */

/**
 * an implementation of the LBFGS optimization algorithm in c++.  This
 * implemenation uses a backtracking linesearch.
 */
#include <vector>
#include <armadillo>

using namespace std;
using namespace arma;

  class LBFGS{
    private : 
      // input parameters
      int M_; /**< The lenth of the LBFGS memory */
      double tol_; /**< The tolerance for the rms gradient */
      double maxstep_; /**< The maximum step size */
      double max_f_rise_; /**< The maximum the function is allowed to rise in a
                           * given step.  This is the criterion for the
                           * backtracking line search.
                           */
      int maxiter_; /**< The maximum number of iterations */
      int iprint_;

      int iter_number_; /**< The current iteration number */
      int nfev_; /**< The number of function evaluations */

	  /**
       * A pointer to the function that computes the function and gradient
       */
	  double (* afunc_f_grad_)(colvec &,colvec &,size_t);

      // variables representing the state of the system
	  colvec ax_;
      double f_;
	  colvec ag_;
      double rms_;

      // places to store the lbfgs memory
	  mat as_;
	  mat ay_;
	  colvec arho_;
      double H0_;
      int k_; /**< Counter for how many times the memory has been updated */

      // 
	  colvec astep_;

    public :
      /**
       * Constructor
       */

	  LBFGS(
          double (*func)(colvec &, colvec &, size_t), 
          colvec & x0, 
          size_t N, 
          int M
          );

      /**
       * Destructor
       */
      ~LBFGS() {}

      /**
       * Do one iteration iteration of the optimization algorithm
       */
      void one_iteration();

      /**
       * Run the optimzation algorithm until the tolerance is satisfied or
       * until the maximum number of iterations is reached
       */
      void run();

      // functions for setting the parameters
      void set_H0(double);
      void set_tol(double tol) { tol_ = tol; }
      void set_maxstep(double maxstep) { maxstep_ = maxstep; }
      void set_max_f_rise(double max_f_rise) { max_f_rise_ = max_f_rise; }
      void set_max_iter(int max_iter) { maxiter_ = max_iter; }
      void set_iprint(int iprint) { iprint_ = iprint; }

      // functions for accessing the results
	  colvec get_ax() { return ax_;}
	  colvec get_ag() { return ag_;}
      double get_f() { return f_; }
      double get_rms() { return rms_; }
      double get_H0() { return H0_; }
      int get_nfev() { return nfev_; }
      int get_niter() { return iter_number_; }
      bool success() { return stop_criterion_satisfied(); }

    private :

      /**
       * Add a step to the LBFGS Memory
       * This updates s_, y_, rho_, H0_, and k_
       */
	  void update_memory(
		  colvec & xold,
		  colvec & gold,
		  colvec & xnew,
		  colvec & gnew
		  );

      /**
       * Compute the LBFGS step from the memory
       */
      void compute_lbfgs_step();

      /**
       * Take the step and do a backtracking linesearch if necessary.
       * Apply the maximum step size constraint and ensure that the function
       * does not rise more than the allowed amount.
       */
      double backtracking_linesearch();

      /**
       * Return true if the termination condition is satisfied, false otherwise
       */
      bool stop_criterion_satisfied();

      /**
       * Compute the func and gradient of the objective function
       */
	  void compute_func_gradient(colvec &x, double &func, colvec &gradient);

  };