// g++ -std=c++11 -O3 -fopenmp gs2D-omp.cpp -o gs2D-omp && ./gs2D-omp
// g++ -std=c++11 -O3 gs2D-omp.cpp -o gs2D-omp && ./gs2D-omp

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "utils.h"

// Gauss Seidel method
// using red-black ordering of unknowns
// N = number of internal (equispaced) grid points in one dimension. Assumed odd.
// f = N^2-vector; the right-hand side forcing term
// u = N^2-vector; pointer to memory to store the output
// maxiter = maximum number of Gauss-Seidel iterations
// output: the solution u at N^2 internal points of (0,1)^2
double* Gauss_Seidel(long N, double* f, double* u, long maxiter) {

  double h = 1 / ((double) N+1); // mesh spacing in one dimension
  double hsq = h*h;

  int i,j; // loop variables

  // initial guess u0 = 0
  #pragma omp parallel for schedule(static)
  for (j=0; j<N*N; j++)
    u[j] = 0;

  double tol = 1e-6;                // option: vary the relative tolerance
  double res0 = 0.0;                // initial (function) residual is norm(A*u0 - f) = norm(f)
  #pragma omp parallel for schedule(static) reduction(+: res0)
  for(j=0; j<N*N; j++)
    res0 = res0 + pow(f[j], 2.0);
  res0 = h*sqrt(res0);
  double res = res0;                // initialize residual

  // let's skip the residual printout
  // printf(" Iteration        Residual\n");

  for (int iter = 0; iter < maxiter; iter++) { // main loop; cannot parallelize

    // stopping condition
    if(res < tol*res0){
      break;
    }

    // Update the solution via Gauss Seidel;
    // specialized for the finite difference matrix

    /*
    begin update
    */

    /*
    The red points
    */

    // the corner points
    u[0]      = (hsq * f[0]     + u[1]       + u[N]) / 4.0;
    u[N-1]    = (hsq * f[N-1]   + u[N-2]     + u[2*N-1]) / 4.0;
    u[N*N-N]  = (hsq * f[N*N-N] + u[N*N-2*N] + u[N*N-N+1]) / 4.0;
    u[N*N-1]  = (hsq * f[N*N-1] + u[N*N-N-1] + u[N*N-2]) / 4.0;

    #pragma omp parallel
    {

      // the edge points
      #pragma omp for schedule(static) nowait
      for (j=2; j<N-1; j+=2)          // left edge
        u[j] = (hsq * f[j] + u[j-1] + u[j+1] + u[j+N]) / 4.0;
      #pragma omp for schedule(static) nowait
      for (j=N*N-N+2; j<N*N-1; j+=2)  // right edge
        u[j] = (hsq * f[j] + u[j-1] + u[j+1] + u[j-N]) / 4.0;
      #pragma omp for schedule(static) nowait
      for (j=2*N; j<N*N-N; j+=2*N)    // bottom edge
        u[j] = (hsq * f[j] + u[j+1] + u[j-N] + u[j+N]) / 4.0;
      #pragma omp for schedule(static) nowait
      for (j=3*N-1; j<N*N-1; j+=2*N)  // top edge
        u[j] = (hsq * f[j] + u[j-1] + u[j-N] + u[j+N]) / 4.0;

      // the interior points
      #pragma omp for schedule(static) nowait
      for(i = 1; i<N-2; i+=2) {

        for (j = 1; j<N-1; j+=2) // the "red long" columns
          u[j+i*N] = (hsq * f[j+i*N] + u[j+i*N-1] + u[j+i*N+1] + u[j+i*N-N] + u[j+i*N+N]) / 4.0;

        for (j = 2; j<N-1; j+=2) // the "red short" columns
          u[j+(i+1)*N] = (hsq * f[j+(i+1)*N] + u[j+(i+1)*N-1] + u[j+(i+1)*N+1] + u[j+(i+1)*N-N] + u[j+(i+1)*N+N]) / 4.0;

      } // for i

      i = N-2;
      #pragma omp for schedule(static) nowait
      for (j = 1; j<N-1; j+=2) { // the last, "mismatched" column ("red long")
        u[j+i*N] = (hsq * f[j+i*N] + u[j+i*N-1] + u[j+i*N+1] + u[j+i*N-N] + u[j+i*N+N]) / 4.0;
      }

      // make sure the red points update before the black points
      #pragma omp barrier

      /*
      The black points
      */

      // the edge points
      #pragma omp for schedule(static) nowait
      for (j=1; j<N-1; j+=2)          // left edge
        u[j] = (hsq * f[j] + u[j-1] + u[j+1] + u[j+N]) / 4.0;
      #pragma omp for schedule(static) nowait
      for (j=N*N-N+1; j<N*N-1; j+=2)  // right edge
        u[j] = (hsq * f[j] + u[j-1] + u[j+1] + u[j-N]) / 4.0;
      #pragma omp for schedule(static) nowait
      for (j=N; j<N*N-N; j+=2*N)      // bottom edge
        u[j] = (hsq * f[j] + u[j+1] + u[j-N] + u[j+N]) / 4.0;
      #pragma omp for schedule(static) nowait
      for (j=2*N-1; j<N*N-1; j+=2*N)  // top edge
        u[j] = (hsq * f[j] + u[j-1] + u[j-N] + u[j+N]) / 4.0;

      // the interior points
      #pragma omp for schedule(static) nowait
      for(i = 1; i<N-2; i+=2) {

        for (j = 2; j<N-1; j+=2) // the "black short" columns
          u[j+i*N] = (hsq * f[j+i*N] + u[j+i*N-1] + u[j+i*N+1] + u[j+i*N-N] + u[j+i*N+N]) / 4.0;

        for (j = 1; j<N-1; j+=2) // the "black long" columns
          u[j+(i+1)*N] = (hsq * f[j+(i+1)*N] + u[j+(i+1)*N-1] + u[j+(i+1)*N+1] + u[j+(i+1)*N-N] + u[j+(i+1)*N+N]) / 4.0;

      } // for i

      i = N-2;
      #pragma omp for schedule(static)
      for (j = 2; j<N-1; j+=2) { // the last, "mismatched" column ("black short")
        u[j+i*N] = (hsq * f[j+i*N] + u[j+i*N-1] + u[j+i*N+1] + u[j+i*N-N] + u[j+i*N+N]) / 4.0;
      }

    } // end parallel region

    /*
    end update
    */

    // Compute the (function) residual, norm(A*uk - f)

    res = 0.0;

    // corner points
    res = res + pow( (4*u[0]     - u[1]       - u[N])/hsq       - f[0] ,     2.0);
    res = res + pow( (4*u[N-1]   - u[N-2]     - u[2*N-1])/hsq   - f[N-1] ,   2.0);
    res = res + pow( (4*u[N*N-N] - u[N*N-2*N] - u[N*N-N+1])/hsq - f[N*N-N] , 2.0);
    res = res + pow( (4*u[N*N-1] - u[N*N-N-1] - u[N*N-2])/hsq   - f[N*N-1] , 2.0);

    #pragma omp parallel
    {

      // Be careful about race conditions for updating the residual res.
      // Parallel loops should keep the implicit barrier at the end

      // edge points
      #pragma omp for schedule(static) reduction(+ : res)
      for (j=1; j<N-1; j++)          // left edge
        res = res + pow( (4*u[j] - u[j-1] - u[j+1] - u[j+N])/hsq - f[j], 2.0);
      #pragma omp for schedule(static) reduction(+ : res)
      for (j=N*N-N+1; j<N*N-1; j++)  // right edge
        res = res + pow( (4*u[j] - u[j-1] - u[j+1] - u[j-N])/hsq - f[j], 2.0);
      #pragma omp for schedule(static) reduction(+ : res)
      for (j=N; j<N*N-N; j+=N)       // bottom edge
        res = res + pow( (4*u[j] - u[j+1] - u[j-N] - u[j+N])/hsq - f[j], 2.0);
      #pragma omp for schedule(static) reduction(+ : res)
      for (j=2*N-1; j<N*N-1; j+=N)   // top edge
        res = res + pow( (4*u[j] - u[j-1] - u[j-N] - u[j+N])/hsq - f[j], 2.0);

      // interior points
      #pragma omp for schedule(static) reduction(+ : res)
      for(i = 1; i<N-1; i++) {

        for (j = 1; j<N-1; j++)
          res = res + pow( (4*u[j+i*N] - u[j+i*N-1] - u[j+i*N+1] - u[j+i*N-N] - u[j+i*N+N])/hsq - f[j+i*N] , 2.0);

      } // for i

    } // end parallel region

    res = h*sqrt(res);

    // let's skip the residual printout
    // print the residual history
    // printf("%10d    %10e\n", iter, res);

  }

  return u;
}



int main(int argc, char** argv) {
  const long N = 101;
  const long maxiter = 10000;
  const long NREPEATS = 100;
  #ifdef _OPENMP
    omp_set_num_threads(6); // option: vary the number of threads
  #endif

  double* u = (double*) aligned_malloc(N*N*sizeof(double));
  double* f = (double*) aligned_malloc(N*N*sizeof(double));

  // forcing is f=1
  for (int i=0; i<N*N; i++)
    f[i] = 1;

  // printf(" Dimension       Time    Gflop/s       GB/s        Error\n");
  printf("--------------------------\n");
  printf("Parameters:\n");
  printf("N        = %d\n", N);
  printf("maxiter  = %d\n", maxiter);
  printf("NREPEATS = %d\n", NREPEATS);
  printf("--------------------------\n");

  Timer t;
  t.tic();
  for (long rep = 0; rep < NREPEATS; rep++) {
    u = Gauss_Seidel(N, f, u, maxiter);
    printf("Instance %d/%d done\n", rep, NREPEATS-1);
  }
  double time = t.toc();

  printf("--------------------------\n");
  printf("Average time over %d trials: %10f seconds\n", NREPEATS, time/NREPEATS);

  // debugging section
  // for (int i=0; i<N*N; i++)
  //   printf("u[%d] = %10f\n", i, u[i]);

  aligned_free(u);
  aligned_free(f);

  return 0;
}
