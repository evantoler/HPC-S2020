
#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "utils.h"

// number of threads per block.
// multiple of warp size, 32.
// blocks are size BLOCK_DIM_X x BLOCK_DIM_Y
#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16
#define BLOCK_SIZE (BLOCK_DIM_X * BLOCK_DIM_Y)

// Jacobi method
// N = number of internal (equispaced) grid points in one dimension.
// f = N^2-vector; the right-hand side forcing term
// u = N^2-vector; pointer to memory to store the output
// maxiter = maximum number of Gauss-Seidel iterations
// output: the solution u at N^2 internal points of (0,1)^2
double* Jacobi_cpu(long N, double* f, double* u, long maxiter) {

  double h = 1 / ((double) N+1); // mesh spacing in one dimension
  double hsq = h*h;

  int i,j; // loop variables
  double* u_old = (double*) aligned_malloc(N*N*sizeof(double));

  // initial guess u0 = 0
  #pragma omp parallel for schedule(static)
  for (j=0; j<N*N; j++)
    u[j] = 0;

  for (long iter = 0; iter < maxiter; iter++) { // main loop; cannot parallelize

    // Update the solution via Jacobi;
    // specialized for the finite difference matrix

    /*
    begin update
    */

    // make a copy of the previous iterate
    #pragma omp parallel for schedule(static)
    for (j=0; j<N*N; j++)
      u_old[j] = u[j];

    // the corner points
    u[0]      = (hsq * f[0]     + u_old[1]       + u_old[N]) / 4.0;
    u[N-1]    = (hsq * f[N-1]   + u_old[N-2]     + u_old[2*N-1]) / 4.0;
    u[N*N-N]  = (hsq * f[N*N-N] + u_old[N*N-2*N] + u_old[N*N-N+1]) / 4.0;
    u[N*N-1]  = (hsq * f[N*N-1] + u_old[N*N-N-1] + u_old[N*N-2]) / 4.0;

    #pragma omp parallel
    {

      // the edge points
      #pragma omp for schedule(static) nowait
      for (j=1; j<N-1; j++)          // left edge
        u[j] = (hsq * f[j] + u_old[j-1] + u_old[j+1] + u_old[j+N]) / 4.0;
      #pragma omp for schedule(static) nowait
      for (j=N*N-N+1; j<N*N-1; j++)  // right edge
        u[j] = (hsq * f[j] + u_old[j-1] + u_old[j+1] + u_old[j-N]) / 4.0;
      #pragma omp for schedule(static) nowait
      for (j=N; j<N*N-N; j+=N)       // bottom edge
        u[j] = (hsq * f[j] + u_old[j+1] + u_old[j-N] + u_old[j+N]) / 4.0;
      #pragma omp for schedule(static) nowait
      for (j=2*N-1; j<N*N-1; j+=N)   // top edge
        u[j] = (hsq * f[j] + u_old[j-1] + u_old[j-N] + u_old[j+N]) / 4.0;



      // the interior points
      #pragma omp for schedule(static)
      for(i = 1; i<N-1; i++) {

        for (j = 1; j<N-1; j++)
          u[j+i*N] = (hsq * f[j+i*N] + u_old[j+i*N-1] + u_old[j+i*N+1] + u_old[j+i*N-N] + u_old[j+i*N+N]) / 4.0;

      } // for i

    } // end of parallel region

    /*
    end update
    */

  }

  aligned_free(u_old);

  return u;
}

// Jacobi method update step.
// Use data in RHS f and previous iterate u_old to store new iterate in u.
// spatial NxN grid with *squared* step size hsq=(1/(N+1))^2 in each direction
// Copies updated iterate into u_old for the next iteration.
// Compoute the squared residual res=||Au - f||^2
__global__
void kernel_Jacobi_update(long N, double hsq, const double* f, double* u_old, double* u) {

  // compute the master indices for the solution u
  long row = threadIdx.x + blockIdx.x * blockDim.x;
  long col = threadIdx.y + blockIdx.y * blockDim.y;
  long idx = row + N*col;

  // update the solution iterate
  u[idx] = hsq * f[idx];
  if(row-1 >= 0 && idx<N*N) u[idx] += u_old[idx-1]; // look up
  if(row+1 <  N && idx<N*N) u[idx] += u_old[idx+1]; // look down
  if(col-1 >= 0 && idx<N*N) u[idx] += u_old[idx-N]; // look left
  if(col+1 <  N && idx<N*N) u[idx] += u_old[idx+N]; // look right
  u[idx] = u[idx] / 4.0;
}

// copy the previous solution after updating
__global__
void kernel_update_u_old(long N, double* u_old, double* u) {
  long row = threadIdx.x + blockIdx.x * blockDim.x;
  long col = threadIdx.y + blockIdx.y * blockDim.y;
  long idx = row + N*col;
  u_old[idx] = u[idx];
}

// Solve the 2-D NxN Laplace problem with Jacobi.
// right-hand side f
// store solution into u
// terminate after maxiter iterations
double* Jacobi_gpu(long N, double* f, double* u, long maxiter) {

  double h = 1 / ((double) N+1); // mesh spacing in one dimension
  double hsq = h*h;

  // initial guess u0 = 0
  double *u_old;
  u_old = (double*) aligned_malloc(N*N*sizeof(double));

  #pragma omp parallel for schedule(static)
  for(long j=0; j<N*N; j++)
    u_old[j] = 0;
  // printf("Made u_old\n");

  // prepare the necessary data for the GPU kernel
  double *device_u, *device_f, *device_u_old;
  cudaMalloc(&device_u_old, N*N * sizeof(double));
  cudaMalloc(&device_u, N*N * sizeof(double));
  cudaMalloc(&device_f, N*N * sizeof(double));
  cudaMemcpy(device_u_old, u_old, N*N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(device_u, u, N*N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(device_f, f, N*N * sizeof(double), cudaMemcpyHostToDevice);

  // structure the block and grid structure to be the same size as the
  // solution u in matrix form
  dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y), gridDim(N/BLOCK_DIM_X, N/BLOCK_DIM_Y);

  for (long iter = 0; iter < maxiter; iter++) { // main loop; cannot parallelize

    // Update the solution via Jacobi;
    // specialized for the finite difference matrix
    cudaDeviceSynchronize();
    kernel_Jacobi_update<<<gridDim, blockDim>>>(N, hsq, device_f, device_u_old, device_u);
    cudaDeviceSynchronize();
    kernel_update_u_old<<<gridDim, blockDim>>>(N, device_u_old, device_u);

  }//iteration loop

  // get solution from device
  cudaDeviceSynchronize();
  cudaMemcpy(u, device_u, N*N * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(device_u_old);
  cudaFree(device_u);
  cudaFree(device_f);
  // printf("did memory free OK\n");

  return u;

}//function

int main(int argc, char** argv) {

  const long N = (1UL<<10);
  const long maxiter = 100;
  const long NREPEATS = 30;

  double* u_cpu = (double*) aligned_malloc(N*N*sizeof(double));
  double* u_gpu = (double*) aligned_malloc(N*N*sizeof(double));
  double* f = (double*) aligned_malloc(N*N*sizeof(double));

  // forcing is f=1
  for (int i=0; i<N*N; i++)
    f[i] = 1;

  printf("--------------------------\n");
  printf("Parameters:\n");
  printf("N        = %d\n", N);
  printf("maxiter  = %d\n", maxiter);
  printf("NREPEATS = %d\n", NREPEATS);
  printf("--------------------------\n");

  Timer t;

  printf("--------------------------\n");
  printf("CPU method.\n");
  t.tic();
  for (long rep = 0; rep < NREPEATS; rep++) {
    u_cpu = Jacobi_cpu(N, f, u_cpu, maxiter);
    printf("Instance %d/%d done\n", rep, NREPEATS-1);
  }
  double time_cpu = t.toc();

  printf("--------------------------\n");
  printf("GPU method.\n");
  t.tic();
  for (long rep = 0; rep < NREPEATS; rep++) {
    u_gpu = Jacobi_gpu(N, f, u_gpu, maxiter);
    printf("Instance %d/%d done\n", rep, NREPEATS-1);
  }
  double time_gpu = t.toc();

  // compute bandwidths: per iteration of Jacobi
  double band = 4.0*4 + 4.0*(N-2)*5 + 6.0*(N-2)*(N-2); // update step ...
  band += 2.0*N*N; // copy old solution
  band *= maxiter * NREPEATS * sizeof(double) / 1e9; //number of iterations

  printf("--------------------------\n");
  printf("      Average time (s)      Average Bandwidth (GB/s)\n");
  printf("CPU   %1.5e       %1.5e\n", time_cpu/NREPEATS, band/time_cpu);
  printf("GPU   %1.5e       %1.5e\n", time_gpu/NREPEATS, band/time_gpu);

  // compute error as the difference between the methods
  // should be 0
  double err = 0;
  for(long j = 0; j<N*N; j++) err += fabs(u_cpu[j]-u_gpu[j]);
  printf("Error = %1.5e\n", err);


  aligned_free(u_cpu);
  aligned_free(u_gpu);
  aligned_free(f);

  return 0;
}
