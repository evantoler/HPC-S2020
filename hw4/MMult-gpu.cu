
#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "utils.h"

// number of threads per block.
// multiple of warp size, 32.
#define BLOCK_SIZE 1024

// Note: matrices are stored in column major order; i.e. the array elements in
// the (m x n) matrix C are stored in the sequence: {C_00, C_10, ..., C_m0,
// C_01, C_11, ..., C_m1, C_02, ..., C_0n, C_1n, ..., C_mn}
void MMult1(long m, long n, long k, double *a, double *b, double *c) {

  // optimal loop ordering for column-major matrices
  #pragma omp parallel for schedule(static)
  for (long j = 0; j < n; j++) {
    for (long p = 0; p < k; p++) {
      double B_pj = b[p+j*k];
      for (long i = 0; i < m; i++) {
        double A_ip = a[i+p*m];
        double C_ij = c[i+j*m];
        C_ij = C_ij + A_ip * B_pj;
        c[i+j*m] = C_ij;
      }
    }
  }

}

// compute the partial dot product of N-vectors x and y.
// sum stores terms of the dot product
__global__
void kernel_dot2(long N, double* dotprod, const double* x, const double* y){
  __shared__ double shared_xy[BLOCK_SIZE]; // element-wise product of x and y
  long idx = (blockIdx.x) * blockDim.x + threadIdx.x;

  // each thread reads data from global into shared memory
  if (idx < N) shared_xy[threadIdx.x] = x[idx] * y[idx];
  else shared_xy[threadIdx.x] = 0;
  __syncthreads();

  // x >>= 1 means "set x to itself shifted by one bit to the right", i.e., a divison by 2
  // write to memory with threadIdx rather than ``index''
  for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
  	if (threadIdx.x < s) {
  		shared_xy[threadIdx.x] += shared_xy[threadIdx.x + s];
  	}
  	__syncthreads();
  }

  // write to global memory
  if (threadIdx.x == 0) dotprod[blockIdx.x] = shared_xy[threadIdx.x];
}

// the summation kernel from our class example gpu16.cu
__global__ void reduction_kernel2(double* sum, const double* a, long N){
  __shared__ double smem[BLOCK_SIZE];
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

  // each thread reads data from global into shared memory
  if (idx < N) smem[threadIdx.x] = a[idx];
  else smem[threadIdx.x] = 0;
  __syncthreads();

  // x >>= 1 means "set x to itself shifted by one bit to the right", i.e., a divison by 2
  // write to memory with threadIdx rather than ``index''
  for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
	if (threadIdx.x < s) {
		smem[threadIdx.x] += smem[threadIdx.x + s];
	}
	__syncthreads();
   }

  // write to global memory
  if (threadIdx.x == 0) sum[blockIdx.x] = smem[threadIdx.x];
}

int main(int argc, char** argv) {

  long N = (1UL<<25); // 2^25

  double *x, *y, *device_dotprod;
  double dotprod, dotprod_ref;
  cudaMallocManaged(&x, N * sizeof(double));
  cudaMallocManaged(&y, N * sizeof(double));

  // initialize data and get reference solution
  dotprod_ref = 0;
  for(long i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    dotprod_ref += x[i] * y[i];
  }

  // make a buffer for efficient memory cudaSuccess
  long N_work = 1;
  for (long i = (N+BLOCK_SIZE-1)/(BLOCK_SIZE); i > 1; i = (i+BLOCK_SIZE-1)/(BLOCK_SIZE)) N_work += i;
  cudaMalloc(&device_dotprod, N_work*sizeof(double));

  // now check with GPU
  long N_block = (N+BLOCK_SIZE-1)/(BLOCK_SIZE);
  kernel_dot2<<<N_block,BLOCK_SIZE>>>(N, device_dotprod, x, y);
  while (N_block > 1) {
    long N_reduce = N_block; // number of dot product terms to add up
    N_block = (N_block+BLOCK_SIZE-1)/(BLOCK_SIZE); // number of blocks in this new, reduced vector
    reduction_kernel2<<<N_block,BLOCK_SIZE>>>(device_dotprod + N_reduce, device_dotprod, N_reduce); // reduce; store new terms shifted
    device_dotprod += N_reduce; // trace/copy the shift
  }
  cudaMemcpy(&dotprod, device_dotprod, sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  // compute Error
  double err = fabs(dotprod - dotprod_ref);
  printf("Dot product test:\n");
  printf("GPU Error       = %1.5e\n", err);

  cudaFree(x);
  cudaFree(y);
  cudaFree(device_dotprod);


  /*
  Matrix multiplication on GPU
  */

  // matrix A is MxN
  long M = (1UL<<10);
  N = (1UL<<10);

  // compute the product y = A*x
  double *A, *device_A, *device_x, *y_ref;
  cudaMalloc(&device_A, M*N * sizeof(double));
  cudaMalloc(&device_x,   N * sizeof(double));
  A = (double*) aligned_malloc(M*N * sizeof(double));
  x = (double*) aligned_malloc(  N * sizeof(double));
  y = (double*) aligned_malloc(M * sizeof(double));
  y_ref = (double*) aligned_malloc(M * sizeof(double));

  // initialize A and x
  for(long i=0; i<M*N; i++) A[i] = drand48();
  for(long i=0; i<N; i++)   x[i] = drand48();

  // compute the reference Matvec on the CPU
  Timer t;
  t.tic();
  for(long i=0; i<M; i++) y_ref[i] = 0.0;
  MMult1(M, 1, N, A, x, y_ref);
  double time_cpu = t.toc();

  /*
  compare with the GPU via M dot products
  */

  // copy constant data to device
  t.tic();
  cudaMemcpyAsync(device_A, A, M*N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(device_x, x,   N * sizeof(double), cudaMemcpyHostToDevice);

  // make a buffer for efficient memory cudaSuccess
  N_work = 1;
  for(long i = (N+BLOCK_SIZE-1)/(BLOCK_SIZE); i > 1; i = (i+BLOCK_SIZE-1)/(BLOCK_SIZE)) N_work += i;
  cudaMalloc(&device_dotprod, N_work*sizeof(double));

  double *Arow;
  cudaMallocManaged(&Arow, N * sizeof(double));

  for(long i=0; i<M; i++) {

    // construct Arow
    for(long j=0; j<N; j++) Arow[j] = A[i+j*N];

    // compute the inner product
    long N_block = (N+BLOCK_SIZE-1)/(BLOCK_SIZE);
    kernel_dot2<<<N_block,BLOCK_SIZE>>>(N, device_dotprod, Arow, device_x);
    while (N_block > 1) {
      long N_reduce = N_block; // number of dot product terms to add up
      N_block = (N_block+BLOCK_SIZE-1)/(BLOCK_SIZE); // number of blocks in this new, reduced vector
      reduction_kernel2<<<N_block,BLOCK_SIZE>>>(device_dotprod + N_reduce, device_dotprod, N_reduce); // reduce; store new terms shifted
      device_dotprod += N_reduce; // trace/copy the shift
    }
    cudaMemcpy(y+i, device_dotprod, sizeof(double), cudaMemcpyDeviceToHost); // the dot product is y[i]
    cudaDeviceSynchronize();

  }
  double time_gpu = t.toc();

  // compute Error
  err = 0;
  for(long i=0; i<M; i++) err += fabs(y[i] - y_ref[i]);
  printf("Matvec computation:\n");
  printf("GPU Error       = %1.5e\n", err);

  // compute Bandwidth
  double band = 4*M*N; // M inner products
  band *= sizeof(double) / 1e9;
  printf("------------------------------------------\n");
  printf("                     CPU        GPU\n");
  printf("Bandwidth (GB/s)     %1.3e  %1.3e\n", band/time_cpu, band/time_gpu);

  cudaFree(A);
  cudaFree(Arow);
  cudaFree(x);
  cudaFree(y);
  cudaFree(device_dotprod);

  return 0;
}
