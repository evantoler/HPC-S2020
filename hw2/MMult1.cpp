// g++ -std=c++11 -O3 -fopenmp -march=native MMult1.cpp -o MMult1 && ./MMult1
// g++ -std=c++11 -O3 -march=native MMult1.cpp -o MMult1 && ./MMult1
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "utils.h"

// A block size of 120 is optimal.
#define BLOCK_SIZE 120

// Note: matrices are stored in column major order; i.e. the array elements in
// the (m x n) matrix C are stored in the sequence: {C_00, C_10, ..., C_m0,
// C_01, C_11, ..., C_m1, C_02, ..., C_0n, C_1n, ..., C_mn}
void MMult0(long m, long n, long k, double *a, double *b, double *c) {
  for (long j = 0; j < n; j++) {
    for (long p = 0; p < k; p++) {
      for (long i = 0; i < m; i++) {
        double A_ip = a[i+p*m];
        double B_pj = b[p+j*k];
        double C_ij = c[i+j*m];
        C_ij = C_ij + A_ip * B_pj;
        c[i+j*m] = C_ij;
      }
    }
  }
}

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

  // // Strictly minimizing memory accesses
  // for (long j = 0; j < n; j++) {
  //   for (long i = 0; i < m; i++) {
  //     double C_ij = c[i+j*m];
  //     for (long p = 0; p < k; p++) {
  //       double A_ip = a[i+p*m];
  //       double B_pj = b[p+j*k];
  //       C_ij = C_ij + A_ip * B_pj;
  //     }
  //     c[i+j*m] = C_ij;
  //   }
  // }
}

void MMult1block(long m, long n, long k, double *a, double *b, double *c) {

  // Use blocking BLOCK_SIZE to speed up matrix multiplication.
  // We assume the following are integers
  long Bm = m / BLOCK_SIZE;
  long Bn = n / BLOCK_SIZE;
  long Bk = k / BLOCK_SIZE;

  // Load blocks *one at a time* and send to MMult1 for each blocks.
  // Blocks are stored on the stack.
  double A_ip[BLOCK_SIZE * BLOCK_SIZE]; // one block of A
  double B_pj[BLOCK_SIZE * BLOCK_SIZE]; // one block of B
  double C_ij[BLOCK_SIZE * BLOCK_SIZE]; // one block of C

  // variables "local" to each block
  int row, col;

  // minimizing memory accesses
  for (long jblock = 0; jblock < Bn; jblock++) {
    for (long iblock = 0; iblock < Bm; iblock++) {

      // build a block of C
      for (col = 0; col < BLOCK_SIZE; col++) {
        for (row = 0; row < BLOCK_SIZE; row++) {
          C_ij[row + col*BLOCK_SIZE] = c[(row + iblock*BLOCK_SIZE) + (col + jblock*BLOCK_SIZE)*m];
        }
      }

      for (long pblock = 0; pblock < Bk; pblock++) {

        // build a block of A
        for (col = 0; col < BLOCK_SIZE; col++) {
          for (row = 0; row < BLOCK_SIZE; row++) {
            A_ip[row + col*BLOCK_SIZE] = a[(row + iblock*BLOCK_SIZE) + (col + pblock*BLOCK_SIZE)*m];
          }
        }

        // build a block of B
        for (col = 0; col < BLOCK_SIZE; col++) {
          for (row = 0; row < BLOCK_SIZE; row++) {
            B_pj[row + col*BLOCK_SIZE] = b[(row + pblock*BLOCK_SIZE) + (col + jblock*BLOCK_SIZE)*k];
          }
        }

        // multiply the blocks
        MMult1(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, (double*) &A_ip, (double*) &B_pj, (double*) &C_ij);
      }

      // put the result in the original pointer c
      for (col = 0; col < BLOCK_SIZE; col++) {
        for (row = 0; row < BLOCK_SIZE; row++) {
          c[(row + iblock*BLOCK_SIZE) + (col + jblock*BLOCK_SIZE)*m] = C_ij[row + col*BLOCK_SIZE];
        }
      }

    }
  }

}

int main(int argc, char** argv) {
  const long PFIRST = BLOCK_SIZE;
  const long PLAST = 2000;
  const long PINC = std::max(50/BLOCK_SIZE,1) * BLOCK_SIZE; // multiple of BLOCK_SIZE

  #ifdef _OPENMP
    omp_set_num_threads(6);
  #endif

  printf(" Dimension       Time    Gflop/s       GB/s        Error\n");
  for (long p = PFIRST; p < PLAST; p += PINC) {
    long m = p, n = p, k = p;
    long NREPEATS = 1e9/(m*n*k)+1;
    double* a = (double*) aligned_malloc(m * k * sizeof(double)); // m x k
    double* b = (double*) aligned_malloc(k * n * sizeof(double)); // k x n
    double* c = (double*) aligned_malloc(m * n * sizeof(double)); // m x n
    double* c_ref = (double*) aligned_malloc(m * n * sizeof(double)); // m x n

    // Initialize matrices
    for (long i = 0; i < m*k; i++) a[i] = drand48();
    for (long i = 0; i < k*n; i++) b[i] = drand48();
    for (long i = 0; i < m*n; i++) c_ref[i] = 0;
    for (long i = 0; i < m*n; i++) c[i] = 0;

    for (long rep = 0; rep < NREPEATS; rep++) { // Compute reference solution
      MMult0(m, n, k, a, b, c_ref);
    }

    Timer t;
    t.tic();
    for (long rep = 0; rep < NREPEATS; rep++) {
      // MMult1(m, n, k, a, b, c);
      MMult1block(m, n, k, a, b, c);
    }
    double time = t.toc();
    double flops = NREPEATS * (2*m*n*k) / 1e9 / time;
    // double bandwidth = NREPEATS * (2*m*n + 2*m*n*k) * sizeof(double) / 1e9 / time; // suboptimal--"bad" MMult1
    // double bandwidth = NREPEATS * (n*k + 3*m*n*k) * sizeof(double) / 1e9 / time; // "good" MMult1
    double bandwidth = NREPEATS * (4*m*n + 5*m*n*k/BLOCK_SIZE + 3*m*n*k) * sizeof(double) / 1e9 / time; // MMult1block
    printf("%10d %10f %10f %10f", p, time/NREPEATS, flops, bandwidth);

    double max_err = 0;
    for (long i = 0; i < m*n; i++) max_err = std::max(max_err, fabs(c[i] - c_ref[i]));
    printf(" %10e\n", max_err);

    aligned_free(a);
    aligned_free(b);
    aligned_free(c);
    aligned_free(c_ref);
  }

  return 0;
}

// * Using MMult0 as a reference, implement MMult1 and try to rearrange loops to
// maximize performance. Measure performance for different loop arrangements and
// try to reason why you get the best performance for a particular order?
//
//
// * You will notice that the performance degrades for larger matrix sizes that
// do not fit in the cache. To improve the performance for larger matrices,
// implement a one level blocking scheme by using BLOCK_SIZE macro as the block
// size. By partitioning big matrices into smaller blocks that fit in the cache
// and multiplying these blocks together at a time, we can reduce the number of
// accesses to main memory. This resolves the main memory bandwidth bottleneck
// for large matrices and improves performance.
//
// NOTE: You can assume that the matrix dimensions are multiples of BLOCK_SIZE.
//
//
// * Experiment with different values for BLOCK_SIZE (use multiples of 4) and
// measure performance.  What is the optimal value for BLOCK_SIZE?
//
//
// * Now parallelize your matrix-matrix multiplication code using OpenMP.
//
//
// * What percentage of the peak FLOP-rate do you achieve with your code?
//
//
// NOTE: Compile your code using the flag -march=native. This tells the compiler
// to generate the best output using the instruction set supported by your CPU
// architecture. Also, try using either of -O2 or -O3 optimization level flags.
// Be aware that -O2 can sometimes generate better output than using -O3 for
// programmer optimized code.
