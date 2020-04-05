#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = A[0];
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i];
  }
}

// Scan A array and write result into prefix_sum array;
// parallel with p sections -- one for each thread
// use long data type to avoid overflow
void scan_omp(long* prefix_sum, const long* A, long n, long p) {

  // initialization
  if (n == 0) return;

  // split the array into (almost) equal sections: one for each thread
  long length_1 = n / p;      // length of sections without extra elements
  long remainder = n % p;     // number of sections that need an extra element

  long partial_sums[p];       // partial sums from scanning each section
  long j, start_idx, end_idx; // indices to track

  // sections which have (length_1) elements
  for(j=0; j<p-remainder; j++){
    start_idx = j*length_1;
    end_idx = (j+1)*length_1;
    #pragma omp task shared(partial_sums, length_1, prefix_sum) firstprivate(j, start_idx, end_idx)
    {
      prefix_sum[start_idx] = A[start_idx];
      for(long i=1; i<length_1; i++){
        prefix_sum[start_idx+i] = prefix_sum[start_idx+i-1] + A[start_idx+i];
      }

      // store the partial sum of this section
      partial_sums[j] = prefix_sum[end_idx-1];
    }
  }

  // sections which have an extra array element because of division remainder
  for(j=p-remainder; j<p; j++){
    start_idx = end_idx;
    end_idx = start_idx + (length_1+1);
    #pragma omp task shared(partial_sums, length_1, prefix_sum) firstprivate(j, start_idx, end_idx)
    {
      prefix_sum[start_idx] = A[start_idx];
      for(long i=1; i<length_1+1; i++){
        prefix_sum[start_idx+i] = prefix_sum[start_idx+i-1] + A[start_idx+i];
      }

      // store the partial sum of this section
      partial_sums[j] = prefix_sum[end_idx-1];
    }
  }

  // wait for all section scans to finish
  #pragma omp taskwait

  /*
  perform corrections in serial
  */

  // initialize counter and partial sum accumulator
  long i, s;
  s = partial_sums[0];

  // sections with (length_1) elements;
  // the first section (section 0) does not need adjustment
  for(j=1; j<p-remainder; j++){

    start_idx = j*length_1;
    end_idx = (j+1)*length_1;
    for(i=0; i<length_1+1; i++)
      prefix_sum[start_idx+i] += s;

    // update partial sum accumulation
    s += partial_sums[j];

  }

  // sections with an extra element
  for(j=p-remainder; j<p; j++){
    start_idx = end_idx;
    end_idx = start_idx + (length_1+1);
    for(i=0; i<length_1+1; i++)
      prefix_sum[start_idx+i] += s;

    // update partial sum accumulation
    s += partial_sums[j];

  }

}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();

  // set the number of parallel threads
  int p = 2;
  omp_set_num_threads(p);

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N, p);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
