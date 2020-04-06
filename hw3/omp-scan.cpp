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
  long j, k, start_idx, end_idx, start_idx_vec[p+1]; // indices to track

  // sections which have (length_1) elements
  for(j=0; j<p-remainder; j++){
    start_idx_vec[j] = j*length_1;
  }
  end_idx = (p-remainder)*length_1;

  // sections which have an extra array element because of division remainder
  for(j=p-remainder; j<p; j++){
    start_idx_vec[j] = end_idx;
    end_idx = start_idx_vec[j] + (length_1+1);
  }
  start_idx_vec[p] = end_idx;

  #pragma omp parallel
  {
    #pragma omp for schedule(static,1)
    for(long i=0; i<p; i++){

      prefix_sum[start_idx_vec[i]] = A[start_idx_vec[i]];
      for(long k=1; k<start_idx_vec[i+1]-start_idx_vec[i]; k++){
        prefix_sum[start_idx_vec[i]+k] = prefix_sum[start_idx_vec[i]+k-1] + A[start_idx_vec[i]+k];
      }

      partial_sums[i] = prefix_sum[start_idx_vec[i+1]-1];

    }
  }

  /*
  An alternative, task-based implementation.
  This ended up performing slowly.
  */

  // // sections which have (length_1) elements
  // for(j=0; j<p-remainder; j++){
  //   start_idx = j*length_1;
  //   end_idx = (j+1)*length_1;
  //   #pragma omp task shared(partial_sums, length_1, prefix_sum) firstprivate(j, start_idx, end_idx)
  //   {
  //     printf("Thread %d", omp_get_thread_num());
  //     prefix_sum[start_idx] = A[start_idx];
  //     for(long i=1; i<length_1; i++){
  //       prefix_sum[start_idx+i] = prefix_sum[start_idx+i-1] + A[start_idx+i];
  //     }
  //
  //     // store the partial sum of this section
  //     partial_sums[j] = prefix_sum[end_idx-1];
  //   }
  // }
  //
  // // sections which have an extra array element because of division remainder
  // for(j=p-remainder; j<p; j++){
  //   start_idx = end_idx;
  //   end_idx = start_idx + (length_1+1);
  //   #pragma omp task shared(partial_sums, length_1, prefix_sum) firstprivate(j, start_idx, end_idx)
  //   {
  //     printf("Thread %d", omp_get_thread_num());
  //     prefix_sum[start_idx] = A[start_idx];
  //     for(long i=1; i<length_1+1; i++){
  //       prefix_sum[start_idx+i] = prefix_sum[start_idx+i-1] + A[start_idx+i];
  //     }
  //
  //     // store the partial sum of this section
  //     partial_sums[j] = prefix_sum[end_idx-1];
  //   }
  // }
  //
  // // wait for all section scans to finish
  // #pragma omp taskwait

  /*
  perform corrections in serial
  */

  // initialize partial sum accumulator
  long s;
  s = partial_sums[0];

  // adjust via partial sums from each section.
  // the first section (section 0) does not need adjustment
  for(j=1; j<p; j++){

    for(k=0; k<start_idx_vec[j+1]-start_idx_vec[j]; k++)
      prefix_sum[start_idx_vec[j]+k] += s;

    // update partial sum accumulation
    s += partial_sums[j];

  }

}

int main() {
  long N = 1000000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();

  // set the number of parallel threads
  printf("Max threads = %d\n", omp_get_max_threads());
  int p = 64;
  omp_set_num_threads(p);
  printf("Using %d theads/sections.\n", p);

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
