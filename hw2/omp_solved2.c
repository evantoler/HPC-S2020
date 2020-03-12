/******************************************************************************
* FILE: omp_bug2.c
* DESCRIPTION:
*   Another OpenMP program with a bug.
* AUTHOR: Blaise Barney
* LAST REVISED: 04/06/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[])
{
int nthreads, i, tid;
double total = 0.0; // Evan: changed float to double for more precision and
                    // initialized the total outside the parallel loop for efficiency.

// Evan: You should not write into i and tid since they are defined
// before the parallel construct. This creates a race condition.
// nthreads is OK since only the master thread writes into it.
// I added private variables to the parallel construct to accurately keep track
// of the thread IDs
/*** Spawn parallel region ***/
#pragma omp parallel private(i, tid)
  {
  /* Obtain thread number */
  tid = omp_get_thread_num();

  /* Only master thread does this */
  if (tid == 0) {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  printf("Thread %d is starting...\n",tid);

  #pragma omp barrier

  /* do some work */
  double my_total = 0.0; // added a thread-private variable to prevent
                         // a race condition
  #pragma omp for schedule(dynamic,10)
  for (i=0; i<1000000; i++)
    my_total = my_total + (double) i*1.0;

   // added this critical section -- revise
   #pragma omp critical
    total = total + my_total;

    #pragma omp barrier
      printf ("Thread %d is done! Total= %e\n",tid,total);

  } /*** End of parallel region ***/
}
