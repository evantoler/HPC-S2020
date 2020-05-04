// Parallel sample sort
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <mpi.h>
#include <stdlib.h>
#include <algorithm>

int main( int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank, p;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  // Number of random numbers per processor (this should be increased
  // for actual tests or could be passed in through the command line
  int N;
  sscanf(argv[1], "%d", &N);

  int* vec = (int*)malloc(N*sizeof(int));
  // seed random number generator differently on every core
  srand((unsigned int) (rank + 393919));

  // fill vector with random integers
  for (int i = 0; i < N; ++i) {
    vec[i] = rand();
  }
  printf("rank: %d, first entry: %d\n", rank, vec[0]);

  // begin timing
  double time = -MPI_Wtime();

  // sort locally
  std::sort(vec, vec+N);

  // sample p-1 entries from vector as the local splitters, i.e.,
  // every N/P-th entry of the sorted vector
  int j, splitters[p-1], s = ceil(N/(double)p);
  for (j=0; j<p-1; j++) splitters[j] = vec[(j+1)*s-1];

  // every process communicates the selected entries to the root
  // process; use for instance an MPI_Gather
  int* gsplitters = (int*) malloc(p*(p-1) * sizeof(int)); // global splitters
  int root = 0;
  MPI_Gather(splitters, p-1, MPI_INT, gsplitters, p-1, MPI_INT, root, MPI_COMM_WORLD); // gather splitters at rank 0 (root)

  // root process does a sort and picks (p-1) splitters (from the
  // p(p-1) received elements)
  int bins[p-1];
  if (rank == root){
    std::sort(gsplitters, gsplitters+p*(p-1));
    for (j=0; j<p-1; j++) bins[j] = gsplitters[(j+1)*(p-1)];
  }

  // root process broadcasts splitters to all other processes
  MPI_Bcast(&bins, p-1, MPI_INT, root, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  // every process uses the obtained splitters to decide which
  // integers need to be sent to which other process (local bins).
  // Note that the vector is already locally sorted and so are the
  // splitters; therefore, we can use std::lower_bound function to
  // determine the bins efficiently.
  //
  // Hint: the MPI_Alltoallv exchange in the next step requires
  // send-counts and send-displacements to each process. Determining the
  // bins for an already sorted array just means to determine these
  // counts and displacements. For a splitter s[i], the corresponding
  // send-displacement for the message to process (i+1) is then given by,
  // sdispls[i+1] = std::lower_bound(vec, vec+N, s[i]) - vec;
  int scounts[p]; // count how many local integers need to be sent to each bin
  int sdispls[p]; // count the displacements

  sdispls[0] = 0;
  for (j=0; j<p-1; j++) sdispls[j+1] = std::lower_bound(vec, vec+N, bins[j]) - vec; // pointer arithmeric
  for (j=0; j<p-1; j++) scounts[j] = sdispls[j+1] - sdispls[j];
  scounts[p-1] = N - sdispls[p-1];

  // send and receive: first use an MPI_Alltoall to share with every
  // process how many integers it should expect, and then use
  // MPI_Alltoallv to exchange the data
  int rcounts[p], rdispls[p], N_new=0; // how many integers to receive from each process 0,...,p-1
  MPI_Alltoall(&scounts, 1, MPI_INT, &rcounts, 1, MPI_INT, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  rdispls[0] = 0;
  for (j=0; j<p-1; j++) rdispls[j+1] = rdispls[j] + rcounts[j]; // pointer arithmeric
  for (j=0; j<p; j++) N_new += rcounts[j]; // how many integers will this process receive?
  int* vec_new = (int*) malloc(N_new * sizeof(int)); // allocate memory for receiving
  MPI_Alltoallv(vec, scounts, sdispls, MPI_INT, vec_new, rcounts, rdispls, MPI_INT, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  // do a local sort of the received data
  std::sort(vec_new, vec_new+N_new);
  MPI_Barrier(MPI_COMM_WORLD);

  // end timing
  time += MPI_Wtime();
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank==root) printf("Time elapsed is %f seconds.\n", time);

  // every process writes its result to a file
  FILE* fd = NULL;
  char filename[256];
  snprintf(filename, 256, "sorted_N%d_rank%02d.txt", N, rank);
  fd = fopen(filename,"w+");

  if(NULL == fd) {
    printf("Error opening file \n");
    return 1;
  }

  for(j = 0; j < N_new; j++) fprintf(fd, "%d\n", vec_new[j]);

  fclose(fd);

  // tidy
  free(vec);
  free(vec_new);
  free(gsplitters);
  MPI_Finalize();
  return 0;
}
