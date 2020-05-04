/* MPI-parallel Jacobi smoothing to solve -(u_xx + u_yy)=f
 * Global vector has N^2 unknowns, each processor works with its
 * part, which has lN = N^2/p unknowns.
 * Adapted from mpi14.cpp by Georg Stadler
 */
#include <stdio.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <mpi.h>
#include <string.h>

/* compuate global residual, assuming ghost values are updated */
double compute_residual(double *lu, int lN, double invhsq) {
  int i, j, lidx;
  double tmp, tmpsum, gres = 0.0, lres = 0.0;

  #pragma omp parallel for default(none) shared(lu,lN,invhsq) private(i,j,lidx,tmp,tmpsum) reduction(+:lres)
  for (i = 1; i <= lN; i++){
    tmpsum = 0.0;
    for (j = 1; j <= lN; j++){
      lidx = i+j*(lN+2); // center point of the stencil
      tmp = ((4.0*lu[lidx] - lu[lidx-1] - lu[lidx+1] - lu[lidx-lN] - lu[lidx+lN]) * invhsq - 1);
      tmpsum += tmp*tmp;
    }
    lres += tmpsum;
  }
  /* use allreduce for convenience; a reduce would also be sufficient */
  MPI_Allreduce(&lres, &gres, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return sqrt(gres);
}


int main(int argc, char * argv[]) {
  int mpirank, i, j, lidx, p, sqrtp, N, lN, iter, max_iters;
  MPI_Status status, status1, status2, status3;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank); // what number process?
  MPI_Comm_size(MPI_COMM_WORLD, &p); // how many processes?

  /* get name of host running MPI process */
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  printf("Rank %d/%d running on %s.\n", mpirank, p, processor_name);

  sscanf(argv[1], "%d", &N); // read inputs
  sscanf(argv[2], "%d", &max_iters);
# pragma omp parallel
  {
#ifdef _OPENMP
    int my_threadnum = omp_get_thread_num();
    int numthreads = omp_get_num_threads();
#else
    int my_threadnum = 0;
    int numthreads = 1;
#endif
    printf("Hello, I'm thread %d out of %d on mpirank %d\n", my_threadnum, numthreads, mpirank);
  }
  /* compute number of unknowns handled by each process */
  sqrtp = sqrt(p);
  lN = N / sqrtp;
  if ((N % sqrtp != 0) && mpirank == 0 ) {
    printf("N: %d, local N: %d\n", N, lN);
    printf("Exiting. Number of processes p must be a power 4^j. N must be a multiple of 2^j.\n");
    MPI_Abort(MPI_COMM_WORLD, 0);
  }
  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double tt = MPI_Wtime();

  /* Allocation of vectors, including left/upper and right/lower ghost points */
  double * lu    = (double *) calloc(sizeof(double), (lN + 2)*(lN + 2));
  double * lunew = (double *) calloc(sizeof(double), (lN + 2)*(lN + 2));
  double * lutemp;

  double h = 1.0 / (N + 1); // spacing in x and y directions
  double hsq = h * h;
  double invhsq = 1./hsq;
  double gres, gres0, tol = 1e-5;
  double tmpout[lN], tmpin[lN], tmp2out[lN], tmp2in[lN];

  /* initial residual */
  gres0 = compute_residual(lu, lN, invhsq);
  gres = gres0;

  // main iteration loop
  for (iter = 0; iter < max_iters && gres/gres0 > tol; iter++) {

    #pragma omp parallel for default(none) shared(lN,lunew,lu,hsq) private(j,lidx,tmpout,tmpin,tmp2out,tmp2in)
    /* Jacobi step for local points */
    for (i = 1; i <= lN; i++){
      for (j = 1; j <= lN; j++){
        lidx = i+j*(lN+2); // center point of the stencil
        lunew[lidx] = 0.25 * (hsq + lu[lidx - 1] + lu[lidx + 1] + lu[lidx - lN] + lu[lidx + lN]);
      }
    }

    /* communicate ghost values */
    if (mpirank % sqrtp > 0) {
      /* If not the left boundary, send/recv bdry values to the left */
      MPI_Send(&(lunew[lN+3]), lN, MPI_DOUBLE, mpirank-1, 124, MPI_COMM_WORLD);
      MPI_Recv(&(lunew[1]),    lN, MPI_DOUBLE, mpirank-1, 123, MPI_COMM_WORLD, &status);
    }
    if (mpirank % sqrtp < sqrtp - 1) {
      /* If not the right boundary, send/recv bdry values to the right */
      MPI_Send(&(lunew[(lN+2)*lN+1]),     lN, MPI_DOUBLE, mpirank+1, 123, MPI_COMM_WORLD);
      MPI_Recv(&(lunew[(lN+2)*(lN+1)+1]), lN, MPI_DOUBLE, mpirank+1, 124, MPI_COMM_WORLD, &status1);
    }
    if (mpirank < p - sqrtp) {
      /* If not the top boundary, send/recv bdry values up */
      for (j = 1; j <= lN; j++) tmpout[j-1] = lunew[lN + j*(lN+2)];
      MPI_Send(&tmpout, lN, MPI_DOUBLE, mpirank+sqrtp, 126, MPI_COMM_WORLD);
      MPI_Recv(&tmpin,  lN, MPI_DOUBLE, mpirank+sqrtp, 125, MPI_COMM_WORLD, &status2);
      for (j = 1; j <= lN; j++) lunew[lN + j*(lN+2) + 1] = tmpin[j-1];
    }
    if (mpirank >= sqrtp) {
      /* If not the bottom boundary, send/recv bdry values down */
      for (j = 1; j <= lN; j++) tmp2out[j-1] = lunew[1 + j*(lN+2)];
      MPI_Send(&tmp2out, lN, MPI_DOUBLE, mpirank-sqrtp, 125, MPI_COMM_WORLD);
      MPI_Recv(&tmp2in,  lN, MPI_DOUBLE, mpirank-sqrtp, 126, MPI_COMM_WORLD, &status3);
      for (j = 1; j <= lN; j++) lunew[j*(lN+2)] = tmp2in[j-1];
    }


    /* copy newu to u using pointer flipping */
    lutemp = lu; lu = lunew; lunew = lutemp;
    if (0 == (iter % 10)) {
      gres = compute_residual(lu, lN, invhsq);
      if (0 == mpirank) {
      	printf("Iter %d: Residual: %g\n", iter, gres);
      }
    }
  }

  /* Clean up */
  free(lu);
  free(lunew);

  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double elapsed = MPI_Wtime() - tt;
  if (0 == mpirank) {
    printf("Time elapsed is %f seconds.\n", elapsed);
  }
  MPI_Finalize();
  return 0;
}
