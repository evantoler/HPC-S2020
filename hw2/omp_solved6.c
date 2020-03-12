/******************************************************************************
* FILE: omp_bug6.c
* DESCRIPTION:
*   This program compiles and runs fine, but produces the wrong result.
*   Compare to omp_orphan.c.
* AUTHOR: Blaise Barney  6/05
* LAST REVISED: 06/30/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define VECLEN 100

float a[VECLEN], b[VECLEN];

float dotprod (float* sum_ptr)
{
int i,tid;
float my_sum = 0.0; // use a thread private variable to track the sum

tid = omp_get_thread_num();
#pragma omp for
  for (i=0; i < VECLEN; i++)
    {
    my_sum = my_sum + (a[i]*b[i]);
    printf("  tid= %d i=%d\n",tid,i);
    }
#pragma omp critical
  *sum_ptr += my_sum;
}


int main (int argc, char *argv[]) {
int i;
float sum;
float* sum_ptr = &sum; // Evan: added a pointer to get around private variables in the reduction

for (i=0; i < VECLEN; i++)
  a[i] = b[i] = 1.0 * i;
sum = 0.0;

// Evan: share a *pointer* to the sum instead of the sum itself
#pragma omp parallel shared(sum_ptr)
  dotprod(sum_ptr); // Evan: pass the pointer to the function

printf("Sum = %f\n",sum);

}
