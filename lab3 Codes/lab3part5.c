/* File:     lab3part5.c
 *
 * Purpose:  Implement parallel Gaussian elimination with OpenMP
 *
 * Compile:  gcc ...
 *                 
 * Run:      ./lab3part5 <number of threads> <n>
 *
 * Input:    None
 * Output:   Max error in solution, runtimes of Gaussian elimination,
 *           back substitution, and total runtime.  If DEBUG flag is set,
 *           matrix, right-hand side, triangular system, and actual solution.
 *           
 *
 * Notes:
 * 1.  Diagonal entries in coefficient matrix are n/10.  Other entries are random doubles between 0 and 1. 
 * 2.  Solution to system is x[i] = 1. This gives the value of the constraint vector.
 * 3.  Gaussian elimination overwrites A with upper triangular matrix.
 * 4.  It is assumed that row-swaps are NOT necessary.
 *
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

void Get_args(int argc, char* argv[], int* thread_count_p, int* n_p);
void Init(double A[], double b[], double x[], int n);
void Gaussian_elim(double A[], double b[], int n, int thread_count);
void Row_solve(double A[], double b[], double x[], int n, int thread_count);
double Find_error(double x[], int n);
void Print_mat(char title[], double A[], int n);
void Print_vect(char title[], double x[], int n);

int main(int argc, char* argv[]) {
   int n, thread_count;
   double *A, *b, *x;
   double ge_start, ge_finish, bs_start, bs_finish, start, finish;

   Get_args(argc, argv, &thread_count, &n);

   A = malloc(n*n*sizeof(double));
   b = malloc(n*sizeof(double));
   x = malloc(n*sizeof(double));

   Init(A, b, x, n);
#  ifdef DEBUG
   Print_mat("A = ", A, n);
   Print_vect("b = ", b, n);
#  endif

   ge_start = start = omp_get_wtime();
   Gaussian_elim(A, b, n, thread_count);
   ge_finish = bs_start = omp_get_wtime();
#  ifdef DEBUG
   Print_mat("Triangular system = ", A, n);
   Print_vect("Updated RHS = ", b, n);
#  endif

   Row_solve(A, b, x, n, thread_count); 
   bs_finish = finish = omp_get_wtime();
   printf("Max error in solution = %e\n", Find_error(x,n));
   printf("Time for Gaussian elim = %e seconds\n", ge_finish-ge_start);
   printf("Time for back sub = %e seconds\n", bs_finish-bs_start);
   printf("Total time for solve = %e seconds\n", finish-start);
#  ifdef DEBUG
   Print_vect("Solution =", x, n);
#  endif

   free(A);
   free(b);
   free(x);

   return 0;
}  /* main */

/*--------------------------------------------------------------------*/
void Get_args(int argc, char* argv[], int* thread_count_p, int* n_p) {
   if (argc != 3) {
      fprintf(stderr, "usage: %s <thread_count> <n>\n", argv[0]);
      exit(0);
   }
   *thread_count_p = strtol(argv[1], NULL, 10);
   *n_p = strtol(argv[2], NULL, 10);
}  /* Get_args */


/*--------------------------------------------------------------------*/
void Init(double A[], double b[], double x[], int n) {
   int i, j;

   for (i = 0; i < n; i++)
      x[i] = 1.0;

   srandom(1);
   for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++)
         if (i != j) 
            A[i*n + j] = random()/((double) RAND_MAX);
         else
            A[i*n+i] = n/10.0;
   }

   for (i = 0; i < n; i++) {
      b[i] = 0;
      for (j = 0; j < n; j++)
         b[i] += A[i*n + j]*x[j];
   }

   memset(x, 0, n*sizeof(double));
}  /* Init */


/*--------------------------------------------------------------------
 * Function:     Gaussian_elim
 * Purpose:      Convert A to an upper triangular system
 * In args:      n, thread_count
 * In/out args:  A, b
 * Note:         It's assumed that row-swaps aren't necessary
 */
void Gaussian_elim(double A[], double b[], int n, int thread_count){
   int i, j, k;
   double fact;

   # pragma omp parallel for 
   for (i = 0; i < n-1; i++)
      for (k = i+1; k < n; k++) {
         fact = -A[k*n + i]/A[i*n + i];
         A[k*n + i] = 0;
         for (j = i+1; j < n; j++)
            A[k*n + j] += fact*A[i*n + j];
         b[k] += fact*b[i];
      }
}  /* Gaussian_elim */

/*--------------------------------------------------------------------
 * Function:  Row_solve
 * Purpose:   Solve a triangular system using the row-oriented algorithm
 * In args:   A, b, n, thread_count
 * Out arg:   x
 */
void Row_solve(double A[], double b[], double x[], int n, int thread_count) {
   int i, j;
   double tmp;

   for (i = n-1; i >= 0; i--) {
      tmp = b[i];
      for (j = i+1; j < n; j++)
         tmp += -A[i*n+j]*x[j];
      #pragma omp parallel num_threads(thread_count)
      {
         x[i] = tmp/A[i*n+i];
#        ifdef DEBUG
         printf("x[%d] = %.1f\n", i, x[i]);
#        endif
      }
   }
}  /* Row_solve */

/*--------------------------------------------------------------------*/
double Find_error(double x[], int n) {
   int i;
   double error = 0.0, tmp;

   for (i = 0; i < n; i++) {
      tmp = fabs(x[i] - 1.0);
      if (tmp > error) error = tmp;
   }
   return error;
}  /* Find_error */


/*--------------------------------------------------------------------*/
void Print_mat(char title[], double A[], int n) {
   int i, j;

   printf("%s:\n", title);
   for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++)
         printf("%4.1f ", A[i*n+j]);
      printf("\n");
   }
   printf("\n");
}  /* Print_mat */


/*--------------------------------------------------------------------*/
void Print_vect(char title[], double x[], int n) {
   int i;

   printf("%s ", title);
   for (i = 0; i < n; i++)
      printf("%.1f ", x[i]);
   printf("\n");
}  /* Print_vect */
