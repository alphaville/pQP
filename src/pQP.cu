#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include "cblas.h"
#include "pQP.cuh"
#include "lapacke.h"

int init_matrices(real_t *Q, real_t *h, real_t *V, real_t *W) {
	real_t *Q_init;
	ptrdiff_t i;
	if (Q == NULL) {
		fprintf(stderr, "Uninitialized matrix (Q) at line %d of %s\n", __LINE__,
				__FILE__);
		return EXIT_FAILURE;
	}
	if (h == NULL) {
		fprintf(stderr, "Uninitialized matrix (h) at line %d of %s\n", __LINE__,
				__FILE__);
		return EXIT_FAILURE;
	}
	if (V == NULL) {
		fprintf(stderr, "Uninitialized matrix (V) at line %d of %s\n", __LINE__,
				__FILE__);
		return EXIT_FAILURE;
	}
	if (W == NULL) {
		fprintf(stderr, "Uninitialized matrix (W) at line %d of %s\n", __LINE__,
				__FILE__);
		return EXIT_FAILURE;
	}
	Q_init = (real_t *) malloc(N * N * sizeof(*Q_init));
	/* Initialize Q with random data */
	srand(time(NULL)); // random seed (time-based)
	for (i = 0; i < N * N; i++) {
		Q_init[i] = (real_t) ((2 * rand() - RAND_MAX) % 1000 + 1) / 1000.0;
	}
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, Q_init,
			N, Q_init, N, 0.0, Q, N);
	for (i = 0; i < N; i++) {
		Q[i * (N + 1)] += 2.0 * N;
		h[i] = (real_t) ((2 * rand() - RAND_MAX) % 1000 + 1) / 1000.0;
		V[i * (2 * N + 1)] = 1;
		V[i * (2 * N + 1) + N] = -1;
		W[i] = 10.0;
		W[i + N] = -9.0;
	}
	free(Q_init);
	return EXIT_SUCCESS;
}

static void print_matrix(creal_t *A, const unsigned int nRows,
		const unsigned int nCols) {
	ptrdiff_t i, j;
	for (i = 0; i < nRows; i++) {
		if (i == 0) {
			printf("[\n");
		}
		for (j = 0; j < nCols; j++) {
			printf("%g ", A[j * nRows + i]);
		}
		if (i < nRows - 1) {
			printf(";\n");
		} else {
			printf("]\n");
		}
	}
}

void do_lapack() {
	char jobz, uplo;
	int n, lda, info;
	double *a;
	double *w;

	n = 1000;
	lda = 1000;
	jobz = 'V';
	uplo = 'U';
	a = (double*) malloc(sizeof(*a) * n * lda);
	w = (double*) malloc(sizeof(*w) * n);
	info = LAPACKE_dsyev(LAPACK_COL_MAJOR, jobz, uplo, n, a, lda, w);

	printf("Lapack status = %d\n", info);
}

/* Parameters */
//#define N 5
#define NRHS 3
#define LDA N
#define LDB NRHS

/* Auxiliary routines prototypes */
extern void print_matrix( char* desc, lapack_int m, lapack_int n, double* a, lapack_int lda );
extern void print_int_vector( char* desc, lapack_int n, lapack_int* a );

/* Main program */
int main() {
        /* Locals */
        lapack_int n = N, nrhs = NRHS, lda = LDA, ldb = LDB, info;
        /* Local arrays */
        lapack_int ipiv[N];
        double a[LDA*N] = {
            6.80, -6.05, -0.45,  8.32, -9.67,
           -2.11, -3.30,  2.58,  2.71, -5.14,
            5.66, 5.36, -2.70,  4.35, -7.26,
            5.97, -4.44,  0.27, -7.17, 6.08,
            8.23, 1.08,  9.04,  2.14, -6.87
        };
        double b[LDB*N] = {
            4.02, -1.56, 9.81,
            6.19,  4.00, -4.09,
           -8.22, -8.67, -4.57,
           -7.57,  1.75, -8.61,
           -3.03,  2.86, 8.99
        };
        /* Print Entry Matrix */
        print_matrix( "Entry Matrix A", n, n, a, lda );
        /* Print Right Rand Side */
        print_matrix( "Right Rand Side", n, nrhs, b, ldb );
        printf( "\n" );
        /* Executable statements */
        printf( "LAPACKE_dgesv (row-major, high-level) Example Program Results\n" );
        /* Solve the equations A*X = B */
        info = LAPACKE_dgesv( LAPACK_ROW_MAJOR, n, nrhs, a, lda, ipiv,
                        b, ldb );
        /* Check for the exact singularity */
        if( info > 0 ) {
                printf( "The diagonal element of the triangular factor of A,\n" );
                printf( "U(%i,%i) is zero, so that A is singular;\n", info, info );
                printf( "the solution could not be computed.\n" );
                exit( 1 );
        }
        /* Print solution */
        print_matrix( "Solution", n, nrhs, b, ldb );
        /* Print details of LU factorization */
        print_matrix( "Details of LU factorization", n, n, a, lda );
        /* Print pivot indices */
        print_int_vector( "Pivot indices", n, ipiv );
        exit( 0 );
} /* End of LAPACKE_dgesv Example */

/* Auxiliary routine: printing a matrix */
void print_matrix( char* desc, lapack_int m, lapack_int n, double* a, lapack_int lda ) {
        lapack_int i, j;
        printf( "\n %s\n", desc );
        for( i = 0; i < m; i++ ) {
                for( j = 0; j < n; j++ ) printf( " %6.2f", a[i*lda+j] );
                printf( "\n" );
        }
}

/* Auxiliary routine: printing a vector of integers */
void print_int_vector( char* desc, lapack_int n, lapack_int* a ) {
        lapack_int j;
        printf( "\n %s\n", desc );
        for( j = 0; j < n; j++ ) printf( " %6i", a[j] );
        printf( "\n" );
}
