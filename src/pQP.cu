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

int main(void) {

	do_lapack();
	real_t *Q = NULL, *h = NULL, *V = NULL, *W = NULL;
	ptrdiff_t i;

	Q = (real_t *) malloc(N * N * sizeof(*Q));
	h = (real_t *) malloc(N * sizeof(*h));
	V = (real_t *) malloc(2 * N * N * sizeof(*V));
	W = (real_t *) malloc(2 * N * sizeof(*W));
	init_matrices(Q, h, V, W);

	for (i = 0; i < N; i++)
		printf("h[%td]=%3.2f\n", i, h[i]);

	print_matrix(V, 2 * N, N);
	print_matrix(W, 2 * N, 1);

	return 0;
}
