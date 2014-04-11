/*
 * pQP.cuh
 *
 *      Created on: Feb 27, 2014
 *      Author: Pantelis Sopasakis
 */

#ifndef PQP_CUH_
#define PQP_CUH_

#include "cblas.h"

#define CUDA_CALL(value) do {		           								\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);        													\
	} } while(0)

#define CUBLAS_CALL(value) do {                                                           \
	cublasStatus_t _m_status = value;                                                             \
	if (_m_status != CUBLAS_STATUS_SUCCESS){                                                      \
		fprintf(stderr, "Error %d at line %d in file %s\n", (int)_m_status, __LINE__, __FILE__);  \
	exit(14);                                                                                     \
	}                                                                                             \
	} while(0)

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) {  \
    printf("Error at %s:%d\n",__FILE__,__LINE__);             \
    exit(115);}} while(0)

#define N 5

typedef double real_t;
typedef const real_t creal_t;

int init_matrices(real_t *Q, real_t *h, real_t *V, real_t *W);

static void print_matrix(creal_t *A, const unsigned int nRows,
		const unsigned int nCols);



#endif /* PQP_CUH_ */
