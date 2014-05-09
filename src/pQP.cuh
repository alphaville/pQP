/*
 * pQP.cuh
 *
 *      Created on: Feb 27, 2014
 *      Author: Pantelis Sopasakis
 */

/*
 *   pQP - CUDA implementation of the parallel QP algorithm by Brand et al.
 *   Copyright (C) 2014 Pantelis Sopasakis <pantelis.sopasakis@imtlucca.it>
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef PQP_CUH_
#define PQP_CUH_

/* Inclusions */
#include "cblas.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include "lapacke.h"


#define CUDA_CALL(value) do {		           								\
	cudaError_t _m_cudaStat = value;			     						\
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
#define NRHS 3
#define LDA N
#define LDB NRHS

typedef double real_t;
typedef const real_t creal_t;

int init_matrices(real_t *Q, real_t *h, real_t *V, real_t *W);

template<typename T> void print_matrix(char* desc, int matrix_order,
		lapack_int m, lapack_int n, T *a);

#endif /* PQP_CUH_ */
