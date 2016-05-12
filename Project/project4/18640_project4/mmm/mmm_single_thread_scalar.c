/*
 * 18-640 F15 Project 4
 * mmm.c: Matrix-Matrix Multiplication
 *   Guanglin Xu <guanglin@andrew.cmu.edu>
 *
 * some helper functions are acquired from Internet.
 *
 */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>

#define N 1024// matrix width
#define NB 32 //  block width. N and NB must be evenly divisible.

// blocked building block.
void mmm_blocked_building_block(float *A, float *B, float *C) {
  int i, j, k;
  for(j=0; j<NB; j++) {
    for(i=0; i<NB; i++) {
      for(k=0; k<NB; k++) {
    	  C[i*N+j]=( C[i*N+j]+(A[i*N+k]*B[k*N+j]));
      }
    }
  }
}

// blocked MMM
void mmm_blocked(float *A, float *B, float *C) {
  int j, i, k;
  for(j=0; j<N; j+=NB) {
    for (i=0; i<N; i+=NB) {
      for (k=0; k<N; k+=NB) {
    	  mmm_blocked_building_block(&(A[i*N+k]), &(B[k*N+j]), &(C[i*N+j]));
      }
    }
  }
}


int main() {

	struct timeval start_time, end_time;

	float *A = (float*) malloc(N * N * sizeof(float)); // input matrix A
	float *B = (float*) malloc(N * N * sizeof(float)); // input matrix B
	float *BT = (float*)malloc(N * N * sizeof(float)); // B transpose for SIMD. 
	float *CT_blocked = (float*) malloc(N * N * sizeof(float));
	assert(A!=NULL);
	assert(B!=NULL);
	assert(BT != NULL);
	assert(CT_blocked != NULL);
	assert(N>=NB);
	assert((N%NB)==0);

	// init matrices randomly
	int x, y;
	srand(time(NULL));
	for (x = 0; x < N; x++) {
		for (y = 0; y < N; y++) {
			A[x * N + y] = (float) rand() * 5 / RAND_MAX;
			B[x * N + y] = (float) rand() * 5 / RAND_MAX;
		}
	}
	for (x = 0; x < N; x++) {
		for (y = 0; y < N; y++) {
			BT[y * N + x] = B[x * N + y];
			CT_blocked[y * N + x] = 0;
		}
	}

	double dur;

	// single thread blocked MMM
	gettimeofday(&start_time, NULL);
	mmm_blocked(A, BT, CT_blocked);
	gettimeofday(&end_time, NULL);
	dur = (end_time.tv_sec - start_time.tv_sec) * 1e6
			+ (end_time.tv_usec - start_time.tv_usec);
	printf(
			"%d-by-%d single thread blocked MMM: %0.1f (microsec) = %0.1f (sec)\n",
			N, N,
			(end_time.tv_sec - start_time.tv_sec) * 1e6
					+ (end_time.tv_usec - start_time.tv_usec), dur / 1000000);

	free(A);
	free(B);
	free(BT);
	free(CT_blocked);
	return 0;
}
