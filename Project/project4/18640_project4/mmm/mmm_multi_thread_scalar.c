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
#include <pthread.h>

#define N 1024// matrix width. You can try 512, 1024, 2048.
#define NB 4 //  block width. N and NB must be evenly divisible.

// blocked building block.
void mmm_blocked_building_block(float *A, float *BT, float *CT) {
	int i, j, k;
	for (j = 0; j<NB; j++) {
		for (i = 0; i<NB; i++) {
			for (k = 0; k<NB; k++) {
				CT[j*N + i] = (CT[j*N + i] + (A[i*N + k] * BT[j*N + k]));
			}
		}
	}
}

// multi-threading
typedef struct {
	long threadid;
	float *A;
	float *B;
	float *C;
} thread_params;

// mult-threading blocked MMM
void * thread_func_multi_thread(void *threadparam) {
	int j, i, k;
	long threadid;
	float *A;
	float *BT;
	float *CT;

	thread_params * param = (thread_params *)threadparam;
	threadid = param->threadid;
	A = param->A;
	BT = param->B;
	CT = param->C;

	j = threadid * NB;
	for (i = 0; i < N; i += NB) {
		for (k = 0; k < N; k += NB) {
			mmm_blocked_building_block(&(A[i * N + k]), &(BT[j * N + k]), &(CT[j * N + i]));
		}
	}

	/* each thread instance calls pthread_exit( ) when
	finished to sync up with pthread_join() call in main(). */
	pthread_exit(NULL);
}

#define NUMTHREADS (N/NB)
void mmm_block_pthread(float *A, float *BT, float *CT) {

	/* allocate bookkeeping data structure for NUMTHREADS threads */
	pthread_t threads[NUMTHREADS];
	int rc;
	long threadid;

	thread_params params[NUMTHREADS];

	/* call pthread_create to spawn NUMTHREADS executions of
	Bash() as threads */
	for (threadid = 0; threadid < NUMTHREADS; threadid++) {
		params[threadid].threadid = threadid;
		params[threadid].A = A;
		params[threadid].B = BT;
		params[threadid].C = CT;

		/* spawn Bash as pthreads sending threadid as argument */
		rc = pthread_create(&threads[threadid], NULL, thread_func_multi_thread,
			(void *)&params[threadid]);
		if (rc) {
			printf("ERROR; return code from pthread_create() is %d\n", rc);
		}
	}

	/* The calls to pthread_join() will block until
	the designated thread (that is "threads[threadid]")
	has exited. */
	{
		void *status;
		for (threadid = 0; threadid < NUMTHREADS; threadid++) {
			pthread_join(threads[threadid], &status);
		}
	}
}

int main() {

	struct timeval start_time, end_time;

	float *A = (float*)malloc(N * N * sizeof(float)); // input matrix A
	float *B = (float*)malloc(N * N * sizeof(float)); // input matrix B
	float *BT = (float*)malloc(N * N * sizeof(float)); // B transpose for SIMD. 
	float *CT_multithread = (float*)malloc(N * N * sizeof(float));
	assert(A != NULL);
	assert(B != NULL);
	assert(BT != NULL);
	assert(CT_multithread != NULL);
	assert(N >= NB);
	assert((N%NB) == 0);

	// init matrices randomly
	int x, y;
	srand(time(NULL));
	for (x = 0; x < N; x++) {
		for (y = 0; y < N; y++) {
			A[x * N + y] = (float)rand() * 5 / RAND_MAX;
			B[x * N + y] = (float)rand() * 5 / RAND_MAX;
		}
	}
	for (x = 0; x < N; x++) {
		for (y = 0; y < N; y++) {
			BT[y * N + x] = B[x * N + y];							
			CT_multithread[y * N + x] = 0;
		}
	}

	double dur;

	// multi-threading blocked MMM
	gettimeofday(&start_time, NULL);
	mmm_block_pthread(A, BT, CT_multithread);
	gettimeofday(&end_time, NULL);
	dur = (end_time.tv_sec - start_time.tv_sec) * 1e6
		+ (end_time.tv_usec - start_time.tv_usec);
	printf(
		"%d-by-%d multi-threading blocked MMM: %0.1f (microsec) = %0.1f (sec)\n",
		N, N,
		(end_time.tv_sec - start_time.tv_sec) * 1e6
		+ (end_time.tv_usec - start_time.tv_usec), dur / 1000000);

	free(A);
	free(B);
	free(BT);
	free(CT_multithread);
	return 0;
}
