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
#define NB_SIMD 4 // block width for SIMD

// MMM in standard triple-loop
void mmmReference(float *A, float *B, float *C) {
  int i, j, k;
  for(j=0; j<N; j++) {
    for(i=0; i<N; i++) {
      for(k=0; k<N; k++) {
		C[i*N+j]=( C[i*N+j]+(A[i*N+k]*B[k*N+j]));
      }
    }
  }
}

// blocked building block.
void mmm_blocked_building_block(float *A, float *BT, float *CT) {
  int i, j, k;
  for(j=0; j<NB; j++) {
    for(i=0; i<NB; i++) {
      for(k=0; k<NB; k++) {
    	  CT[j*N+i]=( CT[j*N+i]+(A[i*N+k]*BT[j*N+k]));
      }
    }
  }
}

// blocked MMM
void mmm_blocked(float *A, float *BT, float *CT) {
  int j, i, k;
  for(j=0; j<N; j+=NB) {
    for (i=0; i<N; i+=NB) {
      for (k=0; k<N; k+=NB) {
    	  mmm_blocked_building_block(&(A[i*N+k]), &(BT[j*N+k]), &(CT[j*N+i]));
      }
    }
  }
}

// blocked simd building block
void mmm_simd_512_by_512_block_4_by_4(float AT[4*4], float BT[4*4], float CT[4*4]) {
	 __asm__ volatile (
			 "mov $0, %%ecx\n\t"
			 "jmp loop_cond%=\n\t"
			 "loop_body%=:\n\t"

			 "movaps 0(%[CT]), %%xmm6\n\t" // load a column of C to xmm6

			 "movaps 0(%[BT]), %%xmm0\n\t" // load a column of B to xmm0
			 "shufps $0, %%xmm0, %%xmm0\n\t" // vbroadcast B[0][0] to xmm0
			 "movaps 0(%[AT]), %%xmm2\n\t" // load a column of A to xmm2
			 "mulps %%xmm0, %%xmm2\n\t" // multiply xmm0 to xmm2

			 "movaps 0(%[BT]), %%xmm0\n\t" // load a column of B to xmm0
			 "shufps $0x55, %%xmm0, %%xmm0\n\t" // vbroadcast B[1][0] to xmm0
			 "movaps 2048(%[AT]), %%xmm3\n\t" // load the 2nd column of A to xmm3
			 "mulps %%xmm0, %%xmm3\n\t" // multiply xmm0 to xmm3
			 "addps %%xmm2, %%xmm3\n\t" // add xmm2 to xmm3

			 "movaps 0(%[BT]), %%xmm0\n\t" // load a column of B to xmm0
			 "shufps $0xAA, %%xmm0, %%xmm0\n\t" // vbroadcast B[2][0] to xmm0
			 "movaps 4096(%[AT]), %%xmm4\n\t" // load the 3rd column of A to xmm4
			 "mulps %%xmm0, %%xmm4\n\t" // multiply xmm0 to xmm4
			 "addps %%xmm3, %%xmm4\n\t" // add xmm3 to xmm4

			 "movaps 0(%[BT]), %%xmm0\n\t" // load a column of B to xmm0
			 "shufps $0xFF, %%xmm0, %%xmm0\n\t" // vbroadcast B[3][0] to xmm0
			 "movaps 6144(%[AT]), %%xmm5\n\t" // load the 4th column of A to xmm5
			 "mulps %%xmm0, %%xmm5\n\t" // multiply xmm0 to xmm5
			 "addps %%xmm4, %%xmm5\n\t" // add xmm4 to xmm5

			 "addps %%xmm5, %%xmm6\n\t" // add xmm5 to xmm6
			 "movaps %%xmm6, 0(%[CT])\n\t" // store xmm5 to a comumn of C

			 "loop_inc%=:\n\t"
			 "inc %%cx\n\t"
			 "add $2048, %[BT]\n\t"
			 "add $2048, %[CT]\n\t"
			 "loop_cond%=:\n\t"
			 "cmp $4, %%cx\n\t"
			 "jnz loop_body%=\n\t"
			 :
			 : [AT]"r"(AT), [BT]"r"(BT), [CT]"r"(CT)
			 : "ecx", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "memory");
}

void mmm_simd_1024_by_1024_block_4_by_4(float AT[4*4], float BT[4*4], float CT[4*4]) {
	 __asm__ volatile (
			 "mov $0, %%ecx\n\t"
			 "jmp loop_cond%=\n\t"
			 "loop_body%=:\n\t"

			 "movaps 0(%[CT]), %%xmm6\n\t" // load a column of C to xmm6

			 "movaps 0(%[BT]), %%xmm0\n\t" // load a column of B to xmm0
			 "shufps $0, %%xmm0, %%xmm0\n\t" // vbroadcast B[0][0] to xmm0
			 "movaps 0(%[AT]), %%xmm2\n\t" // load a column of A to xmm2
			 "mulps %%xmm0, %%xmm2\n\t" // multiply xmm0 to xmm2

			 "movaps 0(%[BT]), %%xmm0\n\t" // load a column of B to xmm0
			 "shufps $0x55, %%xmm0, %%xmm0\n\t" // vbroadcast B[1][0] to xmm0
			 "movaps 4096(%[AT]), %%xmm3\n\t" // load the 2nd column of A to xmm3
			 "mulps %%xmm0, %%xmm3\n\t" // multiply xmm0 to xmm3
			 "addps %%xmm2, %%xmm3\n\t" // add xmm2 to xmm3

			 "movaps 0(%[BT]), %%xmm0\n\t" // load a column of B to xmm0
			 "shufps $0xAA, %%xmm0, %%xmm0\n\t" // vbroadcast B[2][0] to xmm0
			 "movaps 8192(%[AT]), %%xmm4\n\t" // load the 3rd column of A to xmm4
			 "mulps %%xmm0, %%xmm4\n\t" // multiply xmm0 to xmm4
			 "addps %%xmm3, %%xmm4\n\t" // add xmm3 to xmm4

			 "movaps 0(%[BT]), %%xmm0\n\t" // load a column of B to xmm0
			 "shufps $0xFF, %%xmm0, %%xmm0\n\t" // vbroadcast B[3][0] to xmm0
			 "movaps 12288(%[AT]), %%xmm5\n\t" // load the 4th column of A to xmm5
			 "mulps %%xmm0, %%xmm5\n\t" // multiply xmm0 to xmm5
			 "addps %%xmm4, %%xmm5\n\t" // add xmm4 to xmm5

			 "addps %%xmm5, %%xmm6\n\t" // add xmm5 to xmm6
			 "movaps %%xmm6, 0(%[CT])\n\t" // store xmm5 to a comumn of C

			 "loop_inc%=:\n\t"
			 "inc %%cx\n\t"
			 "add $4096, %[BT]\n\t"
			 "add $4096, %[CT]\n\t"
			 "loop_cond%=:\n\t"
			 "cmp $4, %%cx\n\t"
			 "jnz loop_body%=\n\t"
			 :
			 : [AT]"r"(AT), [BT]"r"(BT), [CT]"r"(CT)
			 : "ecx", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "memory");
}

void mmm_simd_2048_by_2048_block_4_by_4(float AT[4*4], float BT[4*4], float CT[4*4]) {
	 __asm__ volatile (
			 "mov $0, %%ecx\n\t"
			 "jmp loop_cond%=\n\t"
			 "loop_body%=:\n\t"

			 "movaps 0(%[CT]), %%xmm6\n\t" // load a column of C to xmm6

			 "movaps 0(%[BT]), %%xmm0\n\t" // load a column of B to xmm0
			 "shufps $0, %%xmm0, %%xmm0\n\t" // vbroadcast B[0][0] to xmm0
			 "movaps 0(%[AT]), %%xmm2\n\t" // load a column of A to xmm2
			 "mulps %%xmm0, %%xmm2\n\t" // multiply xmm0 to xmm2

			 "movaps 0(%[BT]), %%xmm0\n\t" // load a column of B to xmm0
			 "shufps $0x55, %%xmm0, %%xmm0\n\t" // vbroadcast B[1][0] to xmm0
			 "movaps 8192(%[AT]), %%xmm3\n\t" // load the 2nd column of A to xmm3
			 "mulps %%xmm0, %%xmm3\n\t" // multiply xmm0 to xmm3
			 "addps %%xmm2, %%xmm3\n\t" // add xmm2 to xmm3

			 "movaps 0(%[BT]), %%xmm0\n\t" // load a column of B to xmm0
			 "shufps $0xAA, %%xmm0, %%xmm0\n\t" // vbroadcast B[2][0] to xmm0
			 "movaps 16384(%[AT]), %%xmm4\n\t" // load the 3rd column of A to xmm4
			 "mulps %%xmm0, %%xmm4\n\t" // multiply xmm0 to xmm4
			 "addps %%xmm3, %%xmm4\n\t" // add xmm3 to xmm4

			 "movaps 0(%[BT]), %%xmm0\n\t" // load a column of B to xmm0
			 "shufps $0xFF, %%xmm0, %%xmm0\n\t" // vbroadcast B[3][0] to xmm0
			 "movaps 24576(%[AT]), %%xmm5\n\t" // load the 4th column of A to xmm5
			 "mulps %%xmm0, %%xmm5\n\t" // multiply xmm0 to xmm5
			 "addps %%xmm4, %%xmm5\n\t" // add xmm4 to xmm5

			 "addps %%xmm5, %%xmm6\n\t" // add xmm5 to xmm6
			 "movaps %%xmm6, 0(%[CT])\n\t" // store xmm5 to a comumn of C

			 "loop_inc%=:\n\t"
			 "inc %%cx\n\t"
			 "add $8192, %[BT]\n\t"
			 "add $8192, %[CT]\n\t"
			 "loop_cond%=:\n\t"
			 "cmp $4, %%cx\n\t"
			 "jnz loop_body%=\n\t"
			 :
			 : [AT]"r"(AT), [BT]"r"(BT), [CT]"r"(CT)
			 : "ecx", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "memory");
}

// blocked SIMD MMM
void mmm_blocked_simd(float *AT, float *BT, float *CT) {
  int j, i, k;

  for(j=0; j<N; j+=NB_SIMD) {
    for (i=0; i<N; i+=NB_SIMD) {
      for (k=0; k<N; k+=NB_SIMD) {
#if defined N
#if N == 512
    	  mmm_simd_512_by_512_block_4_by_4(&(AT[k*N+i]), &(BT[j*N+k]), &(CT[j*N+i]));
#elif N == 1024
    	  mmm_simd_1024_by_1024_block_4_by_4(&(AT[k*N+i]), &(BT[j*N+k]), &(CT[j*N+i]));
#elif N == 2048
    	  mmm_simd_2048_by_2048_block_4_by_4(&(AT[k*N+i]), &(BT[j*N+k]), &(CT[j*N+i]));
#endif
#endif
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
				(void *) &params[threadid]);
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

// mult-threading SIMD blocked MMM
void * thread_func_multi_thread_simd(void *threadparam) {
	int j, i, k;
	long threadid;
	float *AT;
	float *BT;
	float *CT;

	thread_params * param = (thread_params *)threadparam;
	threadid = param->threadid;
	AT = param->A;
	BT = param->B;
	CT = param->C;

	j = threadid * NB_SIMD;

	for (i = 0; i<N; i += NB_SIMD) {
		for (k = 0; k<N; k += NB_SIMD) {
#if defined N
#if N == 512
			mmm_simd_512_by_512_block_4_by_4(&(AT[k*N + i]), &(BT[j*N + k]), &(CT[j*N + i]));
#elif N == 1024
			mmm_simd_1024_by_1024_block_4_by_4(&(AT[k*N + i]), &(BT[j*N + k]), &(CT[j*N + i]));
#elif N == 2048
			mmm_simd_2048_by_2048_block_4_by_4(&(AT[k*N + i]), &(BT[j*N + k]), &(CT[j*N + i]));
#endif
#endif
		}
	}
	/* each thread instance calls pthread_exit( ) when
	finished to sync up with pthread_join() call in main(). */
	pthread_exit(NULL);
}

#undef NUMTHREADS
#define NUMTHREADS (N/NB_SIMD)
void mmm_blocked_simd_pthread(float *AT, float *BT, float *CT) {

	/* allocate bookkeeping data structure for NUMTHREADS threads */
	pthread_t threads[NUMTHREADS];
	int rc;
	long threadid;

	thread_params params[NUMTHREADS];

	/* call pthread_create to spawn NUMTHREADS executions of
	Bash() as threads */
	for (threadid = 0; threadid < NUMTHREADS; threadid++) {
		params[threadid].threadid = threadid;
		params[threadid].A = AT;
		params[threadid].B = BT;
		params[threadid].C = CT;

		/* spawn Bash as pthreads sending threadid as argument */
		rc = pthread_create(&threads[threadid], NULL, thread_func_multi_thread_simd,
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

// helper functions
void *aligned_malloc(size_t bytes, size_t alignment) {
	void *p1, *p2; // basic pointer needed for computation.

	/* We need to use malloc provided by C. First we need to allocate memory
	 of size bytes + alignment + sizeof(size_t) . We need 'bytes' because
	 user requested it. We need to add 'alignment' because malloc can give us
	 any address and we need to find multiple of 'alignment', so at maximum multiple
	 of alignment will be 'alignment' bytes away from any location. We need
	 'sizeof(size_t)' for implementing 'aligned_free', since we are returning modified
	 memory pointer, not given by malloc ,to the user, we must free the memory
	 allocated by malloc not anything else. So I am storing address given by malloc just above
	 pointer returning to user. Thats why I need extra space to store that address.
	 Then I am checking for error returned by malloc, if it returns NULL then
	 aligned_malloc will fail and return NULL.
	 */
	if ((p1 = (void *) malloc(bytes + alignment + sizeof(size_t))) == NULL)
		return NULL;

	/*	Next step is to find aligned memory address multiples of alignment.
	 By using basic formule I am finding next address after p1 which is
	 multiple of alignment.I am storing new address in p2.
	 */
	size_t addr = (size_t) p1 + alignment + sizeof(size_t);

	p2 = (void *) (addr - (addr % alignment));

	/*	Final step, I am storing the address returned by malloc 'p1' just "size_t"
	 bytes above p2, which will be useful while calling aligned_free.
	 */
	*((size_t *) p2 - 1) = (size_t) p1;

	return p2;
}

void aligned_free(void *p) {
	/*	Find the address stored by aligned_malloc ,"size_t" bytes above the
	 current pointer then free it using normal free routine provided by C.
	 */
	free((void *) (*((size_t *) p - 1)));
}

void verify_C(float* C_v, int transpose, float* C_ref) {
	int x, y;
	if (transpose != 0) {
		for (x = 0; x < N; x++) {
			for (y = 0; y < N; y++) {
				float ref = C_ref[y*N + x];
				float delta = C_v[x*N + y] - ref;
				if ( (ref == 0 && (delta > 0.01 || delta < -0.01))
					|| (delta / ref > 0.01 || delta / ref < -0.01) ){
					printf("C_v[%d][%d] = %f, C_ref[%d][%d] = %f\n", x, y, C_v[x*N + y], y, x, C_ref[y*N + x]);
					assert(C_v[x*N + y] == C_ref[y*N + x]);
				}
			}
		}
	}
	else {
		for (x = 0; x < N; x++) {
			for (y = 0; y < N; y++) {
				float ref = C_ref[x*N + y];
				float delta = C_v[x*N + y] - ref;
				if ((ref == 0 && (delta > 0.01 || delta < -0.01))
					|| (delta / ref > 0.01 || delta / ref < -0.01)) {
					printf("C_v[%d][%d] = %f, C_ref[%d][%d] = %f\n", x, y, C_v[x*N + y], x, y, C_ref[y*N + x]);
					assert(C_v[x*N + y] == C_ref[x*N + y]);
				}
			}
		}
	}
	printf("result verified\n\n");
}

int main() {

	struct timeval start_time, end_time;

	float *A = (float*) malloc(N * N * sizeof(float)); // input matrix A
	float *B = (float*) malloc(N * N * sizeof(float)); // input matrix B
	float *AT = (float*) aligned_malloc( N * N * sizeof(float), 16); // A transpose for SIMD. 16 Bytes aligned.
	float *BT = (float*) aligned_malloc( N * N * sizeof(float), 16); // B transpose for SIMD. 16 Bytes aligned.
	float *C_reference = (float*) malloc(N * N * sizeof(float)); // output matrix C for reference
	float *CT_blocked = (float*) malloc(N * N * sizeof(float));
	float *CT_multithread = (float*) malloc(N * N * sizeof(float));
	float *CT_simd = (float*) aligned_malloc(N * N * sizeof(float), 16); // C transpose
	float *CT_multithread_simd = (float*) aligned_malloc(N * N * sizeof(float),
			16); // C transpose
	assert(A!=NULL);
	assert(B!=NULL);
	assert(AT!=NULL);
	assert(BT!=NULL);
	assert(C_reference!=NULL);
	assert(CT_multithread != NULL);
	assert(CT_blocked != NULL);
	assert(CT_simd != NULL);
	assert(CT_multithread_simd != NULL);
	assert(N>=NB);
	assert((N%NB)==0);

	// init matrices randomly
	int x, y;
	srand(time(NULL));
	for (x = 0; x < N; x++) {
		for (y = 0; y < N; y++) {
			A[x * N + y] = (float) rand() * 5 / RAND_MAX;
			B[x * N + y] = (float) rand() * 5 / RAND_MAX;
			C_reference[x * N + y] = 0;

		}
	}
	for (x = 0; x < N; x++) {
		for (y = 0; y < N; y++) {
			AT[y * N + x] = A[x * N + y];
			BT[y * N + x] = B[x * N + y];							
			CT_blocked[y * N + x] = 0;
			CT_simd[y * N + x] = 0;								
			CT_multithread[y * N + x] = 0;
			CT_multithread_simd[y * N + x] = 0;
		}
	}

	double dur;

	// referenced MMM
	gettimeofday(&start_time, NULL);
	mmmReference(A, B, C_reference);
	gettimeofday(&end_time, NULL);
	dur = (end_time.tv_sec - start_time.tv_sec) * 1e6
			+ (end_time.tv_usec - start_time.tv_usec);
	printf("%d-by-%d reference: %0.1f (microsec) = %0.1f (sec)\n\n",
	N, N,
			(end_time.tv_sec - start_time.tv_sec) * 1e6
					+ (end_time.tv_usec - start_time.tv_usec), dur / 1000000);

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
	verify_C(CT_blocked, 1, C_reference);

	// single thread blocked SIMD MMM */
	gettimeofday(&start_time, NULL);
	mmm_blocked_simd(AT, BT, CT_simd);
	gettimeofday(&end_time, NULL);
	dur = (end_time.tv_sec - start_time.tv_sec) * 1e6
			+ (end_time.tv_usec - start_time.tv_usec);
	printf(
			"%d-by-%d single thread blocked SIMD MMM by %d: %0.1f (microsec)  = %0.1f (sec)\n",
			N, N, NB_SIMD,
			(end_time.tv_sec - start_time.tv_sec) * 1e6
					+ (end_time.tv_usec - start_time.tv_usec), dur / 1000000);
	verify_C(CT_simd, 1, C_reference);

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
	verify_C(CT_multithread, 1, C_reference);

	// multi-threading blocked SIMD MMM
	gettimeofday(&start_time, NULL);
	mmm_blocked_simd_pthread(AT, BT, CT_multithread_simd);
	gettimeofday(&end_time, NULL);
	dur = (end_time.tv_sec - start_time.tv_sec) * 1e6
			+ (end_time.tv_usec - start_time.tv_usec);
	printf(
			"%d-by-%d multi-threading blocked SIMD MMM: %0.1f (microsec) = %0.1f (sec)\n",
			N, N,
			(end_time.tv_sec - start_time.tv_sec) * 1e6
					+ (end_time.tv_usec - start_time.tv_usec), dur / 1000000);
	verify_C(CT_multithread_simd, 1, C_reference);


	free(A);
	free(B);
	aligned_free(AT);
	aligned_free(BT);
	free(C_reference);
	free(CT_multithread);
	free(CT_blocked);
	aligned_free(CT_simd);
	aligned_free(CT_multithread_simd);
	return 0;
}
