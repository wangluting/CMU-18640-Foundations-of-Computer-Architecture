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
#define NB_SIMD 4 // block width for SIMD

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

int main() {

	struct timeval start_time, end_time;

	float *AT = (float*) aligned_malloc( N * N * sizeof(float), 16); // A transpose for SIMD. 16 Bytes aligned.
	float *BT = (float*) aligned_malloc( N * N * sizeof(float), 16); // B transpose for SIMD. 16 Bytes aligned.
	float *CT_simd = (float*) aligned_malloc(N * N * sizeof(float), 16); // C transpose
	assert(AT!=NULL);
	assert(BT!=NULL);
	assert(CT_simd != NULL);

	// init matrices randomly
	int x, y;
	srand(time(NULL));
	for (x = 0; x < N; x++) {
		for (y = 0; y < N; y++) {
			AT[x * N + y] = (float) rand() * 5 / RAND_MAX;
			BT[x * N + y] = (float) rand() * 5 / RAND_MAX;
			CT_simd[x * N + y] = 0;
		}
	}

	double dur;

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

	aligned_free(AT);
	aligned_free(BT);
	aligned_free(CT_simd);
	return 0;
}
