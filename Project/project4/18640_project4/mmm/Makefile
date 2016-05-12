all: mmm_all_in_one mmm_single_thread_scalar mmm_single_thread_simd mmm_multi_thread_scalar mmm_multi_thread_simd

mmm_all_in_one: mmm_all_in_one.c
	gcc  -msse4 -g -o mmm_all_in_one mmm_all_in_one.c -lpthread

mmm_single_thread_scalar: mmm_single_thread_scalar.c
	gcc  -msse4 -g -o mmm_single_thread_scalar mmm_single_thread_scalar.c -lpthread

mmm_single_thread_simd: mmm_single_thread_simd.c
	gcc  -msse4 -g -o mmm_single_thread_simd mmm_single_thread_simd.c -lpthread

mmm_multi_thread_scalar: mmm_multi_thread_scalar.c
	gcc  -msse4 -g -o mmm_multi_thread_scalar mmm_multi_thread_scalar.c -lpthread

mmm_multi_thread_simd: mmm_multi_thread_simd.c
	gcc  -msse4 -g -o mmm_multi_thread_simd mmm_multi_thread_simd.c -lpthread

clean:
	del mmm_all_in_one.exe mmm_single_thread_scalar.exe mmm_single_thread_simd.exe mmm_multi_thread_scalar.exe mmm_multi_thread_simd.exe
