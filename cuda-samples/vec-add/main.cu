#include <stdio.h>
#include <random>

const unsigned long N = 1lu << 16;
const unsigned long block_size = 256;
const float epsilon = 1e-5;

void rand_array(float* const array, const unsigned long size) {
	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<float> dist(-1.f, 1.f);

	for (unsigned long i = 0; i < size; i++) {
		array[i] = dist(mt);
	}
}

__global__ void vec_add_kernel(float* const C, const float* const A, const float* const B, const unsigned long size) {
	const unsigned long tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid >= size) return;

	C[tid] = A[tid] + B[tid];
}

int main() {
	// alloc
	float* const correct = (float*)malloc(sizeof(float) * N);
	float *A, *B, *C;
	cudaMallocManaged((void**)&A, sizeof(float) * N);
	cudaMallocManaged((void**)&B, sizeof(float) * N);
	cudaMallocManaged((void**)&C, sizeof(float) * N);

	// init
	rand_array(A, N);
	rand_array(B, N);
	for (unsigned long i = 0; i < N; i++) correct[i] = A[i] + B[i];

	// run kernel
	vec_add_kernel<<<(N + block_size - 1) / block_size, block_size>>>(C, A, B, N);

	// copy to host
	cudaMemcpy(C, C, sizeof(float) * N, cudaMemcpyDefault);

	// check
	unsigned long num_passed = 0;
	for (unsigned long i = 0; i < N; i++) {
		if (std::abs(correct[i] - C[i]) > epsilon) {
			printf("FAILED : [%7lu] C = %e, correct = %e, error = %e\n", i, C[i], correct[i], std::abs(correct[i] - C[i]));
			continue;
		}
		num_passed++;
	}
	printf("%5lu / %5lu passed\n", num_passed, N);

	// free
	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
	free(correct);
}
