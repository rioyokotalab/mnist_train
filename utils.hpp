#ifndef __UTILS_HPP__
#define __UTILS_HPP__
#include <random>
#include <string>
#include <iostream>

namespace utils {
inline void zero_init(float* const ptr, const std::size_t size) {
	for (std::size_t i = 0; i < size; i++) {
		ptr[i] = 0.0f;
	}
}

inline void random_init(float* const ptr, const std::size_t size, const float min_r = -1.f, const float max_r = 1.f) {
	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<float> dist(min_r, max_r);

	for (std::size_t i = 0; i < size; i++) {
		ptr[i] = dist(mt);
	}
}

inline void print_matrix(const float* const matrix, const std::size_t M, const std::size_t N, const std::size_t ldm, const std::string name) {
	std::printf("%s = \n", name.c_str());
	for (std::size_t m = 0; m < M; m++) {
		for (std::size_t n = 0; n < N; n++) {
			const float v = matrix[m + n * ldm];
			if (v < 0) {
				std::printf("%e ", v);
			} else {
				std::printf(" %e ", v);
			}
		}
		std::printf("\n");
	}
}

inline void print_mnist_image(const float* const image, const std::size_t dim = 28lu) {
	for (std::size_t n = 0; n < dim; n++) {
		for (std::size_t m = 0; m < dim; m++) {
			const auto v = image[m + dim * n] * 256.0f;
			if (v < 1.0f) {
				std::printf("  ");
			} else {
				std::printf("%02x", static_cast<unsigned int>(v));
			}
		}
		std::printf("\n");
	}
}
} // namespace utils

#endif /* end of include guard */
