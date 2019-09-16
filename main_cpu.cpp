#include <iostream>
#include <random>
#include <cmath>
#include <mnist.hpp>
#include <utils.hpp>

constexpr std::size_t minibatch_size = 64;
constexpr std::size_t num_iterations = 1000;

constexpr std::size_t input_size = mnist_loader::IMAGE_DIM * mnist_loader::IMAGE_DIM;
constexpr std::size_t hidden_size = 100;
constexpr std::size_t output_size = mnist_loader::CLASS_SIZE;

constexpr std::size_t print_info_interval = 20;

constexpr float learning_rate = 0.5f;

void matmul(float* const C, const float* const A, const float* const B, const std::size_t M, const std::size_t N, const std::size_t K) {
	for (std::size_t m = 0; m < M; m++) {
		for (std::size_t n = 0; n < N; n++) {
			float sum = 0.0f;
			for (std::size_t k = 0; k < K; k++) {
				sum += A[m + k * M] * B[k + n * K];
			}
			C[m + M * n] = sum;
		}
	}
}

void matmul_tn(float* const C, const float* const A, const float* const B, const std::size_t M, const std::size_t N, const std::size_t K) {
	for (std::size_t m = 0; m < M; m++) {
		for (std::size_t n = 0; n < N; n++) {
			float sum = 0.0f;
			for (std::size_t k = 0; k < K; k++) {
				sum += A[k + m * K] * B[k + n * K];
			}
			C[m + M * n] = sum;
		}
	}
}

void gemm_nt(const float beta, float* const C, const float alpha, const float* const A, const float* const B, const std::size_t M, const std::size_t N, const std::size_t K) {
	for (std::size_t m = 0; m < M; m++) {
		for (std::size_t n = 0; n < N; n++) {
			float sum = 0.0f;
			for (std::size_t k = 0; k < K; k++) {
				sum += A[m + k * M] * B[n + k * N];
			}
			C[m + M * n] = beta * C[m + M * n] + alpha * sum;
		}
	}
}

void elementwise_product(float* const C, const float* const A, const float* const B, const std::size_t size) {
	for (std::size_t i = 0; i < size; i++) {
		C[i] = A[i] * B[i];
	}
}

void add_bias(float* const A, const float* const bias, const std::size_t layer_size, const std::size_t minibatch_size) {
	for (std::size_t mb = 0; mb < minibatch_size; mb++) {
		for (std::size_t ls = 0; ls < layer_size; ls++) {
			A[mb * layer_size + ls] += bias[ls];
		}
	}
}

void ReLU(float* const acted, const float* const pre_act, const std::size_t size, const std::size_t minibatch_size) {
	for (std::size_t i = 0; i < size * minibatch_size; i++) {
		acted[i] = std::max(0.0f, pre_act[i]);
	}
}

void dReLU(float* const d_acted, const float* const pre_act, const std::size_t size, const std::size_t minibatch_size) {
	for (std::size_t i = 0; i < size * minibatch_size; i++) {
		if (pre_act[i] < 0.0f) {
			d_acted[i] = 0.0f;
		} else {
			d_acted[i] = 1.0f;
		}
	}
}

void softmax(float* const acted, const float* const pre_act, const std::size_t layer_size, const std::size_t minibatch_size) {
	for (std::size_t mb = 0; mb < minibatch_size; mb++) {
		float e_sum = 0.0f;
		for (std::size_t ls = 0; ls < layer_size; ls++) {
			const float e = std::exp(pre_act[mb * layer_size + ls] - pre_act[mb * layer_size + 0]);
			acted[mb * layer_size + ls] = e;
			e_sum += e;
		}
		for (std::size_t ls = 0; ls < layer_size; ls++) {
			acted[mb * layer_size + ls] /= e_sum;
		}
	}
}

float compute_accuracy(const float* const forwarded_data, const float* const correct_data, const std::size_t size, const std::size_t minibatch_size) {
	std::size_t num_correct = 0;
	for (std::size_t mb = 0; mb < minibatch_size; mb++) {
		std::size_t max_index = 0;
		for (std::size_t i = 1; i < size; i++) {
			if (forwarded_data[mb * size + i] > forwarded_data[mb * size + max_index]) {
				max_index = i;
			}
		}
		if (correct_data[mb * size + max_index] > 0.5f) {
			num_correct++;
		}
	}
	return (float)num_correct / minibatch_size;
}

float compute_loss(const float* const forwarded_data, const float* const correct_data, const std::size_t size, const std::size_t minibatch_size) {
	float loss = 0.0f;
	for (std::size_t mb = 0; mb < minibatch_size; mb++) {
		std::size_t correct_index = 0;
		for (std::size_t i = 0; i < size; i++) {
			if (correct_data[mb * size + i] > 0.5f) {
				correct_index = i;
			}
		}
		loss -= std::log(forwarded_data[mb * size + correct_index]);
	}
	return loss / minibatch_size;
}

void compute_last_error(float* const last_error, const float* const last_act, const float* const correct_data, const std::size_t output_size, const std::size_t minibatch_size) {
	for (std::size_t i = 0; i < output_size * minibatch_size; i++) {
		last_error[i] = last_act[i] - correct_data[i];
	}
}

void update_weight(float* const W, const float* const error, const float* const acted, const std::size_t W_M, const std::size_t W_N, const std::size_t minibatch_size, const float learning_rate) {
	gemm_nt(1.0f, W, - learning_rate / minibatch_size, error, acted, W_M, W_N, minibatch_size);
}

void update_bias(float* const bias, const float* const error, const std::size_t W_M, const std::size_t minibatch_size, const float learning_rate) {
	for (std::size_t i = 0; i < W_M; i++) {
		float sum = 0.0f;
		for (std::size_t mb = 0; mb < minibatch_size; mb++) {
			sum += error[mb * W_M + i];
		}
		bias[i] -= learning_rate / minibatch_size * sum;
	}
}

int main() {
	mnist_loader train_data, test_data;
	train_data.load("train-images-idx3-ubyte", "train-labels-idx1-ubyte");
	test_data.load("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");
	float* const minibatch_image_data = (float*)malloc(minibatch_size * mnist_loader::IMAGE_DIM * mnist_loader::IMAGE_DIM * sizeof(float));
	float* const minibatch_label_data = (float*)malloc(minibatch_size * mnist_loader::CLASS_SIZE * sizeof(float));

	float* const minibatch_hidden_data_pre = (float*)malloc(minibatch_size * hidden_size * sizeof(float));
	float* const minibatch_hidden_data = (float*)malloc(minibatch_size * hidden_size * sizeof(float));
	float* const minibatch_hidden_error = (float*)malloc(minibatch_size * hidden_size * sizeof(float));
	float* const minibatch_output_data_pre = (float*)malloc(minibatch_size * output_size * sizeof(float));
	float* const minibatch_output_data = (float*)malloc(minibatch_size * output_size * sizeof(float));
	float* const minibatch_output_error = (float*)malloc(minibatch_size * output_size * sizeof(float));

	float* const layer0_weight = (float*)malloc(input_size * hidden_size * sizeof(float));
	float* const layer0_bias = (float*)malloc(hidden_size * sizeof(float));
	float* const layer1_weight = (float*)malloc(hidden_size * output_size * sizeof(float));
	float* const layer1_bias = (float*)malloc(output_size * sizeof(float));
	utils::random_init(layer0_weight, input_size * hidden_size);
	utils::random_init(layer0_bias, hidden_size);
	utils::random_init(layer1_weight, hidden_size * output_size);
	utils::random_init(layer1_bias, output_size);

	std::mt19937 mt(std::random_device{}());
	std::uniform_int_distribution<std::size_t> image_dist(0, train_data.get_num_data());

	// training loop
	for (std::size_t i = 0; i < num_iterations; i++) {
		// load minibatch
		for (std::size_t d = 0; d < minibatch_size; d++) {
			const std::size_t image_id = image_dist(mt);
			train_data.copy(image_id, minibatch_image_data + d * mnist_loader::IMAGE_DIM * mnist_loader::IMAGE_DIM, minibatch_label_data + d * mnist_loader::CLASS_SIZE);
		}

		//
		// Forward
		//
		matmul(
			minibatch_hidden_data_pre,
			layer0_weight,
			minibatch_image_data,
			hidden_size,
			minibatch_size,
			input_size);
		add_bias(
			minibatch_hidden_data_pre,
			layer0_bias,
			hidden_size,
			minibatch_size);
		ReLU(
			minibatch_hidden_data,
			minibatch_hidden_data_pre,
			hidden_size, minibatch_size);

		matmul(
			minibatch_output_data_pre,
			layer1_weight,
			minibatch_hidden_data,
			output_size,
			minibatch_size,
			hidden_size);
		add_bias(
			minibatch_output_data_pre,
			layer1_bias,
			output_size,
			minibatch_size);
		softmax(
			minibatch_output_data,
			minibatch_output_data_pre,
			output_size,
			minibatch_size
			);

		//
		// Backword
		//
		compute_last_error(minibatch_output_error, minibatch_output_data, minibatch_label_data, output_size, minibatch_size);
		dReLU(minibatch_hidden_data_pre, minibatch_hidden_data_pre, hidden_size, minibatch_size);
		matmul_tn(minibatch_hidden_error, layer1_weight, minibatch_output_error, hidden_size, minibatch_size, output_size);
		elementwise_product(minibatch_hidden_error, minibatch_hidden_error, minibatch_hidden_data_pre, minibatch_size * hidden_size);

		update_weight(layer1_weight, minibatch_output_error, minibatch_hidden_data,  output_size, hidden_size, minibatch_size, learning_rate);
		update_weight(layer0_weight, minibatch_hidden_error, minibatch_image_data,  hidden_size, input_size, minibatch_size, learning_rate);
		update_bias(layer1_bias, minibatch_output_error, output_size, minibatch_size, learning_rate);
		update_bias(layer0_bias, minibatch_hidden_error, hidden_size, minibatch_size, learning_rate);

		if (i % print_info_interval == (print_info_interval - 1)) {
			const auto train_acc = compute_accuracy(minibatch_output_data, minibatch_label_data, output_size, minibatch_size);
			const auto train_loss = compute_loss(minibatch_output_data, minibatch_label_data, output_size, minibatch_size);
			std::printf("[%6lu] acc = %.3f \%, loss = %e\n", i, train_acc * 100.0f, train_loss);
		}
	}

	free(minibatch_image_data);
	free(minibatch_label_data);
	free(minibatch_hidden_data);
	free(minibatch_output_data);
	free(layer0_weight);
	free(layer0_bias);
	free(layer1_weight);
	free(layer1_bias);
}
