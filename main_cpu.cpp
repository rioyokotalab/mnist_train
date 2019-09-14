#include <iostream>
#include <random>
#include <cmath>
#include <mnist.hpp>
#include <utils.hpp>

constexpr std::size_t minibatch_size = 16;
constexpr std::size_t num_iterations = 4;

constexpr std::size_t input_size = mnist_loader::IMAGE_DIM * mnist_loader::IMAGE_DIM;
constexpr std::size_t hidden_size = 60;
constexpr std::size_t output_size = mnist_loader::CLASS_SIZE;

constexpr float learning_rate = 0.01f;

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

void add_bias(float* const A, const float* const bias, const std::size_t layer_size, const std::size_t minibatch_size) {
	for (std::size_t mb = 0; mb < minibatch_size; mb++) {
		for (std::size_t ls = 0; ls < layer_size; ls++) {
			A[mb * layer_size + ls] += bias[ls];
		}
	}
}

void ReLU(float* const acted, const float* const pre_act, const std::size_t size) {
	for (std::size_t i = 0; i < size; i++) {
		acted[i] = std::max(0.0f, pre_act[i]);
	}
}

void softmax(float* const acted, const float* const pre_act, const std::size_t layer_size, const std::size_t minibatch_size) {
	for (std::size_t mb = 0; mb < minibatch_size; mb++) {
		float e_sum = 0;
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

int main() {
	mnist_loader train_data, test_data;
	train_data.load("train-images-idx3-ubyte", "train-labels-idx1-ubyte");
	test_data.load("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");
	float* const minibatch_image_data = (float*)malloc(minibatch_size * mnist_loader::IMAGE_DIM * mnist_loader::IMAGE_DIM * sizeof(float));
	float* const minibatch_label_data = (float*)malloc(minibatch_size * mnist_loader::CLASS_SIZE * sizeof(float));

	float* const minibatch_hidden_data_pre = (float*)malloc(minibatch_size * hidden_size * sizeof(float));
	float* const minibatch_hidden_data = (float*)malloc(minibatch_size * hidden_size * sizeof(float));
	float* const minibatch_output_data_pre = (float*)malloc(minibatch_size * output_size * sizeof(float));
	float* const minibatch_output_data = (float*)malloc(minibatch_size * output_size * sizeof(float));

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
			hidden_size * minibatch_size);

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

		utils::print_matrix(minibatch_output_data, output_size, minibatch_size, output_size, "output");
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
