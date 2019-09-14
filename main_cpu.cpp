#include <iostream>
#include <mnist.hpp>

constexpr std::size_t batch_size = 16;

constexpr std::size_t input_size = MNISTLoader::IMAGE_DIM * MNISTLoader::IMAGE_DIM;
constexpr std::size_t hidden_size = 60;
constexpr std::size_t output_size = MNISTLoader::CLASS_SIZE;
constexpr float learning_rate = 0.01f;

int main() {
	float* const minibatch_image_data = (float*)malloc(batch_size * MNISTLoader::IMAGE_DIM * MNISTLoader::IMAGE_DIM);
	float* const minibatch_label_data = (float*)malloc(batch_size * MNISTLoader::CLASS_SIZE);


	free(minibatch_image_data);
	free(minibatch_label_data);
}
