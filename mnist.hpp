#ifndef __MNIST_HPP__
#define __MNIST_HPP__
#include <fstream>
#include <stdexcept>

class MNISTLoader{
	char *data;
	char *labels;
	inline int reverse(int n);
	std::size_t num_data;
public:
	static const std::size_t IMAGE_DIM = 28;
	static const std::size_t CLASS_SIZE = 10;
	inline void load(std::string data_file_name,std::string labels_file_name);
	inline void copy(std::size_t id,float *x,float *t);
	MNISTLoader() : data(nullptr), labels(nullptr) {};
	~MNISTLoader(){delete [] data; delete [] labels;};
};

inline int MNISTLoader::reverse(int n) {
	char a0,a1,a2,a3;
	a0 = (n>>24) & 255;
	a1 = (n>>16) & 255;
	a2 = (n>>8) & 255;
	a3 = n & 255;
	return ((int)a3 << 24) + ((int)a2 << 16) + ((int)a1 << 8) + a0;
}

inline void MNISTLoader::load(std::string data_file_name, std::string labels_file_name){
	std::ifstream image_ifs(data_file_name,std::ios::binary);
	std::ifstream label_ifs(labels_file_name,std::ios::binary);

	if (!image_ifs) {
		throw std::runtime_error("No such a file : " + data_file_name);
	}
	if (!label_ifs) {
		throw std::runtime_error("No such a file : " + labels_file_name);
	}

	uint32_t buffer;
	uint32_t magic_number, row, col;
	int label;
	int read_1byte_int;
	image_ifs.read((char*)&magic_number, sizeof(magic_number));
	magic_number = reverse(magic_number);
	image_ifs.read((char*)&buffer, sizeof(buffer));
	num_data = reverse(buffer) & 0xffff;
	image_ifs.read((char*)&row, sizeof(row));
	row = reverse(row);
	image_ifs.read((char*)&col, sizeof(col));
	col = reverse(col);
	label_ifs.read((char*)&magic_number, sizeof(magic_number));
	magic_number = reverse(magic_number);
	label_ifs.read((char*)&buffer, sizeof(buffer));
	std::size_t num_label_data = reverse(buffer) & 0xffff;

	if (num_label_data != num_data) {
		throw std::runtime_error("The number of image data and label data are mismatch");
	}
	//std::cout<<"magic number = "<<magic_number<<std::endl;
	//std::cout<<"num_data = "<<num_data<<std::endl;
	//std::cout<<"row = "<<row<<std::endl;
	//std::cout<<"col = "<<col<<std::endl;

	data = new char [num_data * IMAGE_DIM * IMAGE_DIM];
	labels = new char [num_data];
	for (std::size_t a = 0; a < num_data; a++) {
		label_ifs.read((char*)&label, sizeof(char));
		label &= 0xf;
		labels[a] = ( label );
		for (std::size_t i = 0; i < IMAGE_DIM * IMAGE_DIM; i++){
			image_ifs.read((char*)&read_1byte_int, sizeof(char));
			read_1byte_int &= 0xf;
			data[a * IMAGE_DIM * IMAGE_DIM + i] = (char)read_1byte_int;
		}
	}
	image_ifs.close();
	label_ifs.close();
}

inline void MNISTLoader::copy(std::size_t id, float *x, float *t){
	if (id >= num_data) {
		std::runtime_error("No such a file : id = " + std::to_string(id));
	}

	for (std::size_t j = 0; j < IMAGE_DIM * IMAGE_DIM; j++) {
		x[IMAGE_DIM * IMAGE_DIM + j] = static_cast<float>(data[id * IMAGE_DIM * IMAGE_DIM + j]) / 255.0f;
	}
	for (std::size_t j = 0; j < 10; j++) {
		t[10 + j] = (static_cast<int>(labels[id]) == j) ? 1.0f : 0.0f;
	}
}
#endif /* end of include guard */
