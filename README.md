# MNIST学習器

## 実行方法
1. clone and change directory
```bash
git clone https://github.com/rioyokotalab/mnist_train.git
cd mnist_train
```
2. download MNIST dataset
```bash
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
```
3. extract MNIST dataset
```
gnuzip *.gz
```
4. build
```bash
make
```
5. run
```bash
./mnist_train_cpu
```
