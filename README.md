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
gunzip *.gz
```
4. build
```bash
make
```
5. run
```bash
./mnist_train_cpu
```
or
```
qsub t3-job.sh
```
on TSUBAME3.0 or
```
sbatch ykt-job.sh
```
on Yokota cluster

## 学習器構造
### ネットワーク
2層全結合ニューラルネットワーク

### 学習
単純なSGD

## ファイル構成
- `main_cpu.cpp`  
CPUで動作する学習器

- `utils.hpp`  
CPU/GPU共通で使用する関数群

- `mnist.hpp`  
MNIST読み込み用関数
