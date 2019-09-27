#!/bin/sh
#$ -cwd
#$ -l q_node=1
#$ -l h_rt=0:10:00
#$ -N ykt-mnist-training

. /etc/profile.d/modules.sh
module load cuda

./mnist_train_cpu
