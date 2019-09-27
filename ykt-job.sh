#!/bin/bash
#SBATCH -p turing
#SBATCH --nodes 2
#SBATCH -J job_name

. /etc/profile.d/modules.sh
module load cuda

./mnist_train_cpu
