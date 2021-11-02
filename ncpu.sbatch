#!/bin/bash

#SBATCH -J exp-lin
#SBATCH -c 8
#SBATCH --mem=16g
#SBATCH --time=2:00:00


# run
cd Memo/regulateme
source activate /global/home/hpc4590/anaconda3/envs/memo
python test_ipopt_mini.py
# sleep 24h