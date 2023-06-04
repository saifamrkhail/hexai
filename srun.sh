#!/bin/bash
#SBATCH --job-name=hexai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:05:00

# Load any required modules
module load python

# Activate your virtual environment if needed
conda activate base

# Change to your working directory
cd /home/ai22m055/hexai/fhtw_hex

# Run your job
srun python TrainAlphaZero.py
```