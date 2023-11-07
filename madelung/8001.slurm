#! /bin/bash
#SBATCH --job-name="8001"
#SBATCH --time=4:00:00
#SBATCH --partition="secondary"
#SBATCH --cpus-per-task=20
#SBATCH --mem=512G
#SBATCH --output=./8001.out
echo $SLURM_SUBMIT_DIR
cd $SLURM_SUBMIT_DIR
pwd
export OMP_NUM_THREADS=1
python run.py -n 8001 8001 1
