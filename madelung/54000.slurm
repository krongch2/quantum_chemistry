#! /bin/bash
#SBATCH --job-name="54000"
#SBATCH --time=4:00:00
#SBATCH --partition="secondary"
#SBATCH --cpus-per-task=20
#SBATCH --mem=512G
#SBATCH --output=./54000.out
echo $SLURM_SUBMIT_DIR
cd $SLURM_SUBMIT_DIR
pwd
export OMP_NUM_THREADS=1
python run.py -n 54000 54000 1
