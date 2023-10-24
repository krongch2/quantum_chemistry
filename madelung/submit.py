import os

def get_hours(queue):
    hours_map = {'secondary': 4, 'qmchamm': 24, 'physics': 14, 'test': 1}
    return hours_map[queue]

def submit_slurm(cmd, dirname='.', queue='secondary', prefix='job'):
    os.makedirs(dirname, exist_ok=True)
    submit_fn = f'{dirname}/{prefix}.slurm'
    with open(submit_fn, 'w') as f:
        f.write(
f'''#! /bin/bash
#SBATCH --job-name="{prefix}"
#SBATCH --time={get_hours(queue)}:00:00
#SBATCH --partition="{queue}"
#SBATCH --cpus-per-task=20
#SBATCH --mem=512G
#SBATCH --output={dirname}/{prefix}.out
echo $SLURM_SUBMIT_DIR
cd $SLURM_SUBMIT_DIR
pwd
export OMP_NUM_THREADS=1
{cmd}
''')
    os.system(f'sbatch {submit_fn}')

if __name__ == '__main__':
    for n in [
        30000, 50000, 100000
        ]:
        submit_slurm(f'python run.py -n {n} {n} 1', prefix=f'{n}')
