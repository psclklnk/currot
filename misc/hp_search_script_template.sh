#!/bin/bash

#SBATCH -A project{project:05d}
#SBATCH -a 1-{n_seeds:d}
#SBATCH -n 1
#SBATCH -c {n_cores:d}
#SBATCH -t 24:00:00
#SBATCH --mem-per-cpu=2000

#SBATCH -o /work/scratch/in86doko/hp_search_logs/{type}_hp_search/{env}/%A_%a.out
#SBATCH -e /work/scratch/in86doko/hp_search_logs/{type}_hp_search/{env}/%A_%a.err

eval "$(/home/in86doko/miniconda3/bin/conda shell.bash hook)"
conda activate currot
module load gurobi intel openblas lapack

mkdir -p /home/in86doko/logs/{type}_hp_search/{env}
cd {project_dir}
python run.py --env {env} --learner {learner} --base_log_dir {base_log_dir} --seed $SLURM_ARRAY_TASK_ID --type {type} --n_cores {n_cores:d} {extra_args}
