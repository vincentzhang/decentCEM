#!/bin/bash
#SBATCH --account=def-schuurma
#SBATCH --gres=gpu:v100l:1      # request GPU v100
#SBATCH --cpus-per-task=1       # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=16000M            # memory per node
#SBATCH --time=3-18:00          # time (DD-HH:MM)
#SBATCH --mail-user=vincent.zhang@ualberta.ca
#SBATCH --mail-type=ALL,TIME_LIMIT
#SBATCH --exclude=cdr2550       # problematic nodes to be excluded

echo "Current working directory: `pwd`"
echo "Starting run at: `date`"

module load nixpkgs/16.09  gcc/7.3.0  cuda/10.0.130 cudnn/7.4
source /home/vincent/projects/def-jag/vincent/tf1p36/bin/activate

python param_search_cc.py configs/params.json $SLURM_ARRAY_TASK_ID

# ---------------------------------------------------------------------
echo "Job finished with exit code $? at: `date`"
# ---------------------------------------------------------------------
