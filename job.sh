#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --job-name=pyjob
#SBATCH --output=pyjob.%j.out

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=chpe7876@colorado.edu 

source ~/.bashrc

cd $SLURM_SUBMIT_DIR

acompile

module purge
module load miniforge
mamba activate pycfd

python3 NewUpdated.py NACAPoints.dat
