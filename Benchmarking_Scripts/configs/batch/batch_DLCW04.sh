#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH --array=1-1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=def-rpicker
#SBATCH -J 1mbenchmarking
#SBATCH -e /scratch/hepworth/hepworth/ucnanalysis/1mbenchmark/errors/error_DLCW04.txt 
#SBATCH -o /scratch/hepworth/hepworth/ucnanalysis/1mbenchmark/outputs/output_DLCW04.txt

SEED=$(date +%s%N)

echo "Current working directory is `pwd`"
echo $SLURM_JOB_NAME
echo "Starting run at: `date`"
/project/6006407/hepworth/PENTrack/PENTrack $SLURM_ARRAY_TASK_ID "/project/6006407/hepworth/ucnanalysis/1mbenchmark/configs/configs/config_DLCW04.in" "/scratch/hepworth/hepworth/ucnanalysis/1mbenchmark/results/DLCW04" $SEED
echo "Program finished with exit code $? at: `date`"
