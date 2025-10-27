#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH --array=#JOB_FIRST-#JOB_LAST
#SBATCH --mem-per-cpu=4G
#SBATCH --account=def-rpicker
#SBATCH -J jparcstorage
#SBATCH -e /scratch/hepworth/hepworth/ucnanalysis/JPARC_2024/storage/errors/error_#SUID.txt 
#SBATCH -o /scratch/hepworth/hepworth/ucnanalysis/JPARC_2024/storage/outputs/output_#SUID.txt

SEED=$(date +%s%N)

echo "Current working directory is `pwd`"
echo $SLURM_JOB_NAME
echo "Starting run at: `date`"
/project/6006407/hepworth/PENTrack/PENTrack $SLURM_ARRAY_TASK_ID "/project/6006407/hepworth/ucnanalysis/JPARC_2024/storage/configs/configs/config_#SUID.in" "/scratch/hepworth/hepworth/ucnanalysis/JPARC_2024/storage/results/#SUID" $SEED
echo "Program finished with exit code $? at: `date`"
