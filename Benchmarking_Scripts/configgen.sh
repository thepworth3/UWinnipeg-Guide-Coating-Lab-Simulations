# configgen.sh
# This is a bash script to generate and run different configurations for a simulation of a benchmarking guide to be tested at JPARC by T.Hepworth, R. Mammei et.al in Feb 2025
# Run from command line using "bash configgen.sh index_from index_to"
# index_from: number of the first line of the parameter csv file that you want to run
# index_to: number of the last line of the parameter csv file that you want to run
# Note on indices: these are numbered with the first line of the file (the header line) being line 1, so the first configuration is line 2
# This numbering is adjusted in the script so that i=1 corresponds to the first configuration line 

#!/bin/bash
# Current working directory
cwd=$(pwd)

# Gets the two input indices from the command line
index_from=$(($1-1))
index_to=$(($2-1))

# Loop iterates over the lines of the spreadsheet
# The first header line is skipped
# For each line, the values are read into the variables listed here
i=1
while IFS="," read -r SUID JOB_FIRST JOB_LAST PARAMDLCW
do
	# Conditional selects which of the cases to generate and run
	if [ $i -ge $index_from ] && [ $i -le $index_to ]
	then
		# Create a copy of the configuration template named with the unique case ID
		cp config_template.in "/project/6006407/hepworth/ucnanalysis/1mbenchmark/configs/configs/config_${SUID}.in"
		echo $SUID
		
		# For each parameter, substitute the placeholder in the configuration file with the value in the spreadsheet
		perl -pi -e "s/#PARAMDLCW/${PARAMDLCW}/g" "configs/config_${SUID}.in"
		

		
		# Create a directory for the results
		mkdir  /scratch/hepworth/hepworth/ucnanalysis/1mbenchmark/results/$SUID
		
		# Create a copy of the batch script template for submission to a computing cluster
		# Some parameters in the spreadsheet correspond to the batch template - eg. number of cores requested is defined by JOB_FIRST/JOB_LAST
		# The batch script handles error/output redirection to files named with the UID
		rm "/project/6006407/hepworth/ucnanalysis/1mbenchmark/configs/configs/batch/batch_${SUID}.sh"
		cp batch_template.sh "/project/6006407/hepworth/ucnanalysis/1mbenchmark/configs/configs/batch/batch_${SUID}.sh"
		perl -pi -e "s/#SUID/${SUID}/g" "/project/6006407/hepworth/ucnanalysis/1mbenchmark/configs/configs/batch/batch_${SUID}.sh"
		perl -pi -e "s/#JOB_FIRST/${JOB_FIRST}/g" "/project/6006407/hepworth/ucnanalysis/1mbenchmark/configs/configs/batch/batch_${SUID}.sh"
		perl -pi -e "s/#JOB_LAST/${JOB_LAST}/g" "/project/6006407/hepworth/ucnanalysis/1mbenchmark/configs/configs/batch/batch_${SUID}.sh"
		cd "/project/6006407/hepworth/ucnanalysis/1mbenchmark/configs/configs/batch"
		# Submit job to the cluster
		sbatch "batch_${SUID}.sh"
		cd $cwd
	fi
	i=$((i+1))
done < <(tail -n +2 input.csv) # End of the loop, "tail -n +2" skips the header line



