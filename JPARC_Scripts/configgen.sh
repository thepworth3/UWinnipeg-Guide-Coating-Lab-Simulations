# configgen.sh
# This is a bash script to generate and run different configurations for a simulation of a time-of-flight experiment at JPARC
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
while IFS="," read -r SUID JOB_FIRST JOB_LAST PARAMANGDIST PARAMSSW PARAMSSP PARAMAL PARAMNIPW PARAMNIPP PARAMCELLMAT PARAMCELLW PARAMCELLP
do
	# Conditional selects which of the cases to generate and run
	if [ $i -ge $index_from ] && [ $i -le $index_to ]
	then
		# Create a copy of the configuration template named with the unique case ID
		cp config_template.in "configs/config_${SUID}.in"
		echo $SUID
		
		# For each parameter, substitute the placeholder in the configuration file with the value in the spreadsheet
		perl -pi -e "s/#PARAMSSW/${PARAMSSW}/g" "configs/config_${SUID}.in"
		perl -pi -e "s/#PARAMSSP/${PARAMSSP}/g" "configs/config_${SUID}.in"
		perl -pi -e "s/#PARAMAL/${PARAMAL}/g" "configs/config_${SUID}.in"
		
		perl -pi -e "s/#PARAMNIPW/${PARAMNIPW}/g" "configs/config_${SUID}.in"
		perl -pi -e "s/#PARAMNIPP/${PARAMNIPP}/g" "configs/config_${SUID}.in"
		perl -pi -e "s/#PARAMDLCW/${PARAMDLCW}/g" "configs/config_${SUID}.in"

		# Parameters can also be used to select cases. Here, the angular distribution is selected. In the configuration file, the sections for the distribution cases are
		# commented out by the parameters. These lines delete the comments to activate the selected case. 
		# This can be expanded to cover anything not defined by a single parameter - even components of a geometry can be selected in this way.
		if [ $PARAMANGDIST = "COL" ]
		then
			perl -pi -e "s/#ANGDISTCOL//g" "configs/config_${SUID}.in"
		fi
		if [ $PARAMANGDIST = "UNI" ]
		then
			perl -pi -e "s/#ANGDISTUNI//g" "configs/config_${SUID}.in"
		fi
		
		# Create a directory for the results
		mkdir ~/scratch/hepworth/hepworth/ucnanalysis/JPARC_2024/filling/results/$SUID
		
		# Create a copy of the batch script template for submission to a computing cluster
		# Some parameters in the spreadsheet correspond to the batch template - eg. number of cores requested is defined by JOB_FIRST/JOB_LAST
		# The batch script handles error/output redirection to files named with the UID
		rm "/project/6006407/hepworth/ucnanalysis/JPARC_2024/filling/configs/batch/batch_${SUID}.sh"
		cp batch_template.sh "/project/6006407/hepworth/ucnanalysis/JPARC_2024/filling/configs/batch/batch_${SUID}.sh"
		perl -pi -e "s/#SUID/${SUID}/g" "/project/6006407/hepworth/ucnanalysis/JPARC_2024/filling/configs/batch/batch_${SUID}.sh"
		perl -pi -e "s/#JOB_FIRST/${JOB_FIRST}/g" "/project/6006407/hepworth/ucnanalysis/JPARC_2024/filling/configs/batch/batch_${SUID}.sh"
		perl -pi -e "s/#JOB_LAST/${JOB_LAST}/g" "/project/6006407/hepworth/ucnanalysis/JPARC_2024/filling/configs/batch/batch_${SUID}.sh"
		cd "/project/6006407/hepworth/ucnanalysis/JPARC_2024/filling/configs/batch"
		# Submit job to the cluster
		sbatch "batch_${SUID}.sh"
		cd $cwd
	fi
	i=$((i+1))
done < <(tail -n +2 input.csv) # End of the loop, "tail -n +2" skips the header line



