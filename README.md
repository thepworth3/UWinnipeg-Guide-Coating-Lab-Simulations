# UWinnipeg-Guide-Coating-Lab-Simulations
This is a summary of the contents of this github repo, and how you could clone the repository to run my simulations yourself. This includes all STL files, PENTrack configuration and materials files, automated scripts for creating a large number of simulations, and analysis scripts written in python used to analyze the data for my undergraduate honours thesis at the University of Winnipeg under Dr. Russell Mammei. Questions can be directed to me via [my email](mailto:thomashepworth12@gmail.com). 










## Summary of Contents 

### PENTrack_Simulation_Procedures_2025
This is a document written and edited by many past and current members of the TUCAN collaboration including R. Picker, S. Sidhu, S. Lan, and K. Drury. It describes how members of TUCAN can use computecanada for PENTrack simulations. TUCAN students will need a computecanada account if they are doing a PENTrack project. To have a new account set up, contact R. Picker at TRIUMF. Note this document is not nessecarily reflecting of the most updated version of PENTrack, which can be found on the [PENTrack Github](https://github.com/wschreyer/PENTrack/tree/master). When in doubt, rely on the information of the most updated version of the PENTrack Github.

### JPARC_Simulation_Model
This folder contains the STL files used for 1m long DLC guide storage experiment simulations at the Japan Proton Accelerator Research Complex [(J-PARC)](https://j-parc.jp/c/en/). It also contains a picture of the model for reference. If you are interested in modify this model for a future TUCAN experiment at J-PARC, I can share the Solidworks Assembly. 

### Benchmark_STL
This folder contains the STL files for a benchmarking model used for analytical testing of PENTrack, as described in my honours thesis. It also contains a picture of the model for reference. This is a great way to learn PENTrack and understand what it is simulating.


### Benchmarking_Scripts
This folder contains all of the scripts I used in my workflow to run the simple benchmarking simulations. If you are goign to run these yourself, you must change the directory stucture in these files to match what you are using on your computecanada account, recalling that data and ouput files should be stored in your scratch folder, and you cannot submit jobs from your home directory. You can use your login node in some cases, put is is best practice to submit things from your project directory. The baseline stucture for this would be /project/6006407/your_username


## Important Scripts Present for all of my simulation workflow
I use a series of scripts to generate my simulations based on desired input paramters for the simulation. While being tailored to each category of simulation I run (benchmarking, J-PARC, TRIUMF, etc.) each scripts has the same defined role in each workflow.


### config_template.in
This file defines the baseline of the kind of simulation you want to run. Lines or segments of code that you want to swap out for different options are defined as commands that can be identified by the other scripts written in Perl. For example, if you wanted to run a simulation with different imaginary optical potentials for DLC, you may specify the material not as DLC, but as 

"""
DLC_#PARAMDLCW
"""
The other scripts will see this, and swap out PARAMDLCW for each entry in the PARAMDLCW column in input.csv, where each row is a seperate simulation.


See the [PENTrack Github](https://github.com/wschreyer/PENTrack/tree/master) for more info about configuration files and how to best use them. 


### batch_template.in 


### configgen.sh


### input.csv
