import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2

#--- NEUTRON MOVIE MAKER using PENTrack ----#
#------------ Author: K. Drury -------------#
#-- Developed for the TUCAN Collaboration --#
#-------- Last updated: 2024-11-14 ---------#

########################################################################################################
## USER DEFINED PARAMETERS ##




## File orgnazation ##      # Specify the task id
data_folder = r'C:\Users\thoma\OneDrive\Documents\University of Winnipeg\2024-2025 Courses\PHYS-4001 Thesis\JPARC_2024\Test_sim\storage\results4000UCN\JPARCMOV00001'        # Specify the relative path to the folder containing all the .out files
image_folder = r'C:\Users\thoma\OneDrive\Documents\University of Winnipeg\2024-2025 Courses\PHYS-4001 Thesis\JPARC_2024\Test_sim\storage\results4000UCN\Frames'     # Specify the relative path to the folder where the frames/movies will be saved

## PENTrack parameters ##
sim_time = 140                # Specify the simulation time in seconds
fps = 28/140            #should it be 24/120?          # Specify the PENTrack snapshot rate

## Movie parameters ##
x_axis = 'x'                  # Specify the x-axis label for the movie
y_axis = 'z'                  # Specify the y-axis label for the movie
x_axis_bounds = [-0.975, 0.9]        # Specify the x-axis bounds
y_axis_bounds = [-1, 0]       # Specify the y-axis bounds
frame_width = 3000            # Specify the frame width
frame_height = 500            # Specify the frame height
movie_framerate = 10          # Specify the movie framerate

########################################################################################################
## NEUTRON CLASS DEFINITION ##

class Neutron:
    def __init__(self, sim_time, fps):  # Constructor for Neutron class
        self.sim_time = sim_time
        Nframes = int(sim_time*fps) + 1
        self.t = np.arange(0, sim_time + 1/fps, 1/fps)
        self.x = np.zeros(Nframes)                      # x,y,z arrays initialized with zeros
        self.y = np.zeros(Nframes)
        self.z = np.zeros(Nframes)

    def startup(self, start_time):     # Method for filling in x,y,z arrays with NaNs before start_time
        self.start_time = start_time

        for i in range(len(self.t)):
            if self.t[i] < start_time*fps:
                self.x[i] = np.nan
                self.y[i] = np.nan
                self.z[i] = np.nan

    def push(self, value, coordinate):          # Push value to first NaN in specified array
        if coordinate not in ["x", "y", "z"]:
            raise ValueError("coordinate must be 'x', 'y', or 'z'")

        array = getattr(self, coordinate)  # Get the specified array dynamically
        self.value = value

        for i in range(len(array)):  # Find the first 0.0 and set it to the given value
            if array[i] == 0.0:
                array[i] = value
                break  # Stop after setting the first 0.0

    def finish(self, coordinate):  # method for filling all the remaining zeroes in a coordinate array with NaNs
        if coordinate not in ["x", "y", "z"]:
            raise ValueError("coordinate must be 'x', 'y', or 'z'")

        array = getattr(self, coordinate)  # Get the specified array dynamically

        for i in range(len(array)):  # Find the first 0.0 and set it to the given value
            if array[i] == 0.0:
                array[i] = np.nan

########################################################################################################
## FUNCTION DEFINITIONS ##

def generate_frame(time, frame_width, frame_height, frame_number):  # Generate a frame at a given time
    x, y = [], []
    axis_map = {'x': 'x', 'y': 'y', 'z': 'z'}  # Map the axis to the correct column name

    x_attr = axis_map[x_axis]
    y_attr = axis_map[y_axis]

    for i in range(num_neutrons):
        for j in range(len(neutrons[i].t)):
            if neutrons[i].t[j] == time:
                x.append(getattr(neutrons[i], x_attr)[j])
                y.append(getattr(neutrons[i], y_attr)[j])
                break

    fig, ax = plt.subplots(figsize=(frame_width/100, frame_height/100), dpi=300)
    ax.set_xlim(x_axis_bounds)
    ax.set_ylim(y_axis_bounds)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.scatter(x, y, s=2, c='blue')
    ax.set_title(f't = {time} s')
    ax.grid(True)
    plt.savefig(f"{image_folder}\\frame_{frame_number:05d}.png", dpi=100)  # Save with 100 DPI for correct resolution
    plt.close()

########################################################################################################
movie_length = (sim_time*fps + 1) / movie_framerate  # Calculate the movie length in seconds
playback_speed = movie_length / sim_time             # Calculate the playback speed
print('Movie length:', movie_length, 'seconds')      # Print the movie length
print('Playback speed:', playback_speed, 'x')        # Print the playback speed

# print("Extracting data from PENTrack output files for job number", job_number, "task id", task_id, "...")
end_file = f'{data_folder}\\000000000001neutronend.out'             # Specify the end file
snapshot_file = f'{data_folder}\\000000000001neutronsnapshot.out'   # Specify the snapshot file

#verified up to here

end_df = pd.read_csv(end_file, sep=' ')             # Read the end file
snapshot_df = pd.read_csv(snapshot_file, sep=' ')   # Read the snapshot file

print("Creating Neutron objects...")
num_neutrons = len(end_df)                                   # Get the number of neutrons
neutrons = [Neutron(sim_time, fps) for _ in range(num_neutrons)]  # Create a list of Neutron objects

print("Getting start times...")
start_times = end_df['tstart']                                 # Get the start times
refresh_rate = 1/fps                                           # Get the refresh rate
start_times = np.floor(start_times/refresh_rate)*refresh_rate  # Round down the start times to nearest multiple of 'fps'

timestamps = snapshot_df['tend']             # Get the timestamps
neutron_ids = snapshot_df['particle']        # Get the neutron ids
x_axis_points = snapshot_df[f'{x_axis}end']  # Get the x-axis points
y_axis_points = snapshot_df[f'{y_axis}end']  # Get the y-axis points

for i in range(num_neutrons):
    neutrons[i].startup(start_times[i])  # Fill in x,y,z arrays with NaNs before start time

print("Pushing data to Neutron objects...")
for i in range(len(snapshot_df)):                       # Loop over all rows in the snapshot file
    for j in range(num_neutrons):                       # Loop over all the neutrons
        if neutron_ids[i] == j+1:                       # If the neutron id matches the neutron index
            neutrons[j].push(x_axis_points[i], x_axis)  # Push x-axis points to the corresponding neutron
            neutrons[j].push(y_axis_points[i], y_axis)  # Push y-axis points to the corresponding neutron
            print(j+1)
            break
print("checkpoint")
for i in range(num_neutrons):   # Loop over all the neutrons
    neutrons[i].finish(x_axis)  # Fill in remaining zeroes in x-axis array with NaNs
    neutrons[i].finish(y_axis)  # Fill in remaining zeroes in y-axis array with NaNs

timesteps = np.arange(0, sim_time, 1/fps)  # Create an array of timesteps

print("Generating frames...")
for i in range(len(timesteps)):
    generate_frame(timesteps[i], frame_width, frame_height, i)  # Generate a frame at the given timestep

print("Creating movie...")
movie_name = f'{image_folder}/000000000001.mp4'                          # Specify the movie name
images = [f for f in os.listdir(image_folder) if f.endswith(".png")]                     # Get all the images in the folder
fourcc = cv2.VideoWriter_fourcc(*'mp4v')                                                 # Codec for .mp4 file
out = cv2.VideoWriter(movie_name, fourcc, movie_framerate, (frame_width, frame_height))  # Create the video

for image_file in images:
    img = cv2.imread(os.path.join(image_folder, image_file))
    out.write(img)  # Write each image to the video

out.release()  # Release the video
print("Movie created successfully, cleaning up...")

# for image_file in images:
#     os.remove(os.path.join(image_folder, image_file))  # Remove the images

# print('--------------------------------------------')
# print("Cleanup complete, movie saved as", movie_name)  # Print cleanup complete message