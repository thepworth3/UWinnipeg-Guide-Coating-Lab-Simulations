# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 18:17:13 2024

@author: thoma

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import re
import os
import argparse
import KT_tools as tools 
# import get_data

#make script I can call that imports every possible df from the output file so I can just call each time I do analysis

#higher W -> higher mu, so more losses should occur and tau decreases.

#make path an input in command line, and make the plots save

print(os.getcwd())


#analysis function
def analyze_data(path):
   df = pd.read_csv(path, index_col =0, sep=" ")
   df = df[df['stopID'] != -1] #cut out neutrons that didn't finish because I made simtime cutoff 200s for test
   
   
   
   
   
   xend = df.xend
   yend = df.yend  #import ending positions to verify geometry is as expected
   zend = df.zend
  
   #Parse out the W value form my workflow  
  
   # Pattern: "DLCW" + one or more digits (\d+)
   pattern = r"DLCW(\d+)"
   # pattern1 = r"(\d+)\results"
   match = re.search(pattern, path)
   # match1 = re.search(pattern1, path)
   if match:
       # match.group(1) returns only the digits part (e.g., "10")
       numeric_part = float(match.group(1))
       # direc = str(match1.group(1))
       # print(direc)
       # print(numeric_part)  # prints "10"
       W = numeric_part/100
       print("-----")
       print("-----")
       print("-----")
       print("Imaginary Fermi Potential Simulated for the guide surface was", W,"neV")
       Wstr = str(W)
   pattern1 = re.compile(r'(\d+)s/results')
   match1 = pattern1.search(path)
   if match:
        direc = match1.group(0)
   

    
   tstart = df.tstart    #start and end times to get the neutron lifetime
   tend = df.tend
   Estart = df.Estart #import starting kinetic energy
   # print(Estart)

    
   lifetime = tend - tstart
   lifetime = lifetime[~np.isnan(lifetime)]  #calculate lifteime and clean data
   lifetime = lifetime[np.isfinite(lifetime)]
  
   
   #want first bin centered at 0 for proper fit...
   #prepare histogram    
   bins = 25
   bin_edges = np.linspace(0, lifetime.max(), bins +1 )  # Adjusted to start from 0
   counts, _ = np.histogram(lifetime, bins=bin_edges)
   bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 
   counts[counts == 0] += 1 #prevent empty bin curve fit issues
   
   # print("counts", counts)
   
   #energy binning to get weighted average of the energy, if needed
   bin_energy = 100
   counts_energy, bin_edges_energy = np.histogram(Estart, bins = bin_energy)
   bin_centers_energy = (bin_edges_energy[:-1] + bin_edges_energy[1:])/2    #gets the bin center by averaging the bin edges

   weighted_average_energy = np.sum(bin_centers_energy *counts_energy) / np.sum(counts_energy)*10**9

   print("Weighted Avg Energy of UCN population:", weighted_average_energy, "neV")
   print("loss per bounce:", W/269)
   



   # print(sigma_counts)

   def lossRateApprox(E, A, V, mu):     #functions for kinetic theory analysis, see chapter of 4 of Golub et.al
       return np.sqrt(2*E/m)*A/(4*V)*mu + 1/880
   
   def lossPerBounceApprox(E, V, W):
       return 2.0*W/V*(V/E * np.arcsin(np.sqrt(E/V)) - np.sqrt(V/E - 1))
   
   initial_guess = [5000, 8]  # [A, tau]
   sigma_counts = np.sqrt(counts)   #set error bar of counting data
   
   
   # Perform the curve fit
   popt, pcov = curve_fit(
       tools.negativeexp,         # the model function
       bin_centers,         # x-values (bin centers)
       counts,              # y-values (histogram counts)
       p0=initial_guess,     # initial parameter guesses
       sigma=sigma_counts,
       absolute_sigma = True,
       # bounds=([1e-5, 1e-5], [np.inf, np.inf]),
       maxfev=50000,
       # method='trf'
   )

   perr = np.sqrt(np.diag(pcov))
   sigma_A = np.sqrt(perr[0])
   sigma_tau = np.sqrt(perr[1])
   sigma_A_str = str(np.round(sigma_A,2))
   sigma_tau_str = str(np.round(sigma_tau,2))
   # popt are the best-fit parameters, pcov is the covariance matrix
   # need to add errors to the parameters
   A_fit, tau_fit = popt
   A_fit_str = str(np.round(A_fit,2))
   tau_fit_str = str(np.round(tau_fit, 2))
   print("Fitted A =", np.round(A_fit,2), '+-',np.round(sigma_A,2))
   print("Fitted tau =", np.round(tau_fit, 2), '+-',np.round(sigma_tau,2))

 
    
   
   
   

   m = 10.454 # neutron mass (neV s^2/m^2)
   mg = 102.5 # neutron mass * gravity (neV/m)
   
   #geometry of guide in setup
   r = 95/2/1000 #radius in m
   h = 1.0127
   A = 2*np.pi*r**2 + 2*np.pi*r*h
   V = np.pi*r**2*h
   
   #analytical calculation (approximate, ignoring energy disribution and height of vessel effects)
   mu_approx = tools.lossPerBounceApprox(weighted_average_energy, 269, W)
   tauinv_approx = tools.lossRateApprox(weighted_average_energy, A, V, mu_approx)
   tau_approx = 1/tauinv_approx
   tauavg_approx = np.average(tau_approx)
   print("tau avg", tau_approx, "s")
   print("percent lifetime is", (tau_fit - tauavg_approx)/tauavg_approx*100)
   
   
   #now the detailed tau calculation using the spectrum, phase space etc (Again, see Golub chapter 4)
   #need to look into tools as I am having problems with this calcualte by around a factor of 2
   
   
   z = np.linspace(min(zend), max(zend), len(Estart)) #make array of z positions to calculate geometry for each UCN simulated
   surface, crossSection = tools.guideDLCSurface(z,0)
   mu_calc = tools.lossPerBounceFermi(weighted_average_energy, 269, W)
   # plt.plot(crossSection)
   plt.show()
   
   
   tauinv_calc = tools.lossRate(weighted_average_energy, 269., W, mu_calc, z, crossSection, surface)    #can I not just input the average energy, what are the z positons I should input 
   tau_calc = 1/tauinv_calc
   print("full detail calculated tau", tau_calc, "s")

   #Some strings for plotting text
   tau_approx_str = str(np.round(tau_approx, 2))
   tau_calc_str = str(np.round(tau_calc, 3))

   # Plot histogram and fitted curve compared to the analytical curve
   plt.figure(figsize=(8,6),dpi=100)
   plt.hist(lifetime, bins=bin_edges, color='b', label='Data Histogram')
   plt.errorbar(bin_centers, tools.negativeexp(bin_centers, A_fit, tau_fit),yerr=sigma_counts, fmt = 'r.', label='Negative exp fit pts')
   plt.plot(bin_centers, tools.negativeexp(bin_centers, A_fit, tau_fit), 'g-', label='Negative exp fit')
   # plt.plot(0, A_fit, "r.")
   plt.plot(bin_centers, tools.negativeexp(bin_centers, A_fit, tau_approx), "r-", label = "approx analytical calculation")
   plt.plot(bin_centers, tools.negativeexp(bin_centers, A_fit, tau_calc), "y-", label = "full analytical calculation")
   plt.xlabel('Time (s)')
   plt.ylabel('Number of UCN')
   text = str("Guide has W = " + Wstr + " neV")
   Atext = str("Fitted amplitude is A = " + A_fit_str + "+/-" + sigma_A_str)
   tautext =str("Fitted lifteime is tau = " + tau_fit_str + "+/-" + sigma_tau_str + "s")
   tautextapprox = str("Approx calculated tau is " + tau_approx_str + "s")
   tautextcalc = str("Full calculated tau is " + tau_calc_str + "s")
   plt.text(250, counts[0]/1.4,  text)
   plt.text(250, counts[0]/1.5,  Atext)
   plt.text(250, counts[0]/1.6,  tautext)
   plt.text(250, counts[0]/1.8, tautextapprox)
   plt.text(250, counts[0]/2.0, tautextcalc)
   # plt.text(110, counts[0]/3.0, tauanalyticaltext)
   plt.legend()
   
   fname = str(direc + "/" + Wstr + "fit_counts_vs_time.png")
   plt.savefig(fname, format = "png")
   
   plt.show()
   
   print("analysis complete, consult plots")
   
   
   


def main():
    # Loop from 1 to 10 and format with 2 digits (01, 02, ..., 10)
    parser = argparse.ArgumentParser(description="Run analysis on given data path.")
    parser.add_argument(
        "path",
        help="Path to the data folder."
    )
    args = parser.parse_args()

    # args.path is the value passed by the user
    if not os.path.exists(args.path):
        print(f"Error: The path {args.path} does not exist.")
        exit(1)

    
    for i in range(1, 11):
        directory_name = f"DLCW{i:02d}"  # e.g., DLCW01, DLCW02, ...
        path = str(args.path + "/" + directory_name + "/000000000001neutronend.out")
        # path = str('results/' + directory_name + "/000000000001neutronend.out")
        analyze_data(path)
        
        if not os.path.exists(path):
          print(f"Warning: {path} does not exist. Skipping.")
          continue
        
#pass args = "folder/results" to analyze data

if __name__ == "__main__": #call script but not run fully unless specified in command line.
    main()
