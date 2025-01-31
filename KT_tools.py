import numpy as np
import scipy.optimize

# calculate cross section (dV/dz) and differential surface area (dA/dz) of stainless steel components in TUCAN guide setup (gate valve + adapter + optional endcap) at an array of z positions
# Parameters:
#  z: array of z positions
#  z0: center of guide
#  steelEndcap: set to true if the guide end cap was steel
# Returns:
#  arrays of dA/dz and dV/dz at z positions
def guideStainlessSurface(z, z0, steelEndcap):
    r_valve = 0.043052 # radius of valve face
    r_guide = 0.047752 # radius of guide
    x_valveAdapter = 0.096558 # length coordinate where valve adapter ends
    valveFace = np.where(np.abs(z - z0) < r_valve, 2*np.sqrt(r_valve**2 - (z - z0)**2), 0.) # surface slice dA/dz of valve face at height z
    
    x = np.array([0., 2.708e-3, x_valveAdapter]) # coordinates of individual components along length of the setup
    r = np.array([r_valve, 0.042342, 0.042342]) # radii of cylindrical components along the length of the setup
    x_hd = np.linspace(0., x_valveAdapter, 100)
    r_hd = np.expand_dims(np.interp(x_hd, x, r), -1) # interpolate radius r of guide setup along its length x
    
    r_z2 = r_hd**2 - (z - z0)**2 # r_z = radius projected onto plane at z
    adapterSurface = np.trapz(np.where(r_z2 > 0, 2*r_hd/np.sqrt(r_z2), 0.), x_hd, axis = 0) # surface slice dA/dz of cylindrical components is the integral 2r/r_z dx along the length
    endcap = np.where(steelEndcap & (np.abs(z - z0) < r_guide), 2*np.sqrt(r_guide**2 - (z - z0)**2), 0.) # surface slice dA/dz of valve face at height z
            
    crossSection = np.trapz(np.where(r_z2 > 0, 2*np.sqrt(r_z2), 0.), x_hd, axis = 0) # cross section dV/dz is the integral 2r_z dx along the length
        
    return valveFace + adapterSurface + endcap, crossSection # return total dA/dz (sum of valve face, adapter, and end cap) and dV/dz

def guideDLCSurface(z, z0, steelEndcap = False):
    r_guide = 0.047752 # radius of guide
    x_guide = 1.0927 # length coordinate where guide ends     #check this out with Russ
    x_DLCCap = 1.0127 # length coordinate of end cap
    
    x = np.array([0, x_guide, x_DLCCap]) # coordinates of individual components along length of the setup
    r = np.array([r_guide, r_guide, r_guide]) # radii of cylindrical components along the length of the setup
    if steelEndcap: # if lengths of setups with and without steel end cap are different
        x_hd = np.linspace(0, x_guide, 1100)
    else:
        x_hd = np.linspace(0, x_DLCCap, 1100)
    r_hd = np.expand_dims(np.interp(x_hd, x, r), -1)
    
    r_z2 = r_hd**2 - (z - z0)**2
    guideSurface = np.trapz(np.where(r_z2 > 0, 2*r_hd/np.sqrt(r_z2), 0.), x_hd, axis = 0)
    endcap = np.where((not steelEndcap) & (np.abs(z - z0) < r_guide), 2*np.sqrt(r_guide**2 - (z - z0)**2), 0.)
            
    crossSection = np.trapz(np.where(r_z2 > 0, 4*np.sqrt(r_z2), 0.), x_hd, axis = 0)
        
    return guideSurface + 2*endcap, crossSection    

# calculate cross section (dV/dz) and differential surface area (dA/dz) of NiP components in TUCAN guide setup (adapter + guide + adapter + optional endcap) at an array of z positions
# Parameters:
#  z: array of z positions
#  z0: center of guide
#  steelEndcap: set to true if the guide end cap was steel, false if it was NiP
# Returns:
#  arrays of dA/dz and dV/dz at z positions
def guideNipSurface(z, z0, steelEndcap):
    r_adapter = 0.042342 # smaller radius of 85-95 guide adapter
    r_guide = 0.047752 # radius of guide
    x_adapter = 0.096558 # length coordinate where adapter starts
    x_guide = 1.122108 # length coordinate where guide ends
    x_nipCap = 1.145358 # length coordinate of end cap
    
    x = np.array([x_adapter, 0.105458, 0.122108, x_guide, 1.138759, x_nipCap]) # coordinates of individual components along length of the setup
    r = np.array([r_adapter, r_adapter, r_guide, r_guide, r_adapter, r_adapter]) # radii of cylindrical components along the length of the setup
    if steelEndcap: # if lengths of setups with and without steel end cap are different
        x_hd = np.linspace(x_adapter, x_guide, 1100)
    else:
        x_hd = np.linspace(x_adapter, x_nipCap, 1100)
    r_hd = np.expand_dims(np.interp(x_hd, x, r), -1)
    
    r_z2 = r_hd**2 - (z - z0)**2
    guideSurface = np.trapz(np.where(r_z2 > 0, 2*r_hd/np.sqrt(r_z2), 0.), x_hd, axis = 0)
    endcap = np.where((not steelEndcap) & (np.abs(z - z0) < r_adapter), 2*np.sqrt(r_adapter**2 - (z - z0)**2), 0.)
            
    crossSection = np.trapz(np.where(r_z2 > 0, 2*np.sqrt(r_z2), 0.), x_hd, axis = 0)
        
    return guideSurface + endcap, crossSection    

# def 1mGuideSurface()


# calculate cross section (dV/dz) and differential surface area (dA/dz) of stainless steel components in TUCAN tail setup (gate valve + valve adapter + tail adapter) at an array of z positions
# Parameters:
#  z: array of z positions
#  z0: center of guide
# Returns:
#  arrays of dA/dz and dV/dz at z positions
def tailStainlessSurface(z, z0):
    r_valve = 0.043052
    r_tailAdapter = 0.05234
    x_tailAdapter = 0.119758
    valveFace = np.where(np.abs(z - z0) < r_valve, 2*np.sqrt(r_valve**2 - (z - z0)**2), 0.)
    
    x = np.array([0., 2.708e-3, 0.101558, x_tailAdapter])
    r = np.array([r_valve, 0.042342, 0.042342, r_tailAdapter])
    x_hd = np.linspace(0., x_tailAdapter, 120)
    r_hd = np.expand_dims(np.interp(x_hd, x, r), -1)
    r_z2 = r_hd**2 - (z - z0)**2
        
    adapterSurface = np.trapz(np.where(r_z2 > 0, 2*r_hd/np.sqrt(r_z2), 0.), x_hd, axis = 0)
    
    r_tailTube = 0.07409
    tailAdapterFace = np.where((np.abs(z - z0) >= r_tailAdapter) & (np.abs(z - z0) <= r_tailTube),
                               2*np.sqrt(r_tailTube**2 - (z - z0)**2), 0.)
    tailAdapterFace = np.where((np.abs(z - z0) < r_tailAdapter) & (np.abs(z - z0) <= r_tailTube),
                               2*np.sqrt(r_tailTube**2 - (z - z0)**2) - 2*np.sqrt(r_tailAdapter**2 - (z - z0)**2), tailAdapterFace)
        
    crossSection = np.trapz(np.where(r_z2 > 0, 2*np.sqrt(r_z2), 0.), x_hd, axis = 0)
        
    return valveFace + adapterSurface + tailAdapterFace, crossSection


# calculate cross section (dV/dz) and differential surface area (dA/dz) of TUCAN tail at an array of z positions
# Parameters:
#  z: array of z positions
#  z0: center of guide
# Returns:
#  arrays of dA/dz and dV/dz at z positions
def tailNipSurface(z, z0):
    r_tube = 0.07409
    r_bulb = 0.180
    x_tube = 0.119758
    x_bulb1 = 2.579105
    x_bulb2 = 2.605105
    z_bulb = z0 - 0.085

    r_z2Tube = r_tube**2 - (z - z0)**2
    r_z2Bulb = r_bulb**2 - (z - z_bulb)**2
    x_intersect = x_bulb1 - np.sqrt(r_z2Bulb - r_z2Tube)
    tubeSurface = np.where(r_z2Tube > 0, 2*r_tube/np.sqrt(r_z2Tube) * (x_intersect - x_tube), 0.)
    
    phi_intersect = 2*np.arcsin(np.sqrt(r_z2Tube/r_z2Bulb))
    bulbSurface = np.where(r_z2Bulb > 0,
                           2*np.pi*r_bulb + 2*(x_bulb2 - x_bulb1)*r_bulb/np.sqrt(r_z2Bulb), 0.)
    bulbSurface = np.where(r_z2Tube > 0,
                           bulbSurface - np.sqrt(r_z2Bulb)*phi_intersect*r_bulb/np.sqrt(r_z2Bulb), bulbSurface)
    
    bulbCrossSection = np.where(r_z2Bulb > 0,
                                r_z2Bulb*np.pi + 2*(x_bulb2 - x_bulb1)*np.sqrt(r_z2Bulb), 0.)
    tubeCrossSection = np.where(r_z2Tube > 0,
                                2*np.sqrt(r_z2Tube)*(x_intersect - x_tube) - r_z2Bulb/2*(phi_intersect - np.sin(phi_intersect)),
                                0.)
    return tubeSurface + bulbSurface, tubeCrossSection + bulbCrossSection


m = 10.454 # neutron mass (neV s^2/m^2)
mg = 102.5 # neutron mass * gravity (neV/m)


# calculate phase space available to UCN with total energies H in a volume with cross section dV/dz
# Parameters:
#  H: array of UCN total energies
#  z: array of z positions
#  crossSection: array of volume cross sections dV/dz at positions z
# Returns:
#  array of phase space available to UCN with energies H
def phaseSpace(H, z, crossSection):
    zz = np.expand_dims(z, -1) # expand the z coordinates into a 2D array
    vv = np.expand_dims(crossSection, -1) # expand energies into a 2D array
    return np.trapz(np.where((H - mg*zz > 0) & (H > 0), vv*np.sqrt((H - mg*zz)/H), 0.), z, axis = 0) # phase space is the integral dV/dz * sqrt( (H - mg z)/H ) dz


# calculate loss probability for UCN with energies E hitting a surface with Fermi potential V - iW, averaged over isotropic velocity distribution
# Parameters:
#  E: array of UCN energies
#  V: real Fermi potential
#  W: imaginary Fermi potential
# Returns:
#  array of loss probabilities at energies E, averaged over isotropic velocity distribution
def lossPerBounceFermi(E, V, W):
    U = V - 1j*W
    theta = np.linspace(0., np.pi/2, 200) # create array of theta angle [0 .. pi/2] to integrate over
    Eperp = np.expand_dims(E, -1) * np.cos(theta)**2 # calculate E*cos(theta)^2 for each E and theta
    reflAmplitude = np.divide(np.sqrt(Eperp) - np.sqrt(Eperp - U), np.sqrt(Eperp) + np.sqrt(Eperp - U)) # reflection amplitude at E*cos(theta)^2
    lossProb = 1. - np.real(reflAmplitude)**2 - np.imag(reflAmplitude)**2 # loss probability = 1 - | reflection amplitude |^2

    return np.trapz(2*lossProb*np.cos(theta)*np.sin(theta), theta) # integrate over hemisphere of incoming angles

def negativeexp(x, A, tau):   #fit function for typical ucn storage experiment
       return A * np.exp(-x / tau)


# approximation of lossPerBounceFermi if W/V << 1 and E < V
def lossPerBounceApprox(E, V, W):
    return 2.*W/V*(V/E * np.arcsin(np.sqrt(E/V)) - np.sqrt(V/E - 1))


def lossRateApprox(E, A, V, mu):
    return np.sqrt(2*E/m)*A/(4*V)*mu + 1/880


# calculate loss probability for UCN with energies E hitting a surface with Fermi potential V - iW and additional constant loss probability mu
# Parameters:
#  E: array of UCN energies
#  V: real Fermi potential
#  W: imaginary Fermi potential
#  mu: constant loss probability
# Returns:
#  array of loss probabilities at energies E
def lossPerBounce(E, V, W, mu):
    return np.where(E < V, lossPerBounceApprox(E, V, W) + mu, lossPerBounceFermi(E, V, W) + mu)


# loss rate of UCN with total energies H in storage volume with Fermi potential V - iW, constant loss probability mu, cross section dV/dz, and differential surface area dA/dz
# Parameters:
#  H: array of UCN total energies
#  V: real Fermi potential
#  W: imaginary Fermi potential
#  mu: constant loss probability
#  z: array of z positions
#  crossSection: array of volume cross sections dV/dz at positions z
#  surface: array of differential surface areas dA/dz at positions z
# Returns:
#  array of loss rates for UCNs with total energies H
def lossRate(H, V, W, mu, z, crossSection, surface):
    velocity = np.sqrt(2.*H/m) # velocities at z = 0
    phSpace = phaseSpace(H, z, crossSection) # phase space available to UCN with energy H
    
    zz = np.expand_dims(z, -1)
    aa = np.expand_dims(surface, -1)
    mumu = np.where(H - mg*zz > 0, lossPerBounce(H - mg*zz, V, W, mu), 0.) # calculate average loss per bounce for each H and z
    return velocity / 4 / phSpace * np.trapz((H - mg*zz)/H * aa * mumu, z, axis = 0) + 1./880 # loss rate = integral sqrt(2H/m)/(4 phasespace) (H - mgz)/H dA/dz mu dz + (beta decay rate)

# normalized energy spectrum (E - Erange[0])^exponent between Erange[0] and Erange[1]
# Parameters:
#  E: array of energies where the spectrum should be evaluated
#  Erange: tuple of minimum and maximum energy
#  exponent: exponent
# Returns:
#  array of probabilities to find UCNs between E_i and E_i+1
def spectrum(E, Erange, exponent):
    if exponent <= -1:
        raise FloatingPointError()
    norm = (Erange[1] - Erange[0])**(exponent + 1) / (exponent + 1)
    return np.where((E > Erange[0]) & (E < Erange[1]), (E - Erange[0])**exponent / norm, 0.)

# calculate number of remaining UCN after storage in volume with two different surfaces
# Parameters:
#  N0: initial number of UCN
#  t: storage time
#  Erange: total energy spectrum range
#  exponent: exponent of total energy spectrum
#  V1: real Fermi potential of first surface
#  W1: imaginary Fermi potential of first surface
#  mu1: constant loss probability of first surface
#  V2: real Fermi potential of second surface
#  W2: imaginary Fermi potential of second surface
#  mu2: constant loss probability of second surface
#  z: array of z positions
#  crossSection: array of volume cross sections dV/dz at positions z
#  surface1: array of differential surface areas dA/dz at positions z (of surface 1)
#  surface2: array of differential surface areas dA/dz at positions z (of surface 2)
# Returns:
#  array of numbers of remaining UCNs after each storage time
def storedUCN(N0, t, Erange, exponent, V1, W1, mu1, V2, W2, mu2, z, crossSection, surface1, surface2):
    H = np.linspace(Erange[0], Erange[1])
    N0_H = N0*np.expand_dims(spectrum(H, Erange, exponent), -1)
    loss_H = np.expand_dims(lossRate(H, V1, W1, mu1, z, crossSection, surface1) + lossRate(H, V2, W2, mu2, z, crossSection, surface2), -1)
    N = np.trapz(np.where(loss_H > 0., N0_H * np.exp(-t * loss_H), 0.), H, axis = 0)
    return N

# perform fit to calculate imaginary Fermi potential of one surface, given numbers of remaining neutrons after several different storage times in volume with two different surfaces
# Parameters:
#  t: array of storage times
#  N: array of numbers of remaining UCNs
#  dN: array of uncertainties on numbers of remaining UCNs
#  Erange: total energy spectrum range
#  exponent: exponent of total energy spectrum
#  V1: real Fermi potential of first surface
#  V2: real Fermi potential of second surface
#  W2: imaginary Fermi potential of second surface
#  z: array of z positions
#  crossSection: array of volume cross sections dV/dz at positions z
#  surface1: array of differential surface areas dA/dz at positions z (of surface 1)
#  surface2: array of differential surface areas dA/dz at positions z (of surface 2)
# Returns:
#  fit results for imaginary Fermi potential of surface 1, initial number of UCNs, uncertainties, and array of numbers of remaining UCNs from fit
def fitW1(t, N, dN, Erange, exponent, V1, V2, W2, z, crossSection, surface1, surface2):
    Nfit = lambda t, W1, N0: storedUCN(N0, t, Erange, exponent, V1, W1, 0., V2, W2, 0., z, crossSection, surface1, surface2)
    popt, pcov = scipy.optimize.curve_fit(Nfit, t, N, [0.05, 10000], dN, absolute_sigma = True)
    perr = np.sqrt(np.diag(pcov))
    return popt[0], popt[1], perr[0], perr[1], Nfit(t, popt[0], popt[1])


# perform fit to calculate constant loss probability of one surface, given numbers of remaining neutrons after several different storage times in volume with two different surfaces
# Parameters:
#  t: array of storage times
#  N: array of numbers of remaining UCNs
#  dN: array of uncertainties on numbers of remaining UCNs
#  Erange: total energy spectrum range
#  exponent: exponent of total energy spectrum
#  V1: real Fermi potential of first surface
#  V2: real Fermi potential of second surface
#  mu2: constant loss probability of second surface
#  z: array of z positions
#  crossSection: array of volume cross sections dV/dz at positions z
#  surface1: array of differential surface areas dA/dz at positions z (of surface 1)
#  surface2: array of differential surface areas dA/dz at positions z (of surface 2)
#  fit results for constant loss probability of surface 1, initial number of UCNs, uncertainties, and array of numbers of remaining UCNs from fit
def fitMu1(t, N, dN, Erange, exponent, V1, V2, mu2, z, crossSection, surface1, surface2):
    Nfit = lambda t, mu1, N0: storedUCN(N0, t, Erange, exponent, V1, 0., mu1, V2, 0., mu2, z, crossSection, surface1, surface2)
    popt, pcov = scipy.optimize.curve_fit(Nfit, t, N, [0.0003, 10000], dN, absolute_sigma = True)
    perr = np.sqrt(np.diag(pcov))
    return popt[0], popt[1], perr[0], perr[1], Nfit(t, popt[0], popt[1])
