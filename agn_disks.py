'''
INPUT:
    Spectral Energy Distribution
OUTPUT:
    Temperature structure and new SED
    Plot of input and calculated SED
'''

import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u


light_speed = 2.9979e8  # m s^-1
planck_c = 6.626e-34    # J s
boltzmann_c = 1.38e-23  # J K^-1
grav_c = 6.67384e-11    # m^3 kg^-1 s^-2

def readin_data(fname):
    """
    Read in the input SED.
    """
    return 0

def generate_spec(nu1, nu2, dist):
    """
    Generate a fake spectrum under the assumptions of
    the standard accretion disk model.

    dist needs to be in cm

    returns spectrum with luminosity at the source of
    the object
    """
    freq = np.linspace(nu1, nu2, 1e3)
    return [freq, (freq**(1./3.))]

def compute_planck_freq(freq, temp):
    """
    freq is a numpy array. Constants in SI units.
    Returning units of flux density (jansky). Assuming that the
    angular size (?) of the object is << 1
    """


    spec_radiance = (8.*np.pi*planck_c*freq**3.)/(light_speed**2)*\
                    1./(np.exp(planck_c*freq/(boltzmann_c*temp))-1)

    return spec_radiance

def sum_planck_curves():
    """
    Take the results of compute_planck_curve and
    sum (at each wavelength?)
    """
    return 0

def check_planck():
    # http://www.astro.umd.edu/~cychen/MATLAB/ASTR120/Blackbody2.html
    temps = np.linspace(3000, 9000, 3)
    comp_spec_freq = np.linspace(light_speed/1e-9, light_speed/3000e-9, 1e6)

    comp_spec_flux = np.zeros((len(comp_spec_freq), len(temps)))
    for i, temp in enumerate(temps):
        comp_spec_flux[:,i] = compute_planck_freq(comp_spec_freq, temp)

    plt.plot(comp_spec_freq, comp_spec_flux[:,0])
    plt.plot(comp_spec_freq, comp_spec_flux[:,1])
    plt.plot(comp_spec_freq, comp_spec_flux[:,2])
    plt.plot(comp_spec_freq, np.sum(comp_spec_flux, axis=1), color='k')

    plt.xlim(1e14,9e14)
    plt.show()

    sys.exit()

def temp_struc(radii, power):
    """
    Scale the temperatures to 1300 K at
    inner disk
    """
    temp_struct = radii**power
    scale = 1300 / temp_struct[len(temp_struct)-1]

    return temp_struct * scale

def find_peak_luminosity(freqs, flux, disk_radii, wavelength, freq, color):
    emiss_lum = []
    index = (np.abs(freqs - freq)).argmin()

    for i in range(len(disk_radii)):
        emiss_lum.append(flux[:,i][index])
    emiss_lum = np.asarray(emiss_lum)

    max_ind = emiss_lum.argmax()
    print disk_radii[max_ind]/2.5902e15
    plt.axvline(disk_radii[max_ind]/2.5902e15, color=color, ls='--')
    plt.plot(disk_radii/2.5902e15, emiss_lum, label=wavelength, color=color, lw=2)

if __name__ == '__main__':
    check = False
    if check:
        check_planck()
    '''
    Make computed SED
    '''
    #dist = 2.854e27 # in cm, usuing NGC 5548 (92.5 Mpc)
    dist =  48*(3.08567758e24) # converting from Mpc to cm, for NGC 7469
    bh_mass = (3e7)*(1.981e30) # converting from solar mass to kg

    r_in = (3*((2*grav_c*bh_mass)/light_speed**2))*3.86e-14  # in light days
    r_out = 80   # in light days
    disk_radii = (np.linspace(r_in, r_out, 1e2)*2.5902e15) # create log-spaced array in cm
    disk_max = max(disk_radii)
    disk_min = min(disk_radii)
    # make this logspaced
    disk_radii_log = []
    for i, radius in enumerate(disk_radii):
        q2 = np.exp(np.log(disk_max/disk_min)/(len(disk_radii)-1))
        disk_radii_log.append(disk_min*q2**(i-1.0))
    disk_radii_log = np.asarray(disk_radii_log)
    temps = temp_struc(disk_radii_log, -0.58)

    comp_spec_freq = np.linspace(light_speed/1e-9, light_speed/1e-4, 1e5)
    comp_spec_flux = np.zeros((len(comp_spec_freq), len(temps)))
    for i in range(len(temps)-1):
        comp_spec_flux[:,i] = compute_planck_freq(comp_spec_freq, temps[i])* \
                              (2*np.pi*(disk_radii_log[i+1] - disk_radii_log[i])*disk_radii_log[i])
    total_comp_flux = np.sum(comp_spec_flux, axis=1)

    wavelengths = [1315, 1810, 4865, 6962]
    frequencies = [2281368821292775.5, 1657458563535911.8, 616649537512847., 430910657856937.7]
    colors = ['k', 'g', 'r', 'b']
    for color, wave, freq in zip(colors, wavelengths, frequencies):
        find_peak_luminosity(comp_spec_freq, comp_spec_flux, disk_radii_log, wave, freq, color)
    #find_peak_luminosity(comp_spec_freq, comp_spec_flux, disk_radii_log, 5100, 5.9*10**14, 'b')
    plt.xlabel('Light Days', fontsize=20)
    plt.ylabel('Luminosity', fontsize=20)
    plt.legend(frameon=False, loc='upper right', fontsize=12)
    plt.ylim(1e25, 2.5e26)
    plt.tight_layout()
    plt.show()

    #std_spec = generate_spec(3.20435466e-16, light_speed/5300e-10, dist)
    #plt.plot(std_spec[0], std_spec[1], ls='-', lw=2, color='k', label='Input, $F^{1/3}$')

    #plt.plot(comp_spec_freq, comp_spec_flux, ls='--', color='r', label='')
    plt.plot(comp_spec_freq, total_comp_flux, ls='-', lw=2, color='r', label='Computed')
    plt.xlabel(r'$\nu$', fontsize=20)
    plt.ylabel('Flux', fontsize=20)
    plt.legend(frameon=False, loc='upper left', fontsize=12)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()


