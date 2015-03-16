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

    freq = np.linspace(nu1, nu2, 1e4)
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
    #comp_spec_freq = np.linspace(2000*light_speed*8.0655e-5, light_speed/1e-6, 1e2)

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
    scale = 1300 / temp_struct[0]

    return temp_struct * scale

if __name__ == '__main__':
    check = False
    if check:
        check_planck()

    dist = 2.854e27 # in cm, usuing NGC 5548 (92.5 Mpc)
    std_spec = generate_spec(2000*light_speed*8.0655e-5, light_speed/1e-6, dist)

    r_in = 1e-9  # in light days
    r_out = 40   # in light days
    disk_radii = (np.logspace(r_in, r_out) * 2.5902e15) # create log-spaced array in cm

    # Make computed SED
    comp_spec_freq = np.linspace(2000*light_speed*8.0655e-5, light_speed/1e-6, 1e3)
    temps = temp_struc(disk_radii, -3./4.)

    comp_spec_flux = np.zeros((len(comp_spec_freq), len(temps)))
    for i in range(len(temps)-1):
            comp_spec_flux[:,i] = compute_planck_freq(comp_spec_freq, temps[i])* \
                                  (2*np.pi*(disk_radii[i+1] - disk_radii[i])*disk_radii[i])

    total_comp_flux = np.sum(comp_spec_flux, axis=1)

    plt.plot(std_spec[0], std_spec[1], ls='-', lw=2, color='k', label='Input, $F^{1/3}$')
    plt.plot(comp_spec_freq, comp_spec_flux, ls='--', color='r', label='')
    plt.plot(comp_spec_freq, total_comp_flux, ls='-', lw=2, color='r', label='Computed')
    plt.xlabel(r'$\nu$', fontsize=20)
    plt.ylabel('Flux', fontsize=20)
    plt.legend()
    #plt.xlim(10e14, 10e17)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()


