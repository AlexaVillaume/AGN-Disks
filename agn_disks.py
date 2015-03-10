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


light_speed = 2.9979e11  # cm s^-1

def readin_data(fname):
    """
    Read in the input SED.
    """
    return 0

def generate_spec(nu1, nu2):
    """
    Generate a fake spectrum under the assumptions of
    the standard accretion disk model.

    dist needs to be in cm

    returns spectrum with luminosity at the source of
    the object
    """

    freq = np.linspace(nu1, nu2, 1e4)
    return [freq, (freq**(1./3.))]

def compute_planck_freq(freq, temp, dist):
    """
    freq is a numpy array. Constants in SI units.
    Returning units of flux density (jansky). Assuming that the
    angular size (?) of the object is << 1
    """

    planck_c = 6.626e-34    # J s
    boltzmann_c = 1.38e-23  # J K^-1

    theta = 1e-3 # for now until I can get real values

    spec_radiance = ((2*planck_c*freq**3)/(light_speed**2))*\
                    (1/np.expm1((planck_c*freq)/(boltzmann_c*temp)))

    return spec_radiance
    #return spec_radiance*4*np.pi*dist**2

def sum_planck_curves():
    """
    Take the results of compute_planck_curve and
    sum (at each wavelength?)
    """
    return 0

def temp_struc(radii, power):
    return radii**power

if __name__ == '__main__':
    check = False
    if check:
        check_planck()

    r_in = (40*(1./365))*9.4606e17
    r_out = (150*(1./365))*9.4606e17

    std_spec = generate_spec(2000*light_speed*8.0655e-5, light_speed/1e-6)

    # Make computed SED
    comp_spec_freq = np.linspace(2000*light_speed*8.0655e-5, light_speed/1e-6, 1e2)
    disk_radii = np.linspace(r_in, r_out)

    comp_spec_flux = []
    temps = temp_struc(disk_radii, -3./4.)
    for nu in comp_spec_freq:
        lum = []
        for temp in temps:
            lum.append(compute_planck_freq(temp, nu, 1000)*np.pi)
        comp_spec_flux.append(sum(lum)*1e59)

    plt.plot(std_spec[0], std_spec[1], ls='none', marker='o', color='k', label='Input, $F^{1/3}$')
    plt.plot(comp_spec_freq, comp_spec_flux, ls='-', color='r', label='Computed')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.show()


