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

light_speed = 2.9979e11  # cm s^-1

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
    return [freq, (freq**(1./3.))*4*np.pi*dist**2]

def compute_planck_wave(wave, temp):
    """
    Compute blackbodies using Planck's lawat each
    flux point using the the result from
    compute_temp_struct.

    wave is a numpy array.

    Constants in SI units
    """

    planck_c = 6.626e-34    # J s
    boltzmann_c = 1.38e-23  # J K^-1

    spec_radiance = ((2*planck_c*light_speed**2)/(wave**5))*\
                    (1/np.expm1((planck_c*light_speed)/(boltzmann_c*temp*wave)))

    return spec_radiance

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

    return spec_radiance*4*np.pi*dist**2

def check_planck():
    """
    Plot some test Planck curves to make sure those functions
    are working
    """
    fig = plt.figure(figsize=(11, 5.5))

    ax1 = plt.subplot(1,2,1)
    wave = np.linspace(6.2e-10, 1e-6, 1e6)
    ax1.plot(wave, compute_planck_wave(wave, 4500))
    ax1.plot(wave, compute_planck_wave(wave, 6000))
    ax1.plot(wave, compute_planck_wave(wave, 7500))

    ax2 = plt.subplot(1,2,2)
    freq = np.linspace(0, 8e13, 1e6)
    ax2.plot(freq, compute_planck_freq(freq, 500))
    ax2.plot(freq, compute_planck_freq(freq, 400))
    ax2.plot(freq, compute_planck_freq(freq, 200))
    plt.show()

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

    std_spec = generate_spec(2000*light_speed*8.0655e-5, light_speed/1e-6, 1000)

    # Make computed SED
    comp_spec_freq = np.linspace(2000*light_speed*8.0655e-5, light_speed/1e-6, 1e2)
    disk_radii = np.logspace(1, 10) # units of ? reasonable values?

    comp_spec_flux = []
    for nu in comp_spec_freq:
        temps = temp_struc(disk_radii, -3./4.)
        lum = []
        for temp in temps:
            lum.append(compute_planck_freq(temp, nu, 1000))
        comp_spec_flux.append(sum(lum)*1e39)

    plt.plot(std_spec[0], std_spec[1], ls='none', marker='o', color='k')
    plt.plot(comp_spec_freq, comp_spec_flux, ls='-', color='r')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()


