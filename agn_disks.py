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

light_speed = 2.9979e8  # m s^-1
def readin_data(fname):
    """
    Read in the input SED.
    """

    return 0

def compute_temp(flux):
    """
    Assume that the disk is optically thick such that
    we can approximate the spectrum at each point as
    a blackbody spectrum.

    Input is a numpy array
    """

    return flux**(1./4.)

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

    spec_radiance = ((8*np.pi*planck_c*light_speed)/(wave**5))*\
                    (1/np.expm1((planck_c*light_speed)/(boltzmann_c*temp*wave)))

    return spec_radiance

def compute_planck_freq(freq, temp):
    """
    freq is a numpy array.
    Constants in SI units
    """

    planck_c = 6.626e-34    # J s
    boltzmann_c = 1.38e-23  # J K^-1

    spec_radiance = ((8*np.pi*planck_c)/(light_speed**3))*\
                    ((freq**3)/np.expm1((planck_c*freq)/(boltzmann_c*temp)))

    return spec_radiance

def sum_planck_curves():
    """
    Take the results of compute_planck_curve and
    sum (at each wavelength?)
    """

    return 0

if __name__ == '__main__':
    check = False
    if check:
        wave = np.linspace(6.2e-10, 1e-6, 1e6)
        plt.plot(wave, compute_planck_wave(wave, 4500))
        plt.plot(wave, compute_planck_wave(wave, 6000))
        plt.plot(wave, compute_planck_wave(wave, 7500))
        plt.show()

    test_wave = np.array([3.55, 4.49, 5.72, 7.87, 23.68, 71.42, 155.9])
    test_flux = np.array([132.65, 80.18, 58.74, 35.24, 8.8, 25.9, 87.4])
    print compute_temp(test_flux)

    #plt.plot(test_wave, test_flux, ls='none', marker='o', color='k')
    #plt.show()
