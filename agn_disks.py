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

def readin_data(fname):
    """
    Read in the input SED.
    """

    return 0

def compute_temp_struct(flux):
    """
    Assume that the disk is optically thick such that
    we can approximate the spectrum at each point as
    a blackbody spectrum.

    Input is a numpy array
    """

    return flux**(1./4.)

def compute_planck_curve(wave, temp):
    """
    Compute blackbodies using Planck's lawat each
    flux point using the the result from
    compute_temp_struct.

    wave is a numpy array.

    Constants in SI units
    """

    planck_c = 6.626e-34    #
    boltzmann_c = 1.38e-23  #
    light_speed = 3e8       #m s^-1

    spec_radiance = ((2*planck_c*light_speed**2)/wave**5)*\
                    (1/np.expm1((planck_c*light_speed)/(wave*boltzmann_c*temp)))

    return spec_radiance

def sum_planck_curves():
    """
    Take the results of compute_planck_curve and
    sum (at each wavelength?)
    """

    return 0


wave = np.linspace(1e-5,3)

plt.plot(wave, compute_planck_curve(wave, 3000))
plt.xscale('log')
plt.yscale('log')
plt.show()
