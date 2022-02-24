# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 12:14:27 2017

@author: swwang
"""

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
from scipy.optimize import minimize


def dist(theta, v0):
    """calculate the distance travelled by a projectile launched
    at theta degrees with v0 (m/s) initial velocity.
    """
    g = 9.8
    theta_rad = np.pi * theta / 180
    return 2 * v0 ** 2 / g * np.sin(theta_rad) * np.cos(theta_rad)


def neg_dist(theta, v0):
    return -1 * dist(theta, v0)


result = minimize(neg_dist, 40, args=(1,))
print("optimal angle = {:.1f} degrees".format(result.x[0]))
