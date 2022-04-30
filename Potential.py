# -*- coding: utf-8 -*-
"""
@author: Laguirre
"""

import numpy as np
from scipy.special import spherical_jn


#Hard-sphere short range potential
def hard_sphere(r, sigma):
    us = np.where(r[:,None,None] <= sigma, np.inf, 0)    
    return us

#Auxiliary function to use in the charge distributions
def theta(x):
    return spherical_jn(2,x) + spherical_jn(0,x)

#Charge distribution in Fourier space
def fourier_charge_distribution(k, qv, Rv, Sv):
    kv = k[:,None,None]
    zk = qv*((Rv**3)*theta(kv*Rv) - (Sv**3)*theta(kv*Sv))/((Rv**3)-(Sv**3))   
    return zk

#Coulomb potential in Fourier space
def electrostatic(k, zk, lb):    
    kv = k[:,None,None]
    zk_T = np.transpose(zk, [0,2,1])
    with np.errstate(divide='ignore'): # To ignore the warning due to zero division in k = 0
        uek = 4*np.pi*lb*np.matmul(zk, zk_T)/kv**2
        uek[0] = uek[1]
    return uek