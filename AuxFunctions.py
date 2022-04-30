# -*- coding: utf-8 -*-
"""
@author: Laguirre
"""


import numpy as np
from scipy.fftpack import fft
from scipy.linalg import inv


##Sine transform
def dst(f, **kargs):
    
    #Length of the function f
    if 'f_len' in kargs.keys():
        N = kargs['f_len']
    else:
        N = len(f)
    
    #Fast fourier sine transform by using the parity of f and the fast fourier transform fft.
    f2 = np.zeros(2*N)
    f2[0] = f2[N] = 0.0
    f2[1:N] = f[1:]
    f2[:N:-1] = -f[1:]
    sine_transform = -0.5*(fft(f2).imag)[:N]
    return sine_transform



#Radial Fourier Transform
def rft(radial_function_r, r, k, **kargs):
    """
    Fourier transform of a radially symmetric function in r space.
    
    Parameters
    ----------
    radial_function_k: N array
        Real space function to obtain its transform.
    r: N array
        Discretization of the real space.
    k: N array
        Discretization of the Fourier space.
        
    N: int, optional
        Length of r
    """
    
    #Length of the function f
    if 'f_len' in kargs.keys():
        N = kargs['f_len']
    else:
        N = len(radial_function_r)
        
    #Separation between points of r
    dr = r[1]
    
    #Radial Fourier transform by using the fast sine transform.
    #-------------------------------------------------------------
    g = radial_function_r*r
    
    #g_0 is the value of the transform at r = 0
    g_0 = np.sum(g*r).real
    
    #g_r is the value of the transform at r != 0
    st = dst(g)
    g_r = (st[1:N]/k[1:N]).real
    #-------------------------------------------------------------
    return (4*np.pi)*dr*np.append(g_0, g_r)


#Inverse Fourier transform
def irft(radial_function_k, k, r, **kargs):
    """
    Inverse Fourier transform of a radially symmetric function in k space.
    
    Parameters
    ----------
    radial_function_k: N array
        Fourier space function to obtain its inverse transform.
    k: N array
        Discretization of the Fourier space.
    r: N array
        Discretization of the real space.
        
    N: int, optional
        Length of k
    """
    
    #Length of the function f
    if 'f_len' in kargs.keys():
        N = kargs['f_len']
    else:
        N = len(radial_function_k)
        
    #Separation between points of k
    dk = k[1]
        
     #Radial inverse Fourier transform by using the fast sine transform.
    #-------------------------------------------------------------   
    g = radial_function_k*k
    
    #g_0 is the value of the transform at k = 0
    g_0 = np.sum(g*k).real
    
    #g_r is the value of the transform at k != 0
    st = dst(g)
    g_k = (st[1:N]/r[1:N]).real
    #-------------------------------------------------------------
    return dk*(1/(2*np.pi**2))*np.append(g_0, g_k)

def mrft(g, r, k):
    """
    Fourier transform of a matrix function g_{Mc Mr}(r).  
    
    Parameters:
    -----------
    g: N x Mr x Mc array
        Matrix function to obtain its inverse transform.
    r: N array
        Discretization of the real space.
    k: N array
        Discretization of the Fourier space.
    
    """
    #-------------------------------------------------------------------   
    g_dimensions = g.shape
    N = g_dimensions[0]
    Mr = g_dimensions[1]
    Mc = g_dimensions[2]
    #-------------------------------------------------------------------
    
    g = np.transpose(g, [1,2,0])
    gk = np.zeros([Mr,Mc,N])
    
    #Operations for symmetrical matrices
    if Mr == Mc:       
        for i in range(0,Mr):
            for j in range(i,Mr):
                gk[i,j] = gk[j,i] = rft(g[i,j], r, k, f_len = N)
        gk = np.transpose(gk,[2,0,1])
    
    #Operations for npn-symmetrical matrices
    else:
        for i in range(0,Mr):
            for j in range(0,Mc):
                gk[i,j] = rft(g[i,j], r, k, f_len = N)
        gk = np.transpose(gk,[2,0,1])
        
    return gk

def mirft(gk, k, r):
    """
    Inverse Fourier transform of a matrix function g_{Mc Mr}(k).  
    
    Parameters:
    -----------
    gk: N x Mr x Mc array
        Matrix function to obtain its inverse transform.
    k: N array
        Discretization of the Fourier space.
    r: N array
        Discretization of the real space.
    
    """
    #-------------------------------------------------------------------
    gk_dimensions = gk.shape
    N = gk_dimensions[0]
    Mr = gk_dimensions[1]
    Mc = gk_dimensions[2]
    #-------------------------------------------------------------------
    
    gk = np.transpose(gk, [1,2,0])
    g = np.zeros([Mr,Mc,N])
    
    #Operations for symmetrical matrices
    if Mr == Mc:       
        for i in range(0,Mr):
            for j in range(i,Mr):
                g[i,j] = g[j,i] = irft(gk[i,j], k, r, f_len = N)
        g = np.transpose(g,[2,0,1])
    
    #Operations for non-symmetrical matrices
    else:
        for i in range(0,Mr):
            for j in range(0,Mc):
                g[i,j] = irft(gk[i,j], k, r, f_len = N)
        g = np.transpose(g,[2,0,1])
 
    return g

#Given a function f(k), with k = (0, dk, 2*dk..., N*dk), extrapolates the value f(k=0)
#from a cuadratic function obtained from the values f(dk), f(2*dk) and f(3*dk)
#The shape of f(k) has to be N x Mc x Mr, where N match the dimension of k and
#Mc and Mr are arbitrary positive integers.
def fk0(f,k):
    kinv = inv(np.array([[k[1]**2, k[1], 1], [k[2]**2, k[2], 1], [k[3]**2, k[3],1]]))
    ft = np.transpose(f[1:4],[1,2,0])
    
    Mc = f.shape[1]
    Mr = f.shape[2]
    coef = np.zeros([Mc,Mr,3])
    for x in range(0, Mc):
        for y in range(0,Mr):
            coef[x,y] = np.matmul(kinv, ft[x,y])            
    coef = np.transpose(coef, [2,0,1])    
    f[0] = coef[2]
    return f   

#Reshaping functions:
#------------------------------------------------------------------------------
#Reshapes a vector of N*(M+1)*M/2 entries, to an array of shape N,M,M
def vec_to_array(x,N,M):
    vec_of_matr = np.zeros([N,M,M])
    num = 0
    for im in range(0,M):
        for jm in range(im,M):
            num = num + 1
            vec_of_matr[0:N, im, jm] = vec_of_matr[0:N, jm, im] = x[(num-1)*N:num*N]            
    return vec_of_matr    


#Reshapes an array of shape N,M,M, to a vector with N*(M+1)*M/2 entries
def array_to_vec(x,N,M):
    vec = np.zeros(int(N*M*(M+1)/2))
    num = 0
    for im in range(0,M):
        for jm in range(im,M):
            num = num + 1
            vec[(num-1)*N:num*N] = x[0:N, im, jm]
    return vec
#------------------------------------------------------------------------------