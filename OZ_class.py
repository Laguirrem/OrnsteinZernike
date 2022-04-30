# -*- coding: utf-8 -*-
"""
@author: Laguirre
"""

import numpy as np
from numpy.linalg import inv
from AuxFunctions import mrft, mirft, fk0, vec_to_array, array_to_vec
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import gmres

class OrnsteinZernike():
    """
    Solves the Ornstein Zernike (OZ) formalism with M components.
    
 
    Attributes
    ----------
    Rd : MxM array
        Densities of the species along the diagonal.
    ur_short : NxMxM array
        Short-ranged part of the interaction among the components in real space.
    uk_long : NxMxM array
        Long-ranged part of the interaction among the components in Fourier space.
    gs : NxMxM array
        Short-ranged indirect correlation function. This is taken as the
        initial guess to solve the OZ equation.    
    """ 
    
    #Defining the discretization shape of all the functions in the formalism.    
    #The election of N and dr depends on a proper representation of the potentials
    N = 2**19  #Must be a power of 2.
    dr = .03 
    rmax = dr*(N-1)
    dk = np.pi/(N*dr)
    r = np.arange(0, rmax + dr, dr)
    k = np.arange(0, dk*N, dk)
    
    
    def __init__(self, Rd, ur_short, uk_long, gs):
      
        self.Rd = Rd
        self.ur_short = ur_short
        self.uk_long = uk_long
        self.gs = gs
        self.gsk = mrft(self.gs, self.r, self.k)       
        self.cs = self.csij()
        self.csk = mrft(self.cs, self.r, self.k )       
        self.M = len(Rd)
        
        
    #------------------------------------------------------------------------------- 
    def radial_distribution(self):
        """
        Returns the radial distribution functions g_{ij}(r) between species i and j
        
        """
        return self.gs + self.csij() + 1
    #------------------------------------------------------------------------------- 
    #Short-ranged form of the closure relations
    def csij(self):
        # us - short-range potential 
        # gs - short-range indirect correlation function (gamma^s) 
        csij = np.exp(-self.ur_short + self.gs) - 1 - self.gs
        return csij 
    
    ##Jacobian of the short-range direct correlation function cs
    def dcs_dgs(self):
        hs = np.exp(-self.ur_short + self.gs) - 1
        return hs
    #------------------------------------------------------------------------------- 

    #Ornstein-Zernike equation
    def residual(self):
        """
        Returns the residuals of C(k) Rd [Gamma^{s}(k) + C^{s}(k)] - Gamma(k).
        
        When the residuals are zero, then the corresponding matrix Gamma^{s} is
        
        the solution of the OZ equation.

        """
        
        #Building the direct and indirect correlation functions
        ck = self.csk - self.uk_long 
        gk = self.gsk + self.uk_long    
        
        #Residuals
        residual_k = np.matmul(ck, np.matmul(self.Rd, self.csk + self.gsk)) - gk 
        
        #Symmetrizing residual_k for further numerical stability
        residual_k = (residual_k + np.transpose(residual_k,[0,2,1]))/2
        
        #Correcting residual_k(k=0) by extrapolating the nexts firsts values of residual_k
        residual_k = fk0(residual_k,self.k)
        return residual_k
    
    ##Operator that acts on xk = d_gsk. It is the negative of the jacobian of the 
    #function we are looking the roots
    def neg_residual_jacobian(self, xk):
        """
        Negative jacobian of the residuals of the OZ formalism in Fourier space,
        
        operating on xk. The jacobian is evaluated at the current value of the 
        
        short-range indirect correlation function gamma^{s} in the instance.
        
        Parameters
        ----------
        
        xk: NxMxM array
            Arbitrary array to operate the jacobian on.        
        """
        
        #Preliminar operations
        x = mirft(xk, self.k, self.r)
        ck = self.csk - self.uk_long 
        hs = self.dcs_dgs()
        fk = mrft(hs*x, self.r, self.k)   
        CR = np.matmul(ck, self.Rd)
        I = np.identity(self.Rd.shape[0])
        
        #neg_J_xk is the jacobian operator acting on xk
        neg_J_xk = np.matmul(I - CR , xk) - np.matmul(CR, fk) \
               - np.matmul(fk, np.matmul(self.Rd, self.csk+self.gsk))
               
        #Symmetrizing A_xk for further numerical stability
        neg_J_xk = (neg_J_xk + np.transpose(neg_J_xk,[0,2,1]))/2
        
        #Correcting A_xk(k=0) by extrapolating the firsts values of A_xk
        neg_J_xk = fk0(neg_J_xk, self.k)
        return neg_J_xk
    
    def linear_solver(self, neg_residual_jacobian, B):
        """
        This function finds x in the equation Ax = B. Rather than using the matrix
        A alone, it uses the vector that results of the operation Ax.
        
        A - Linear Operator
        B - vector of the same dimensions of x
        """
 
        #Preconditioning factor: It acelerates the convergence to the solution
        #of the linear equations. A and B should be multiplied by this factor
        #prior to initilize the linear solver gmres.
        P = inv(np.identity(self.M) - np.matmul(self.csk - self.uk_long, self.Rd))
        
        B = np.matmul(P, B)
          
        def A_xk(xk):
            xk = vec_to_array(xk, self.N, self.M)    
            A_xk = self.neg_residual_jacobian(xk)
            A_xk = np.matmul(P, A_xk)
            A_xk = array_to_vec(A_xk, self.N, self.M)
            return A_xk
        

        #Changes the shape of B from N matrices to a vector of N*dim(B)[0]*dim(B)[1]
        Bvec = array_to_vec(B, self.N, self.M)
       
        Axk = LinearOperator((len(Bvec),len(Bvec)),matvec = A_xk)
        
        #Linear solver. It uses the result of A on x and the vector Bvec
        xk, info = gmres(Axk,Bvec, restart = 100, tol = 10**-5)
        xk = vec_to_array(xk, self.N, self.M)  
        
        return xk

    def solve(self, tol = 10**-9, max_newton_iterations = 20):
        
        """
        Solves the OZ equation by Newton's method. The linear part of the method
        
        is solved by the Krylov type algorithm 'gmres' for a fast performance.
        
        Parameters
        ----------
        
        tol: float
            Error of residuals to consider the equation solved. The error is 
            
            defined as sqrt( sum(residuals_r^2)/(N*M*M) ), where residuals_r
            
            is the real representation the method residuals_k.
            
        max_newton_iterations: int
            Maximum number of iterations of the newton's method. 
        """
        
        #Evaluating the residual function on the initial guess of the indirect
        #correlation to check if it is already the solution of the OZ equation:
        
        #res_k is the residual in k space and res_r in real space
        res_k = self.residual()
        res_r = mirft(res_k, self.k, self.r)
        newton_err = np.sqrt(np.sum(res_r*res_r)/(self.N*self.M*self.M))   
        if newton_err < tol: 
            return print('Solved')
        
        #If the initial guess is not the solution, then iterate:
        newton_counter = 0
        while newton_err > tol and newton_counter < max_newton_iterations:      
            print('newton_method_step = {}'.format(newton_counter), 'current_error = {}'.format(newton_err))
            
            #Solving the linear system of equations (jacobian_residuals)(gsk) *  d_gsk = residuals(gsk).            
            d_gsk = self.linear_solver(self.neg_residual_jacobian, res_k)
            
            #Once d_gsk is found, all the relevant functions are updated:
            self.gsk = self.gsk + d_gsk
            self.gs = mirft(self.gsk, self.k, self.r)
            self.cs = self.csij()
            self.csk = mrft(self.cs, self.r, self.k)
                      
            #Checking the residual with the updated indirect correlation
            res_k = self.residual()
            res_r = mirft(res_k, self.k, self.r)
            newton_err = np.sqrt(np.sum(res_r*res_r)/(self.N*self.M*self.M))   
            newton_counter += 1
        
        if newton_err < tol:
            print('Solved')
        else:
            print('Solution was not reached. Try reducing the tolerance, increasing the maximum iterations or using a better initial guess for gs')
                   
   
    
    
