# misc_splines.py
# -*- coding: utf-8 -*-
#
# Bug fixed for the `ni` glm python modeling toolbox
# by Jacob Huth
#
# (c) 2012 IKW Universität Osnabrück
# ported to Python by Robert Costa <rcosta@uni-osnabrueck.de>
#
# This version is based on the original Matlab code written by ??????
# in ???? 20??.
#
from __future__ import division
import numpy as np

def create_splines_linspace(length, nr_knots, remove_last_spline):
    """
        Generates B-spline basis functions based on the length and number of
        knots of the ongoing iteration
    """
    design_matrix = create_splines(length+1, nr_knots, remove_last_spline, lambda l,n: np.round(np.linspace(2, l, n)))
    design_matrix[2,0] = 1.0
    return design_matrix[2:,:]

def create_splines_logspace(length, nr_knots, remove_last_spline):
    """
        Generates B-spline basis functions based on the length and number of 
        knots of the ongoing iteration
     
        A logarithmic spacing of knots means that there are more nuanced history
        effects in the immediate vicinity of an occurence of the word than at far
        off positions

    """
    design_matrix = create_splines(length, nr_knots, remove_last_spline, lambda l,n: np.logspace(0.0, np.log10(l), n))
    design_matrix[1,0] = 1.0
    return design_matrix[1:,:]

def create_splines(length, nr_knots, remove_last_spline, fn_knots):
    """
        Generates B-spline basis functions based on the length and number of
        knots of the ongoing iteration.

        This functions augments (increases) the number of node by repeating the
        outermost knots 4 times each. This is so that the B-splines at the
        extremeties still have enough knots to span. 

        `fn_knots` is a function that computes the knot positions and accepts
        the length `l` and number of knots `n` and returns the knot positions
        (ie. spline borders).

        Alternatively `fn_knots` can also be a list or numpy array of knots.

    """
    
    knots = augknt(np.unique(fn_knots(length, nr_knots)) if callable(fn_knots) else fn_knots, 4)

    # This is the function that actually generates the B-splines given a
    # particular length and number of knots
    tau = range(int(length)+1)
    design_matrix = spcol(tau, knots, 4)#[:,1:-1] # TODO first and last column are always zero and are not present in Matlab... figure out what's going on
    return design_matrix[:,:-1] if remove_last_spline else design_matrix

# have a look at
# http://www.scipy.org/doc/api_docs/SciPy.signal.bsplines.html#bspline
# http://docs.scipy.org/doc/scipy/reference/interpolate.html
# http://www.scipy.org/Cookbook/Interpolation

# following code is copied from nwilming's ocupy/spline_base.py
# see https://github.com/nwilming

def augknt(knots,order):
    """Augment knot sequence such that some boundary conditions 
    are met."""
    a = []
    [a.append(knots[0]) for t in range(0,order)]
    [a.append(k) for k in knots]
    [a.append(knots[-1]) for t in range(0,order)]
    return np.array(a)     


def spcol(x,knots,spline_order):
    """Computes the spline colocation matrix for knots in x.
    
    The spline collocation matrix contains all m-p-1 bases 
    defined by knots. Specifically it contains the ith basis
    in the ith column.
    
    Input:
        x: vector to evaluate the bases on
        knots: vector of knots 
        spline_order: order of the spline
    Output:
        colmat: m x m-p matrix
            The colocation matrix has size m x m-p where m 
            denotes the number of points the basis is evaluated
            on and p is the spline order. The colums contain 
            the ith basis of knots evaluated on x.
    """
    columns = len(knots) - spline_order - 1
    colmat = np.nan*np.ones((len(x), columns))
    for i in range(columns):
        colmat[:,i] = spline(x, knots, spline_order, i)
    return colmat

def spline(x,knots,p,i=0.0):
    """Evaluates the ith spline basis given by knots on points in x"""
    assert(p+1<len(knots))
    return np.array([N(float(u),float(i),float(p),knots) for u in x])

def N(u,i,p,knots):
    """Compute Spline Basis
    
    Evaluates the spline basis of order p defined by knots 
    at knot i and point u.
    """
    if p == 0:
        if knots[int(i)] < u and u <=knots[int(i+1)]:
            return 1.0
        else:
            return 0.0
    else:
        try:
            k = (( float((u-knots[int(i)]))/float((knots[int(i+p)] - knots[int(i)]) )) 
                    * N(u,i,p-1,knots))
        except ZeroDivisionError:
            k = 0.0
        try:
            q = (( float((knots[int(i+p+1)] - u))/float((knots[int(i+p+1)] - knots[int(i+1)])))
                    * N(u,i+1,p-1,knots))
        except ZeroDivisionError:
            q  = 0.0 
        return float(k + q)

