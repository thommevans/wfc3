import numpy as np
import pdb, sys, os
import matplotlib.pyplot as plt
from bayes.gps_dev.gps import gp_class, kernels



def custom_kernel_sqexp_invL_ard( x, y, **cpars ):
    cpars1 = { 'amp':cpars['amp_baset'], 'iscale':cpars['iscale_baset'] }
    cpars2 = { 'amp':cpars['amp'], 'iscale':cpars['iscale'] }
    xt = np.column_stack( x[:,0] ).T
    if y is not None:
        yt = np.column_stack( y[:,0] ).T
        yv = y[:,1:]
    else:
        yt = yv = None
    cov1 = kernels.sqexp_invL( xt, yt, **cpars1 )
    cov2 = kernels.sqexp_invL_ard( x[:,1:], yv, **cpars2 )
    return cov1+cov2

def custom_kernel_mat32_invL_ard( x, y, **cpars ):
    cpars1 = { 'amp':cpars['amp_baset'], 'iscale':cpars['iscale_baset'] }
    cpars2 = { 'amp':cpars['amp'], 'iscale':cpars['iscale'] }
    xt = np.column_stack( x[:,0] ).T
    if y is not None:
        yt = np.column_stack( y[:,0] ).T
        yv = y[:,1:]
    else:
        yt = yv = None
    cov1 = kernels.sqexp_invL( xt, yt, **cpars1 ) # always sqexp for t-baseline
    cov2 = kernels.matern32_invL_ard( x[:,1:], yv, **cpars2 )
    return cov1+cov2

