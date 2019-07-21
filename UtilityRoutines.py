import pdb, sys, os, glob, pickle, time, re
import numpy as np
import scipy.ndimage
from bayes.pyhm_dev import pyhm
import numexpr



def rvFunc( t, a1, a2 ):
    """
    Within the context of the detector charge-trapping model,
    a reasonable physical constraint is a1<0 and a2>0. These
    restrictions are not applied within this routine, but can 
    be easily be built into whatever optimization metric is used 
    for determining the unknown parameters of the ramp model.
    """
    return 1 + a1*np.exp( -a2*t )

def r0Func( torb, rvt, a3, a4, a5 ):
    """
    Within the context of the detector charge-trapping model, 
    a reasonable physical constraint is a3<0, a4>0, and 
    a5<torb.min(). These restrictions are not applied within 
    this routine, but can be easily be built into whatever 
    optimization metric is used for determining the unknown 
    parameters of the ramp model.
    """
    return 1 + a3*np.exp( -( torb-a5 )/(a4*rvt) )

def DERampNoBase( t, torb, pars ):
    """
    Implementation of the double-exponential ramp model for WFC3 systematics.
    Taken from Eq 1-3 of de Wit et al (2018).
    """
    a1 = pars[0]
    a2 = pars[1]
    a3 = pars[2]
    a4 = pars[3]
    a5 = pars[4]
    rvt = rvFunc( t, a1, a2 )
    r0t = r0Func( torb, rvt, a3, a4, a5 )
    ttr = np.ones( t.size )
    return ttr, rvt*r0t


def DERampLinBase( t, torb, pars ):
    """
    Implementation of the double-exponential ramp model for WFC3 systematics.
    Taken from Eq 1-3 of de Wit et al (2018).
    """
    a1 = pars[0]
    a2 = pars[1]
    a3 = pars[2]
    a4 = pars[3]
    a5 = pars[4]
    b0 = pars[5]
    b1 = pars[6]
    rvt = rvFunc( t, a1, a2 )
    r0t = r0Func( torb, rvt, a3, a4, a5 )
    ttr = b0 + b1*t # linear-time baseline trend
    return ttr, rvt*r0t


def DERampQuadBase( t, torb, pars ):
    """
    Implementation of the double-exponential ramp model for WFC3 systematics.
    Taken from Eq 1-3 of de Wit et al (2018).
    """
    a1 = pars[0]
    a2 = pars[1]
    a3 = pars[2]
    a4 = pars[3]
    a5 = pars[4]
    b0 = pars[5]
    b1 = pars[6]
    b2 = pars[7]
    rvt = rvFunc( t, a1, a2 )
    r0t = r0Func( torb, rvt, a3, a4, a5  )
    ttr = b0 + b1*t + b2*(t**2.) # quadratic-time baseline trend
    return ttr, rvt*r0t


def Zap2D( ecounts2d, nsig_transient=8, nsig_static=10, niter=1 ):
    """
    Routine for identifying static and transient bad pixels in a 2d spectroscopic data cube.

    Inputs:
    ecounts2d - NxMxK data cube where N is cross-dispersion, M is dispersion, K is frame number.
    nsig_cull_transient - threshold for flagging transient bad pixels.
    nsig_cull_static - threshold for flagging static bad pixels.
    niter - number of iterations to be used

    Outputs:
    e2d_zapped - NxMxK cube containing the data with bad pixels corrected.
    transient_bad_pixs - NxMxK cube containing 1's for transient bad pixels and 0's otherwise
    static_bad_pixs - NxMxK cube containing 1's for static bad pixels and 0's otherwise
    e2d_medfilt - NxMxK cube containing nominal PSF for each frame made using median filter
    """
    print( '\nCleaning cosmic rays:' )
    # Initialise arrays to hold all the outputs:
    ndisp, ncross, nframes = np.shape( ecounts2d )
    e2d_zapped = np.zeros( [ ndisp, ncross, nframes ] )
    e2d_medfilt = np.zeros( [ ndisp, ncross, nframes ] )
    transient_bpixs = np.zeros( [ ndisp, ncross, nframes ] )
    static_bpixs = np.zeros( [ ndisp, ncross, nframes ] )
    ############ DIFFERENCE?
    # First apply a Gaussian filter to the pixel values
    # along the time axis of the data cube:
    e2d_smth = scipy.ndimage.filters.gaussian_filter1d( ecounts2d, sigma=5, axis=2 )
    e2d_smthsub = ecounts2d - e2d_smth # pixel deviations from smoothed time series
    ############ DOES THE ABOVE ACTUALLY HELP?
    med2d = np.median( e2d_smthsub, axis=2 )  # median deviation for each pixel
    stdv2d = np.std( e2d_smthsub, axis=2 ) # standard deviation in the deviations for each pixel
    # Loop over the data frames:
    for i in range( nframes ):
        e2d_zapped[:,:,i] = ecounts2d[:,:,i].copy()
        # Identify and replace transient bad pixels, possibly iterating more than once:
        for k in range( niter ):
            # Find the deviations of each pixel in the current frame in terms of 
            # number-of-sigma relative to the corresponding smoothed time series for
            # each pixel:
            e2d_smthsub = e2d_zapped[:,:,i] - e2d_smth[:,:,i]
            dsig_transient = np.abs( ( e2d_smthsub-med2d )/stdv2d )
            # Flag the outliers:
            ixs_transient = ( dsig_transient>nsig_transient )
            # Create a median-filter frame by taking the median of 5 pixels along the
            # cross-dispersion axis for each pixel, to be used as a nominal PSF:
            medfilt_ik = scipy.ndimage.filters.median_filter( e2d_zapped[:,:,i], size=[5,1] )
            # Interpolate any flagged pixels:
            e2d_zapped[:,:,i][ixs_transient] = medfilt_ik[ixs_transient]
            # Record the pixels that were flagged in the transient bad pixel map:
            transient_bpixs[:,:,i][ixs_transient] = 1
        ntransient = transient_bpixs[:,:,i].sum() # number of transient bad pixels for current frame
        ######## FLAGGING STATIC BAD PIXELS LIKE THIS SEEMS TO PRODUCE PROBLEMATIC
        ######## RESULTS SO I'M NULLING IT OUT FOR NOW BY SETTING THE NSIG REALLY HIGH:
        nsig_static = 1e9 # delete/change eventually...
        # Identify and replace static bad pixels, possibly iterating more than once:
        for k in range( niter ):
            # Create a median-filter frame by taking the median of 5 pixels along the
            # cross-dispersion axis for each pixel, to be used as a nominal PSF:
            medfilt_ik = scipy.ndimage.filters.median_filter( e2d_zapped[:,:,i], size=[5,1] )
            # Find the deviations of each pixel in the current frame in terms of 
            # number-of-sigma relative to the nominal PSF:
            dcounts_static = e2d_zapped[:,:,i] - medfilt_ik
            stdv_static = np.std( dcounts_static )
            dsig_static = np.abs( dcounts_static/stdv_static )
            # Flag the outliers:
            ixs_static = ( dsig_static>nsig_static )
            # Interpolate any flagged pixels:
            e2d_zapped[:,:,i][ixs_static] = medfilt_ik[ixs_static]
            # Record the pixels that were flagged in the static bad pixel map:
            static_bpixs[:,:,i][ixs_static] = 1
        nstatic = static_bpixs[:,:,i].sum() # number of transient bad pixels for current frame
        e2d_medfilt[:,:,i] = medfilt_ik # record the nominal PSF for the current frame
        print( '... frame {0} of {1}: ntransient={2}, nstatic={3}'.format( i+1, nframes, ntransient, nstatic ) )
    return e2d_zapped, transient_bpixs, static_bpixs, e2d_medfilt


def Zap1D( ecounts1d, nsig_transient=5, niter=2 ):
    # todo=adapt this routine; NOTE THAT I NOW PASS IN THE TRIMMED ECOUNTS1D ARRAY
    nframes, ndisp = np.shape( ecounts1d )
    x = np.arange( ndisp )
    bad_pixs = np.zeros( [ nframes, ndisp ] )
    y = np.median( ecounts1d[:-5,:], axis=0 )
    ecounts1d_zapped = ecounts1d.copy()
    
    for k in range( niter ):
        zk = ecounts1d_zapped.copy()
        # Create the normalised common-mode lightcurve:
        x0 = np.mean( zk, axis=1 )
        x0 /= x0[-1]
        # Remove the common-mode signal:
        for j in range( ndisp ):
            zk[:,j] /= x0
        # Update the master spectrum:
        #y0 = np.median( zk[-5:,:], axis=0 )
        y0 = np.median( zk, axis=0 )
        # Compute the relative variations for each individual 
        # spectrum relative to the master spectrum:
        for i in range( nframes ):
            zk[i,:] /= y0
        zkmed = np.median( zk, axis=0 )
        zksig = np.std( zk, axis=0 )
        dsig = np.zeros( [ nframes, ndisp ] )
        for i in range( nframes ):
            dsig[i,:] = np.abs( zk[i,:]-zkmed )/zksig
        cixs = dsig>nsig_transient
        bad_pixs[cixs] = 1
        medfilt = scipy.ndimage.filters.median_filter( ecounts1d_zapped, size=[5,1] )
        ecounts1d_zapped[cixs] = medfilt[cixs]
        print( 'Iter {0}: max(dsig)={1:.2f}'.format( k+1, dsig.max() ) )
    print( 'Zap1D flagged {0:.0f} bad pixels.\n'.format( bad_pixs.sum() ) )
    return ecounts1d_zapped, bad_pixs


def WFC3Nreads( hdu ):
    nreads = int( ( len( hdu )-1 )/5 )
    if nreads!=hdu[0].header['NSAMP']:
        nreads = -1 # must be a corrupt/incomplete file
    else:
        nreads -= 1 # exclude the zeroth read
    return nreads


def WFC3JthRead( hdu, nreads, j ):
    ix = 1+nreads*5-j*5
    read = hdu[ix].data
    sampt = hdu[ix].header['SAMPTIME']
    # Get as electrons:
    if hdu[1].header['BUNIT'].lower()=='electrons':
        ecounts = read
    elif hdu[1].header['BUNIT'].lower()=='electrons/s':
        ecounts = read*sampt
    else:
        pdb.set_trace()
    return ecounts

def SplitHSTOrbixs( thrs ):
    tmins = thrs*60.0
    n = len( tmins )
    ixs = np.arange( n )
    dtmins = np.diff( tmins )
    a = 1 + np.arange( n-1 )[dtmins>5*np.median( dtmins )]
    a = np.concatenate( [ [0], a, [n] ] )
    norb = len( a ) - 1
    orbixs = []
    for i in range( norb ):
        orbixs += [ np.arange( a[i], a[i + 1] ) ]
    return orbixs


def DERampOLD( thrs, torb, pars ):
    """
    Double-exponential ramp function from de Wit et al (2018),
    """
    a1 = pars[0]
    a2 = pars[1]
    a3 = pars[2]
    a4 = pars[3]
    a5 = pars[4]
    a6 = pars[5]
    a7 = pars[6]
    rvt = rvFunc( thrs, a1, a2 )
    r0t = r0Func( thrs, torb, rvt, a3, a4, a5 )
    lintrend = a6+a7*thrs
    return rvt*r0t*lintrend

def rvFuncOLD( thrs, a1, a2 ):
    return 1+a1*np.exp( -thrs/a2 )

def r0FuncOLD( thrs, torb, rvt, a3, a4, a5 ):
    return 1+a3*np.exp( -( torb-a5 )/(a4*rvt) )



def BTSettlBinDown( fpathu, fpathb ):
    d = np.loadtxt( fpathu )
    wav = d[:,0]*(1e-4) # convert A to micr
    flux = d[:,1] # leave flux per A for consistency
    # Restrict to wavelengths shorter than some threshold:
    wu = 20
    ixs = ( wav<wu )
    wav = wav[ixs]
    flux = flux[ixs]
    # Define the number of bins per micron:
    nbin_per_micron = 300
    nbins = wu*nbin_per_micron
    # Bin the high-res spectrum:
    wavb, fluxb, stdvs, npb = Bin1D( wav, flux, nbins=nbins )
    ixs = npb>0
    wav_A = wavb[ixs]*(1e4) # convert back to A for consistency
    flux = fluxb[ixs]
    # Save the binned spectrum:
    output = np.column_stack( [ wav_A, flux ] )
    np.savetxt( fpathb, output )
    print( 'Saved:\n{0}'.format( fpathb ) )
    return None


def Bin1D( x, y, nbins=10, shift_left=0.0, shift_right=0.0 ):
    """
    shift_left and shift_right are optional arguments that allow you to shift the
    bins either to the left or right by a specified fraction of a bin width.
    """

    x = x.flatten()
    y = y.flatten()
    if len( x )!=len( y ):
        raise ValueError( 'vector dimensions do not match' )
    
    binw = (x.max()-x.min())/float(nbins)
    if nbins>1:
        # Have half-bin overlap at start:
        wmin = x.min()-binw/2.-binw*shift_left+binw*shift_right
    elif nbins==1:
        wmin = x.min()
    else:
        pdb.set_trace()
    wmax = x.max()
    wrange = wmax - wmin
    if nbins>1:
        xbin = list(np.r_[wmin+binw/2.:wmax-binw/2.:nbins*1j])
    elif nbins==1:
        xbin = [wmin+0.5*wrange]
    else:
        pdb.set_trace()
    ybin = list(np.zeros(nbins))
    ybinstdvs = np.zeros(nbins)
    nperbin = np.zeros(nbins)
    for j in range(nbins):
        l = (abs(x - xbin[j]) <= binw/2.)
        if l.any():
            nperbin[j] = len(y[l])
            ybin[j] = np.mean(y[l])
            ybinstdvs[j] = np.std(y[l])
    return np.array(xbin), np.array(ybin), np.array(ybinstdvs), np.array(nperbin)



def GetStrs( prelim_fit, beta_free ):
    if prelim_fit==True:
        prelimstr = 'prelim'
    else:
        prelimstr = 'final'
    if beta_free==True:
        betastr = 'beta_free'
    else:
        betastr = 'beta_fixed'
    return prelimstr, betastr

def GetLDKey( ld ):
    if ld=='ldatlas_nonlin_fixed':
        ldkey = 'ld_nonlin_fixed'
    elif ld=='ldatlas_nonlin_free':
        ldkey = 'ld_nonlin_free'
    elif ld=='ldatlas_quad_free': 
        ldkey = 'ld_quad_free'
    elif ld=='ldatlas_quad_fixed': 
        ldkey = 'ld_quad_fixed'
    elif ld=='ldsing_nonlin_fixed': 
        ldkey = 'ldsing_nonlin_fixed'
    elif ld=='ldtk_free': 
        ldkey = 'ldtk_quad'
    elif ld=='ldsing_free': 
        ldkey = 'ldsing_quad'
    else:
        pdb.set_trace()
    return ldkey

def LinTrend( jd, tv, flux ):
    delt = jd-jd[0]
    orbixs = SplitHSTOrbixs( delt*24 )

    #t1 = tv[orbixs[0]][-1]
    #t2 = tv[orbixs[-1]][-1]
    #f1 = flux[orbixs[0]][-1]
    #f2 = flux[orbixs[-1]][-1]
    #t1 = np.mean( tv[orbixs[0]] )
    #t2 = np.mean( tv[orbixs[-1]] )
    #f1 = np.mean( flux[orbixs[0]] )
    #f2 = np.mean( flux[orbixs[-1]] )

    n1 = int( np.floor( 0.5*len( orbixs[0] ) ) )
    n2 = int( np.floor( 0.5*len( orbixs[-1] ) ) )
    t1 = np.mean( tv[orbixs[0][n1:]] )
    t2 = np.mean( tv[orbixs[-1][n2:]] )
    f1 = np.median( flux[orbixs[0][n1:]] )
    f2 = np.median( flux[orbixs[-1][n2:]] )
    
    v1 = [ 1, t1 ]
    v2 = [ 1, t2 ]
    A = np.row_stack( [ v1, v2 ] )
    c = np.reshape( [f1,f2], [2,1] )    
    z = np.linalg.lstsq( A, c, rcond=None )[0].flatten()
    
    return z

def MVNormalWhiteNoiseLogP( r, u, n  ):
    term1 = -np.sum( numexpr.evaluate( 'log( u )' ) )
    term2 = -0.5*np.sum( numexpr.evaluate( '( r/u )**2.' ) )
    return term1 + term2 - 0.5*n*np.log( 2*np.pi )

def NormalLogP( x, mu, sig  ):
    term1 = -0.5*np.log( 2*np.pi*( sig**2. ) )
    term2 = -0.5*( ( ( x-mu )/sig )**2. )
    return term1+term2

              

def GetVarKey( y ):
    if y=='hstphase':
        return 'hstphase', 'phi'
    if y=='wavshift':
        return 'wavshift_pix', 'wavshift'
    if y=='cdshift':
        return 'cdcs', 'cdshift'
    if y=='t':
        return 'tv', 't'

def GetWalkerState( mcmc ):
    keys = list( mcmc.model.free.keys() )
    walker_state = {}
    for k in keys:
        walker_state[k] = mcmc.walker_chain[k][-1,:]
    return walker_state


def GetGPStr( gpinputs ):
    gpstr = ''
    for k in gpinputs:
        gpstr += '{0}_'.format( k )
    gpstr = gpstr[:-1]
    if gpstr=='hstphase_t_wavshift_cdshift':
        gpstr = 'gp1111'
    elif gpstr=='hstphase_wavshift_cdshift':
        gpstr = 'gp1011'
    elif gpstr=='hstphase_wavshift':
        gpstr = 'gp1010'
    elif gpstr=='hstphase':
        gpstr = 'gp1000'
    elif gpstr=='hstphase_t':
        gpstr = 'gp1100'
    else:
        pdb.set_trace()
    return gpstr


def RefineMLE( walker_chain, mbundle ):
    """
    Takes a walker group chain and refines the MLE.
    """
    ix = np.argmax( walker_chain['logp'] )        
    ix = np.unravel_index( ix, walker_chain['logp'].shape )
    print( '\nRefining the best-fit solution...' )
    mp = pyhm.MAP( mbundle )
    for key in mp.model.free.keys():
        mp.model.free[key].value = walker_chain[key][ix]
    mp.fit( xtol=1e-4, ftol=1e-4, maxfun=10000, maxiter=10000 )
    print( 'Done.\nRefined MLE values:' )
    mle_refined = {}
    for key in mp.model.free.keys():
        mle_refined[key] = mp.model.free[key].value
        print( '{0} = {1}'.format( key, mp.model.free[key].value ) )
    return mle_refined
    
def DefineLogiLprior( z, vark, label, priortype='uniform' ):
    if priortype=='uniform':
        nrupp = 100
        zrange = z.max()-z.min()
        #dz = np.median( np.abs( np.diff( z ) ) )
        dz = np.min( np.abs( np.diff( z ) ) )
        #zlow = -10
        zlow = np.log( 1./( nrupp*zrange ) )
        if vark!='t':
            #zupp = 10
            zupp = np.log( 1/( dz ) )
        else: # prevent t having short correlation length scale
            zupp = np.log( 1./zrange )
        #zupp = np.log( 1/( dz ) )
        logiL_prior = pyhm.Uniform( label, lower=zlow, upper=zupp )
        #print( '\n', label )
        #print( 'zlow = ', zlow )
        #print( 'zupp = ', zupp )
        #print( '  dz = ', dz )
        #print( np.shape( z ) )
        #pdb.set_trace()
    elif priortype=='gamma':
        logiL_prior = pyhm.Gamma( label, alpha=1, beta=1 )
    #print( label, zlow, zupp )
    #print( z.max()-z.min() )
    #pdb.set_trace()
    return logiL_prior

def GetChainFromWalkers( walker_chains, nburn=0 ):
    ngroups = len( walker_chains )
    for i in range( ngroups ):
        walker_chain = walker_chains[i]
        keys = list( walker_chain.keys() )
        keys.remove( 'logp' )
        npar = len( keys )
    chain_dicts = []
    chain_arrs = []
    for i in range( ngroups ):
        chain_i = pyhm.collapse_walker_chain( walker_chains[i], nburn=nburn )
        try:
            chain_i['incl'] = np.rad2deg( np.arccos( chain_i['b']/chain_i['aRs'] ) )
        except:
            pass
        chain_dicts += [ chain_i ]
    grs = pyhm.gelman_rubin( chain_dicts, nburn=0, thin=1 )
    chain = pyhm.combine_chains( chain_dicts, nburn=nburn, thin=1 )        
    return chain, grs



def BestFitsEval( mle, evalmodels ):
    dsets = list( evalmodels.keys() )
    bestfits = {}
    batpars = {}
    pmodels = {}
    for k in dsets:
        scankeys = list( evalmodels[k].keys() )
        bestfits[k] = {}
        batpars[k] = {}
        pmodels[k] = {}
        for j in scankeys:
            z = evalmodels[k][j][0]( mle )
            bestfits[k][j] = z['arrays']
            batpars[k][j] = z['batpar']
            pmodels[k][j] = z['pmodel']
            #print( 'rrrr', z['arrays'].keys() )
            #pdb.set_trace()
    return bestfits, batpars, pmodels



def MultiColors():
    z = [ [31,120,180], \
          [166,206,227], \
          [178,223,138], \
          [51,160,44], \
          [251,154,153], \
          [227,26,28], \
          [253,191,111], \
          [255,127,0], \
          [202,178,214], \
          [106,61,154], \
          [177,89,40] ]
    rgbs = []
    for i in range( len( z ) ):
        rgbs += [ np.array( z[i] )/256. ]
    return rgbs

def NaturalSort( iterable, key=None, reverse=False):
    """
    Return a new naturally sorted list from the items in *iterable*.

    The returned list is in natural sort order. The string is ordered
    lexicographically (using the Unicode code point number to order individual
    characters), except that multi-digit numbers are ordered as a single
    character.

    Has two optional arguments which must be specified as keyword arguments.

    *key* specifies a function of one argument that is used to extract a
    comparison key from each list element: ``key=str.lower``.  The default value
    is ``None`` (compare the elements directly).

    *reverse* is a boolean value.  If set to ``True``, then the list elements are
    sorted as if each comparison were reversed.

    The :func:`natural_sorted` function is guaranteed to be stable. A sort is
    stable if it guarantees not to change the relative order of elements that
    compare equal --- this is helpful for sorting in multiple passes (for
    example, sort by department, then by salary grade).

    Taken from: 
    https://github.com/bdrung/snippets/blob/master/natural_sorted.py
    """
    prog = re.compile(r"(\d+)")

    def alphanum_key(element):
        """Split given key in list of strings and digits"""
        return [int(c) if c.isdigit() else c for c in prog.split(key(element)
                if key else element)]

    return sorted(iterable, key=alphanum_key, reverse=reverse)
    

def ScanVal( x ):
    if x=='f':
        return 1
    elif x=='b':
        return -1
    else:
        return None


    

def RefineMLEfromGroups( walker_chains, mbundle ):
    ngroups = len( walker_chains )
    # Identify which walker group hits the highest logp:
    logp = np.zeros( ngroups )
    for i in range( ngroups ):
        logp[i] = np.max( walker_chains[i]['logp'] )
    ix = np.argmax( logp )
    # Restrict to this walker group:
    return RefineMLE( walker_chains[ix], mbundle )


def GetInitWalkers( mcmc, nwalkers, init_par_ranges ):
    init_walkers = {}
    for key in list( mcmc.model.free.keys() ):
        init_walkers[key] = np.zeros( nwalkers )
    for i in range( nwalkers ):
        for key in mcmc.model.free.keys():
            startpos_ok = False
            counter = 0
            while startpos_ok==False:
                startpos = init_par_ranges[key].random()
                mcmc.model.free[key].value = startpos
                if np.isfinite( mcmc.model.free[key].logp() )==True:
                    startpos_ok = True
                else:
                    counter += 1
                if counter>100:
                    print( '\n\nTrouble initialising walkers!\n\n' )
                    for key in mcmc.model.free.keys():
                        print( key, mcmc.model.free[key].value, \
                               mcmc.model.free[key].parents, \
                               mcmc.model.free[key].logp() )
                    pdb.set_trace()
            init_walkers[key][i] = startpos
    return init_walkers

