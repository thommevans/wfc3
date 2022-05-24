import pdb, sys, os, glob, pickle, time, re
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from bayes.pyhm_dev import pyhm
import numexpr
import pysynphot



def loadStellarModel( Teff, MH, logg, stellarModel='k93models' ):
    sp = pysynphot.Icat( stellarModel, Teff, MH, logg )
    wavA = sp.wave # A
    flam = sp.flux # erg s^-1 cm^-2 A^-1
    #flam = flam*wavA
    sp.convert( 'photlam' )
    photlam = sp.flux
    c = pysynphot.units.C
    h = pysynphot.units.H
    ePhot = h*c/wavA
    myPhotlam = flam/ePhot
    wavMicr = wavA*(1e-10)*(1e6)
    ixs = ( wavMicr>0.2 )*( wavMicr<6 )
    wavMicr = wavMicr[ixs]
    flam = flam[ixs]
    photlam = photlam[ixs]
    #photlam = myPhotlam[ixs]
    return wavMicr, flam, photlam

def checkStellarModel( e1d, bp, Teff, MH, logg, trim_disp_ixs, \
                       wavsol_dispbound_ixs, stellarModel='k93models' ):
    """
    Routine for quickly trialing Teff, MH, logg values to 
    cross-correlate with model stellar spectrum. Adapted
    from GetWavSol() in ClassDefs.py.

    Note:
      bp = ClassDefs.Bandpass()
      bp.config = 'G141'
      bp.fpath = '/path/to/bandpass/file'
      bp.Read()

    """
    #d1, d2 = trim_box[1] # dispersion limits
    d1, d2 = trim_disp_ixs # dispersion limits
    dwav_max = 0.3 # in micron
    nshifts = int( np.round( 2*dwav_max*(1e4)+1 ) ) # 0.0001 micron = 0.1 nm
    ndisp = len( e1d )
    A2micron = 1e-4
    ndisp = e1d.size
    wbp = bp.bandpass_wavmicr
    ybp = bp.bandpass_thput
    dwbp = np.median( np.diff( wbp ) )
    wstar, flam, photlam = loadStellarModel( Teff, MH, logg, stellarModel=stellarModel )
    ystar = flam # still unsure why this works better than photlam...
    #ystar = photlam
    # Interpolate the stellar model onto the transmission wavelength grid:
    ixs = ( wstar>wbp[0]-0.1 )*( wstar<wbp[-1]+0.1 )
    ystar_interp = np.interp( wbp, wstar[ixs], ystar[ixs] )
    # Modulate the interpolated stellar model by the throughput to 
    # simulate a measured spectrum:
    ystar = ystar_interp*ybp
    ystar /= ystar.max()
    wstar = wbp
    dwstar = np.median( np.diff( wstar ) )
    ix = np.argmax( ystar )
    w0 = wstar[ix]
    x = np.arange( ndisp )
    ix = np.argmax( e1d )
    delx = x-x[ix]
    wavsol0 = w0 + bp.dispersion_micrppix*delx
    #x0 = np.arange( wavsol0.size )
    # Smooth the stellar flux and model spectrum, because we use
    # the sharp edges of the throughput curve to calibrate the 
    # wavelength solution:
    fwhm_e1d = 4. # stdv of smoothing kernel in dispersion pixels
    sig_e1d = fwhm_e1d/2./np.sqrt( 2.*np.log( 2 ) )
    e1d_smth = scipy.ndimage.filters.gaussian_filter1d( e1d, sig_e1d )
    sig_star = (sig_e1d*bp.dispersion_micrppix)/dwstar
    ystar_smth = scipy.ndimage.filters.gaussian_filter1d( ystar, sig_star )
    e1d_smth /= e1d_smth.max()
    ystar_smth /= ystar_smth.max()
    ix0, ix1 = wavsol_dispbound_ixs
    cc = CrossCorrSol( wavsol0, e1d_smth, wstar, ystar_smth, \
                       ix0, ix1, dx_max=dwav_max, nshifts=nshifts )
    wshift = cc[0]
    vstretch = cc[1]
    wavmicr0 = wavsol0-wshift
    nl = np.arange( d1 )[::-1]
    nr = np.arange( ndisp-d2-1 )
    extl = wavmicr0[0]-(nl+1)*bp.dispersion_micrppix
    extr = wavmicr0[-1]+(nr+1)*bp.dispersion_micrppix
    wavmicr = np.concatenate( [ extl, wavmicr0, extr ] )
    # Plot for checking the spectrum and wavelength solution:
    plt.figure( figsize=[12,8] )
    #specname = os.path.basename( self.btsettl_fpath )
    titlestr = 'Teff={0:.0f}K, [M/H]={1:.2f}, logg={2:.2f}'.format( Teff, MH, logg )
    plt.title( titlestr, fontsize=20 )
    plt.plot( wbp, ybp/ybp.max(), '-g', \
              label='{0} bandpass'.format( bp.config ) )
    plt.plot( wavmicr0, e1d/e1d.max(), '-m', lw=2, \
              label='cross-correlation' )
    plt.plot( wstar, ystar_interp/ystar_interp.max(), '-r', \
              label='stellar flux' )
    plt.plot( wstar, ystar, '--c', lw=2, label='model spectrum' )
    ixs = ( ybp>(1e-3)*ybp.max() )
    plt.xlim( [ wbp[ixs].min(), wbp[ixs].max() ] ) 
    plt.ylim( [ -0.1, 1.4 ] )
    plt.legend( loc='upper left', ncol=2, fontsize=16 )
    plt.xlabel( 'Wavelength (micron)', fontsize=18 )
    plt.ylabel( 'Relative Flux/Throughput', fontsize=18 )
    return None

def residsRMSVsBinSize( thrs, resids ):
    """
    Given an array of residuals, computes the rms as a function
    of bin size. For checking how the noise bins down compared
    to white noise expectations.
    """
    ndat = len( resids )
    nbinsMin = 6 # minimum number of binned points for which to compute RMS
    nptsMax = int( np.floor( ndat/float( nbinsMin ) ) )
    nptsPerBin = 1+np.arange( nptsMax )
    nbinSizes = len( nptsPerBin )
    oixs = SplitHSTOrbixs( thrs )
    norb = len( oixs )
    x0 = np.arange( ndat )
    rms = np.zeros( nbinSizes )
    dthrs = np.median( np.diff( thrs ) )
    tmin = np.zeros( nbinSizes )
    #import matplotlib.pyplot as plt
    #plt.close('all')
    #plt.ion()
    #plt.figure()
    #plt.plot( thrs, resids, 'ok' )
    
    for i in range( nbinSizes ):
        npts_i = nptsPerBin[i]
        tmin[i] = npts_i*dthrs*60
        finished = False
        residsb = []
        tbin = []
        for j in range( norb ):
            ixs = oixs[j]
            xj = x0[ixs]
            nj = len( xj )
            ixl = np.arange( 0, nj, npts_i )
            ixu = ixl+npts_i#np.arange( npts_i, nj+1 )
            nbj = len( ixl )
            for k in range( nbj ):
                residsb_jk = resids[ixs][ixl[k]:ixu[k]]
                if len( residsb_jk )!=npts_i:
                    residsb_jk = resids[ixs][nj-npts_i:nj]
                    #print( len( residsb_jk ), npts_i )
                    #pdb.set_trace()
                #print( 'aaaaa', j, len( residsb_jk ), npts_i )
                residsb += [ np.mean( residsb_jk ) ]
                #print( '\naaaa', j, k, ixl[k], ixu[k] )
                #print( 'bbbbb', residsb_jk, resids[ixs][ixl[k]] )
                #if npts_i>11:
                #    plt.plot( thrs[ixs][ixl[k]:ixu[k]], residsb_jk, '-x' )
                #    print( thrs[ixs][ixl[k]:ixu[k]], residsb_jk, residsb[-1] )
                #    pdb.set_trace()
        residsb = np.array( residsb )
        #pdb.set_trace()
        #while finished==False:
        #nbins_i = int( np.floor( ndat/float( npts_i ) ) )
        #residsb = np.zeros( nbins_i )
        #for j in range( nbins_i ):
        #    ix1 = j*npts_i
        #    ix2 = (j+1)*npts_i+1
        #    residsb[j] = np.mean( resids[ix1:ix2] )
        rms[i] = np.sqrt( np.mean( residsb**2. ) )
    #print( tmin )
    #pdb.set_trace()
    return nptsPerBin, tmin, rms
    

def residsRMSVsBinSizeBasic( resids ):
    """
    Given an array of residuals, computes the rms as a function
    of bin size. For checking how the noise bins down compared
    to white noise expectations.
    """
    ndat = len( resids )
    nbinsMin = 6 # minimum number of binned points for which to compute RMS
    nptsMax = int( np.floor( ndat/float( nbinsMin ) ) )
    nptsPerBin = 1+np.arange( nptsMax )
    nbinSizes = len( nptsPerBin )
    
    rms = np.zeros( nbinSizes )
    for i in range( nbinSizes ):
        npts_i = nptsPerBin[i]
        nbins_i = int( np.floor( ndat/float( npts_i ) ) )
        residsb = np.zeros( nbins_i )
        for j in range( nbins_i ):
            ix1 = j*npts_i
            ix2 = (j+1)*npts_i+1
            residsb[j] = np.mean( resids[ix1:ix2] )
        rms[i] = np.sqrt( np.mean( residsb**2. ) )
    #pdb.set_trace()
    return nptsPerBin, rms
    

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
    bline = np.ones( t.size )
    return bline, rvt*r0t


def DERampLinBase( bvar, t, torb, pars ):
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
    bline = b0 + b1*bvar # linear-time baseline trend
    return bline, rvt*r0t


def DERampQuadBase( bvar, t, torb, pars ):
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
    bline = b0 + b1*bvar + b2*(bvar**2.) # quadratic-time baseline trend
    return bline, rvt*r0t


def DERampExpBase( bvar, t, torb, pars ):
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
    bline = b0 + b1*np.exp( -b2*bvar ) # exponential-time baseline trend
    return bline, rvt*r0t


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
    elif ld=='ldatlas_linear_fixed':
        ldkey = 'ld_linear_fixed'
    elif ld=='ldatlas_linear_free':
        ldkey = 'ld_linear_free'
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
    if y=='loghstphase':
        return 'hstphase', 'logphi'
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


def MaxLogLikePoint( walker_chain, mbundle ):
    ixs0 = np.isfinite( walker_chain['logp'] )
    ix = np.argmax( walker_chain['logp'][ixs0] )        
    ix = np.unravel_index( ix, walker_chain['logp'][ixs0].shape )
    print( '\nLocating maximum likelihood values...' )
    parVals = {}
    mp = pyhm.MAP( mbundle )
    for key in mp.model.free.keys():
        parVals[key] = walker_chain[key][ixs0][ix]
    logLikeMax = walker_chain['logp'][ixs0][ix]
    return parVals, logLikeMax

def RefineMLE( walker_chain, mbundle ):
    """
    Takes a walker group chain and refines the MLE.
    """
    #ixs0 = np.isfinite( walker_chain['logp'] )
    #ix = np.argmax( walker_chain['logp'][ixs0] )        
    #ix = np.unravel_index( ix, walker_chain['logp'][ixs0].shape )
    parVals, logLikeMax = MaxLogLikePoint( walker_chain, mbundle )
    print( '\nRefining the best-fit solution...' )
    mp = pyhm.MAP( mbundle )
    for key in mp.model.free.keys():
        #mp.model.free[key].value = walker_chain[key][ixs0][ix]
        mp.model.free[key].value = parVals[key]
    mp.fit( xtol=1e-4, ftol=1e-4, maxfun=10000, maxiter=10000 )
    print( 'Done.\nRefined MLE values:' )
    mle_refined = {}
    for key in mp.model.free.keys():
        mle_refined[key] = mp.model.free[key].value
        print( '{0} = {1}'.format( key, mp.model.free[key].value ) )
    return mle_refined


def RefineMLE_PREVIOUS( walker_chain, mbundle ):
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
        dzarr = np.abs( np.diff( z ) )
        dz = np.min( dzarr[dzarr>0] )
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


def MaxLogLikefromGroups( walker_chains, mbundle ):
    ngroups = len( walker_chains )
    # Identify which walker group hits the highest logp:
    parVals = []
    logp = np.zeros( ngroups )
    for i in range( ngroups ):
        z = MaxLogLikePoint( walker_chains[i], mbundle )
        parVals += [ z[0] ]
        logp[i] = z[1]
    ix = np.argmax( logp )
    # Restrict to this walker group:
    #return RefineMLE( walker_chains[ix], mbundle )
    return parVals[ix]


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


#def CrossCorrSol( self, x0, ymeas, xtarg, ytarg, ix0, ix1, dx_max=1, nshifts=1000 )
def CrossCorrSol( x0, ymeas, xtarg, ytarg, ix0, ix1, dx_max=1, nshifts=1000 ):
    """
    The mapping is: [ x0-shift, ymeas ] <--> [ xtarg, ytarg ]
    [ix0,ix1] are the indices defining where to compute residuals along dispersion axis.
    """
    dw = np.median( np.diff( xtarg ) )
    wlow = x0.min()-dx_max-dw
    wupp = x0.max()+dx_max+dw
    # Extend the target array at both edges:
    dwlow = np.max( [ xtarg.min()-wlow, 0 ] )
    dwupp = np.max( [ wupp-xtarg.max(), 0 ] )
    wbuff_lhs = np.r_[ xtarg.min()-dwlow:xtarg.min():dw ]
    wbuff_rhs = np.r_[ xtarg.max()+dw:xtarg.max()+dwupp:dw ]
    xtarg_ext = np.concatenate( [ wbuff_lhs, xtarg, wbuff_rhs ] )
    fbuff_lhs = np.zeros( len( wbuff_lhs ) )
    fbuff_rhs = np.zeros( len( wbuff_rhs ) )
    ytarg_ext = np.concatenate( [ fbuff_lhs, ytarg, fbuff_rhs ] )
    # Interpolate the extended target array:
    interpf = scipy.interpolate.interp1d( xtarg_ext, ytarg_ext )
    shifts = np.linspace( -dx_max, dx_max, nshifts )
    vstretches = np.zeros( nshifts )
    rms = np.zeros( nshifts )
    # Loop over the wavelength shifts, where for each shift we move
    # the target array and compare it to the measured array:
    A = np.ones( [ ymeas.size, 2 ] )
    b = np.reshape( ymeas/ymeas.max(), [ ymeas.size, 1 ] )
    ss_fits = []
    diffsarr = []
    for i in range( nshifts ):
        # Assuming the default x-solution is x0, shift the model
        # array by dx. If this provides a good match to the data,
        # it means that the default x-solution x0 is off by dx.
        ytarg_shifted_i = interpf( x0 - shifts[i] )
        A[:,1] = ytarg_shifted_i/ytarg_shifted_i.max()
        res = np.linalg.lstsq( A, b, rcond=None )
        c = res[0].flatten()
        vstretches[i] = c[1]
        fit = np.dot( A, c )
        diffs = b.flatten() - fit.flatten()
        rms[i] = np.mean( diffs[ix0:ix1+1]**2. )
        ss_fits +=[ fit.flatten() ]
        diffsarr += [ diffs ]
    ss_fits = np.row_stack( ss_fits )
    diffsarr = np.row_stack( diffsarr )
    rms -= rms.min()
    # Because the rms versus shift is well-approximated as parabolic,
    # refine the shift corresponding to the minimum rms by fitting
    # a parabola to the shifts evaluated above:
    offset = np.ones( nshifts )
    phi = np.column_stack( [ offset, shifts, shifts**2. ] )
    nquad = min( [ nshifts, 15 ] )
    ixmax = np.arange( nshifts )[np.argsort( rms )][nquad]
    ixs = rms<rms[ixmax]
    coeffs = np.linalg.lstsq( phi[ixs,:], rms[ixs], rcond=None )[0]
    nshiftsf = 100*nshifts
    offsetf = np.ones( nshiftsf )
    shiftsf = np.linspace( shifts.min(), shifts.max(), nshiftsf )
    phif = np.column_stack( [ offsetf, shiftsf, shiftsf**2. ] )
    rmsf = np.dot( phif, coeffs )
    vstretchesf = np.interp( shiftsf, shifts, vstretches )
    ixf = np.argmin( rmsf )
    ix = np.argmin( rms )
    ix0 = ( shifts==0 )
    diffs0 = diffsarr[ix0,:].flatten()
    return shiftsf[ixf], vstretchesf[ixf], ss_fits[ix,:], diffsarr[ix,:], diffs0


def PrepRampPars( datasets, data, data_ixs, scankeys, baseline, \
                  rampScanShare ):
    # For each scan direction, the systematics model consists of a
    # double-exponential ramp (a1,a2,a3,a4,a5):
    rlabels0 = [ 'a1', 'a2', 'a3', 'a4', 'a5' ]
    # Initial values for systematics parameters:
    rlabels = []
    rfixed = []
    rinit = []
    rixs = {}
    fluxc = {}
    c = 0 # counter
    ndsets = len( datasets )
    # fluxc is split by dataset, not scan direction; however, it isn't
    # actually used for specLC fits as a good estimate is already
    # available for the psignal from the whiteLC fit:
    for k in range( ndsets ):
        rparsk, fluxck = PrelimRPars( datasets[k], data, data_ixs, scankeys, \
                                      baseline, rampScanShare )
        fluxc[datasets[k]] = fluxck
        #print( fluxck.keys() )
        #pdb.set_trace()
        rlabels += [ rparsk['rlabels'] ]
        rfixed = np.concatenate( [ rfixed, rparsk['rfixed'] ] )
        rinit = np.concatenate( [ rinit, rparsk['rpars_init'] ] )
        for i in list( rparsk['rixs'].keys() ):
            rixs[i] = rparsk['rixs'][i]+c
        c += len( rparsk['rlabels'] )
    rlabels = np.concatenate( rlabels ).flatten()
    r = { 'labels':rlabels, 'fixed':rfixed, 'pars_init':rinit, 'ixs':rixs }
    return r, fluxc
    
def PrelimRPars( dataset, data, data_ixs, scankeys, baseline, rampScanShare ):
    """
    """
    if len( scankeys[dataset] )>1:
        if rampScanShare==True:
            r, fluxc = PrelimRParsScanShared( dataset, data, data_ixs, \
                                              scankeys, baseline )
        else:
            r, fluxc = PrelimRParsScanSeparate( dataset, data, data_ixs, \
                                                scankeys, baseline )
    else:
        r, fluxc = PrelimRParsScanSeparate( dataset, data, data_ixs, \
                                            scankeys, baseline )
    if 0: # DELETE
        plt.figure()
        for k in scankeys[dataset]:
            idkey = '{0}{1}'.format( dataset, k )
            plt.plot( data[:,1][data_ixs[dataset][k]], fluxc[k], 'o' )
        pdb.set_trace()
    return r, fluxc

def PrelimRParsScanShared( dataset, data, data_ixs, scankeys, baseline ):

    ixsd = data_ixs
    thrs = data[:,1]
    torb = data[:,2]
    dwav = data[:,3]
    bvar = thrs # TODO allow this to be another variable        
    flux = data[:,4]

    # Must loop over scan directions to get fluxc right
    # for each scan direction:
    rpars0 = {}
    fluxFit = flux*np.ones_like( flux )
    for k in scankeys[dataset]:
        ixsk = ixsd[dataset][k] # data ixs for current dataset + scan direction
        fluxFit[ixsk] = fluxFit[ixsk]/np.median( fluxFit[ixsk][-3:] )
    # Run a quick double-exponential ramp fit on the first and last HST
    # orbits to get reasonable starting values for the parameters:
    rpars0, bfit, rfit = PrelimDEFit( dataset, bvar, thrs, \
                                      torb, fluxFit, baseline )
    # Note that the above ramp fit 'rfit' will perform fit with self.baseline, 
    # then return only the ramp parameters, but fluxc will be the flux
    # corrected by model=ramp*baseline, which is then used for a
    # preliminary planet signal fit.
    
    # For dataset, one set of ramp parameters for both scan directions:
    rlabels = [ 'a1_{0}'.format( dataset ), 'a2_{0}'.format( dataset ), \
                'a3_{0}'.format( dataset ), 'a4_{0}'.format( dataset ), \
                'a5_{0}'.format( dataset ) ]
    rlabels = np.array( rlabels, dtype=str )

    # The ramp parameter ixs are split by scan direction, however:
    nrpar = len( rpars0 )
    rfixed = np.zeros( nrpar ) # all parameters free
    rixs = {}
    fluxc = {}
    for k in scankeys[dataset]:
        idkey = '{0}{1}'.format( dataset, k )
        rixs[idkey] = np.arange( 0, 0+nrpar )
        ixsk = ixsd[dataset][k]
        #fluxc[idkey] = flux[ixsk]/( bfit[ixsk]*rfit[ixsk] )
        fluxc[k] = flux[ixsk]/rfit[ixsk] # only remove ramp; preserve offset
    r = { 'rlabels':rlabels, 'rfixed':rfixed, 'rpars_init':rpars0, 'rixs':rixs }
    return r, fluxc


def PrelimRParsScanSeparate( dataset, data, data_ixs, scankeys, baseline ):
    rlabels = []    
    rfixed = []
    rinit = []
    rixs = {}
    fluxc = {}
    c = 0 # counter
    ixsd = data_ixs
    for k in scankeys[dset]:
        ixsdk = ixsd[dset][k] # data ixs for current dataset + scan direction
        thrsdk = data[:,1][ixsdk]
        torbdk = data[:,2][ixsdk]
        dwavdk = data[:,3][ixsdk]
        bvardk = thrsdk # TODO allow this to be another variable
        fluxdk = data[:,4][ixsdk]
        idkey = '{0}{1}'.format( dset, k )
        # Run a quick double-exponential ramp fit on the first
        # and last HST orbits to get reasonable starting values
        # for the parameters:
        rpars0k, bfitk, rfitk = PrelimDEFit( dset, bvardk, thrsdk, torbdk, \
                                             fluxdk, baseline )
        rinit = np.concatenate( [ rinit, rpars0 ] )
        nrpar = len( rpars0 )
        rixs[idkey] = np.arange( c*nrpar, (c+1)*nrpar )
        fluxc[idkey] = fluxcik # TODO = fix this in future...
        rfixed = np.concatenate( [ rfixed, np.zeros( nrpar ) ] )
        rlabels_ik = []
        for j in range( nrpar ):
            rlabels_ik += [ '{0}_{1}{2}'.format( rlabels0[j], dset, k ) ]
        rlabels += [ np.array( rlabels_ik, dtype=str ) ]
        c += 1
    r = { 'rlabels':rlabels, 'rfixed':rfixed, 'rpars_init':rinit, 'rixs':rixs }
    # NOTE: This hasn't been tested in current format (2020-Nov-10th).
    return r, fluxc

    
def PrelimDEFit( dset, bvar, thrs, torb, flux, baseline ):
    """
    Performs preliminary fit for the ramp systematics, only
    fitting to the first and last HST orbits.
    """
    print( '\nRunning preliminary DE ramp fit for {0}'.format( dset ) )
    print( '(using only the first and last orbits)' )
    if ( baseline=='linearT' )+( baseline=='linearX' ):
        rfunc = DERampLinBase
        nbase = 2
    elif baseline=='quadratic':
        rfunc = DERampQuadBase
        nbase = 3
    elif baseline=='exponential':
        rfunc = DERampExpBase
        nbase = 3
    else:
        pdb.set_trace()
    orbixs = SplitHSTOrbixs( thrs )
    ixs = np.concatenate( [ orbixs[0], orbixs[-1] ] )
    def CalcRMS( pars ):
        baseline, ramp = rfunc( bvar[ixs], thrs[ixs], torb[ixs], pars )
        resids = flux[ixs]-baseline*ramp
        rms = np.sqrt( np.mean( resids**2. ) )
        return rms
    ntrials = 30
    rms = np.zeros( ntrials )
    pfit = []
    for i in range( ntrials ):
        print( '... trial {0:.0f} of {1:.0f}'.format( i+1, ntrials ) )
        b0i = flux[-1]
        #b0i = np.median( flux )
        b1i = 0
        # These starting values seem to produce reasonable results:
        a1b = 1e-3
        a2i = 1
        a3b = 1e-3
        a4i = 0.01
        a5i = 0.001
        bb = 0.1
        pinit = [ a1b*np.random.randn(), a2i*( 1+bb*np.random.randn() ), \
                  a3b*np.random.randn(), a4i*( 1+bb*np.random.randn() ), \
                  a5i*( 1+bb*np.random.randn() ), b0i, b1i ]
        #pinit = [ (1e-3)*np.random.randn(), 0.1+0.005*np.random.random(), \
        #          (1e-3)*np.random.randn(), 0.1+0.005*np.random.random(), \
        #          (1.+0.005*np.random.random() )/60., flux[-1], 0 ]
        if nbase==3:
            pinit += [ 0 ]
        pfiti = scipy.optimize.fmin( CalcRMS, pinit, maxiter=1e4, xtol=1e-3, \
                                     ftol=1e-4, disp=False )
        rms[i] = CalcRMS( pfiti )
        pfit += [ pfiti ]
    pbest = pfit[np.argmin(rms)]            
    a1, a2, a3, a4, a5 = pbest[:-nbase]
    rpars = [ a1, a2, a3, a4, a5 ]
    bfit, rfit = rfunc( bvar, thrs, torb, pbest )
    #fluxc = flux/( tfit*rfit )
    #fluxc = flux/rfit
    if 0:
        plt.figure()
        plt.plot( thrs, flux, 'ok' )
        plt.plot( thrs, bfit*rfit, '-r' )
        #pdb.set_trace()
    return rpars, bfit, rfit


def PrelimBPars( self, dataset ):
    """
    """
    if len( self.scankeys[dataset] )>1:
        if self.baselineScanShare==True:
            b = self.PrelimBParsScanShared( dataset )
        else:
            b = self.PrelimBParsScanSeparate( dataset )
    else:
        b = self.PrelimBParsScanSeparate( dataset )
    return b

def InitialBPars( baseline ):
    """
    Returns clean starting arrays for baseline trend arrays.
    """
    if ( baseline=='linearT' )+( baseline=='linearX' ):
        binit0 = [ 1, 0 ]
        bfixed0 = [ 0, 0 ]
        blabels0 = [ 'b0', 'b1' ]
    elif baseline=='quadratic':
        binit0 = [ 1, 0, 0 ]
        bfixed0 = [ 0, 0, 0 ]
        blabels0 = [ 'b0', 'b1', 'b2' ]
    elif baseline=='expDecayT':
        binit0 = [ 1, 0, 0 ]
        bfixed0 = [ 0, 0, 0 ]
        blabels0 = [ 'b0', 'b1', 'b2' ]
    return blabels0, binit0, bfixed0


def PrelimBParsScanSeparate( slcs, dataset, data, data_ixs, scankeys, baseline ):
    blabels0, binit0, bfixed0 = InitialBPars( baseline )
    nbpar = len( binit0 )
    ixs = np.arange( slcs[dataset]['jd'].size )
    bpars_init = []
    bfixed = []
    blabels = []
    bixs = {}
    c = 0 # counter
    if ( baseline=='linearT' ):
        bvar = data[:,1]
    elif ( baseline=='linearX' ):
        bvar = data[:,3]
    for k in scankeys[dataset]:
        idkey = '{0}{1}'.format( dataset, k )
        ixsk = data_ixs[dataset][k]
        bvark = bvar[ixsk]
        fluxk = data[:,4][ixsk]#[scanixs[k]]
        orbixs = UR.SplitHSTOrbixs( bvark )
        t1 = np.median( bvark[0] )
        t2 = np.median( bvark[-1] )
        f1 = np.median( fluxk[0] )
        f2 = np.median( fluxk[-1] )
        grad = ( f2-f1 )/( t2-t1 )
        offs = f1-grad*t1
        if ( baseline=='linearT' )+( baseline=='linearX' ):
            binitk = [ offs, grad ]
        elif baseline=='quadratic':
            binitk = [ offs, grad, 0 ]
        elif baseline=='exponential':
            binitk = [ offs, 0, 0 ]
        else:
            pdb.set_trace()
        bpars_init = np.concatenate( [ bpars_init, binitk ] )
        bfixed = np.concatenate( [ bfixed, bfixed0 ] )
        blabels_k = []
        for j in range( nbpar ):
            blabels_k += [ '{0}_{1}'.format( blabels0[j], idkey ) ]
        blabels += [ np.array( blabels_k, dtype=str ) ]
        bixs[idkey] = np.arange( c, c+nbpar )
        c += nbpar
    blabels = np.concatenate( blabels )
    b = { 'blabels':blabels, 'bfixed':bfixed, 'bpars_init':bpars_init, 'bixs':bixs }
    return b


def PrelimBParsScanShared( slcs, dataset, data, data_ixs, scankeys, baseline ):
    blabels0, binit0, bfixed0 = InitialBPars( baseline )
    nbpar = len( binit0 )
    ixs = np.arange( slcs[dataset]['jd'].size )

    # Forward scan is the reference:
    k = 'f'
    ixsf = data_ixs[dataset]['f']
    if ( baseline=='linearT' ):
        bvarf = data[:,1][ixsf]
    elif ( baseline=='linearX' ):
        bvarf = data[:,3][ixsf]
    fluxf = data[:,4][ixsf]
    orbixs = UR.SplitHSTOrbixs( bvarf )
    t1f = np.median( bvarf[0] )
    t2f = np.median( bvarf[-1] )
    f1f = np.median( fluxf[0] )
    f2f = np.median( fluxf[-1] )
    grad = ( f2f-f1f )/( t2f-t1f )
    offsf = f1f-grad*t1f
    # Estimate offset of backward-scan flux:
    ixsb = data_ixs[dataset]['b']
    fluxb = data[:,4][ixsb]
    f2b = np.median( fluxb[-1] )
    offsb = f2f/f2b

    if ( baseline=='linearT' )+( baseline=='linearX' ):
        binit = np.array( [ offsf, grad, offsb ] )
        bfixed = np.array( [ 0, 0, 0 ] )
        blabels = [ '{0}_{1}f'.format( blabels0[0], dataset ), \
                    '{0}_{1}'.format( blabels0[1], dataset ), \
                    '{0}_{1}b'.format( blabels0[0], dataset ) ]
    elif baseline=='quadratic':
        binit = np.array( [ offsf, grad, 0, offsb ] )
        bfixed = np.array( [ 0, 0, 0, 0 ] )
        blabels = [ '{0}_{1}f'.format( blabels0[0], dataset ), \
                    '{0}_{1}'.format( blabels0[1], dataset ), \
                    '{0}_{1}'.format( blabels0[2], dataset ), \
                    '{0}_{1}b'.format( blabels0[0], dataset ) ]
    elif baseline=='exp-decay':
        binit = np.array( [ offsf, 0, 0, offsb ] )
        bfixed = np.array( [ 0, 0, 0, 0 ] )
        blabels = [ '{0}_{1}f'.format( blabels0[0], dataset ), \
                    '{0}_{1}'.format( blabels0[1], dataset ), \
                    '{0}_{1}'.format( blabels0[2], dataset ), \
                    '{0}_{1}b'.format( blabels0[0], dataset ) ]
    else:
        pdb.set_trace()

    blabels = np.array( blabels, dtype=str )

    # Important bit is to ensure that the offsets are different,
    # but the baseline shape parameters are shared:
    bixs = {}
    for k in scankeys[dataset]:
        idkey = '{0}{1}'.format( dataset, k )
        bixs[idkey] = np.arange( 0, 0+nbpar )
    idkey = '{0}b'.format( dataset )
    bixs[idkey][0] = nbpar # replace offset for backward scan
    b = { 'blabels':blabels, 'bfixed':bfixed, 'bpars_init':binit, 'bixs':bixs }
    # THIS ROUTINE ABOVE DOES THE SHARING CORRECTLY...
    #print( self.baseline, offsf, offsb )
    #pdb.set_trace()
    return b

