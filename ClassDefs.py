import pdb, sys, os, glob, pickle, time, re
import copy
import numpy as np
import scipy.ndimage, scipy.interpolate
import astropy.io.fits as pyfits
import matplotlib
import matplotlib.pyplot as plt
import batman
from limbdark_dev import ld
from bayes.pyhm_dev import pyhm
from bayes.gps_dev.gps import gp_class, kernels
from . import UtilityRoutines as UR
from . import Systematics
from .mpfit import mpfit

try:
    os.environ['DISPLAY']
except:
    matplotlib.use('Agg')

# TODO: save pickle output in format that doesn't require the wfc3 package
# to be opened, i.e. don't save wfc3 class objects in this pickle output.
    
class WFC3SpecFit():
    def __init__( self ):
        self.slcs = None
        self.wmles = None
        self.results_dir = ''
        self.akey = ''
        self.lctype = 'ss'
        self.nchannels = None
        self.analysis = 'rdiff_zap'
        self.gpkernels = ''
        self.gpinputs = []
        self.syspars = {}
        self.ld = ''
        self.ldbat = ''
        self.ldpars = []
        self.orbpars = {}
        self.beta_free = True
        self.lineartbase = {} # set True/False for each visit
        #self.tr_type = ''
        self.prelim_fit = False
        self.ngroups = 5
        self.nwalkers = 100
        self.nburn1 = 100
        self.nburn2 = 250
        self.nsteps = 250
        self.RpRs_shared = True
        self.EcDepth_shared = True
    
    def GenerateMBundle( self ):
        parents = {}
        self.mbundle = {}
        self.initvals = {}
        if ( self.syspars['tr_type']=='primary' )*( self.RpRs_shared==True ):
            RpRs = pyhm.Uniform( 'RpRs', lower=0, upper=1 )
            self.mbundle['RpRs'] = RpRs
            parents['RpRs'] = RpRs
            self.initvals['RpRs'] = self.syspars['RpRs'][0]
            ldpars = self.SetupLDPars()
            parents.update( ldpars )
        if ( self.syspars['tr_type']=='secondary' )*( self.EcDepth_shared==True ):
            EcDepth = pyhm.Uniform( 'EcDepth', lower=0, upper=1 )
            self.mbundle['EcDepth'] = EcDepth
            parents['EcDepth'] = EcDepth
            self.initvals['EcDepth'] = self.syspars['EcDepth'][0]
        self.AddVisitMBundles( parents )
        print( '\nGlobal list of model parameters:' )
        for k in list( self.mbundle.keys() ):
            try:
                print( '{0} (free)'.format( self.mbundle[k].name.rjust( 30 ) ) )
            except:
                print( '{0}={1} (fixed)'.format( k, self.mbundle[k] ).rjust( 30 ) )
        return None

    def SetupLDPars( self ):
        dsets = list( self.slcs.keys() )
        ldkey = UR.GetLDKey( self.ld )
        if ldkey.find( 'nonlin' )>=0:
            self.ldbat = 'nonlinear'
            k = 'nonlin1d'
        elif ldkey.find( 'quad' )>=0:
            self.ldbat = 'quadratic'
            k = 'quad1d'
        else:
            pdb.set_trace()
        configs = []
        self.ldpars = {}
        for dset in dsets:
            configs += [ self.slcs[dset].config ]
            self.ldpars[configs[-1]] = self.slcs[dset].ld[k]
        configs = list( np.unique( np.array( configs ) ) )
        for c in configs:
            ldc = self.ldpars[c][self.chix,:]
            gamk = [ 'gam1_{0}'.format( c ), 'gam2_{0}'.format( c ) ]
            ck = [ 'c1_{0}'.format( c ), 'c2_{0}'.format( c ), \
                   'c3_{0}'.format( c ), 'c4_{0}'.format( c ) ]
            if ( self.ld.find( 'free' )>=0 ):
                ldsig = 0.6
                if ( self.ldbat=='quadratic' ):
                    gam1 = pyhm.Gaussian( gamk[0], mu=ldc[0], sigma=ldsig )
                    gam2 = pyhm.Gaussian( gamk[1], mu=ldc[1], sigma=ldsig )
                    self.initvals.update( { gamk[0]:ldc[0], gamk[1]:ldc[1] } )
                if ( self.ldbat=='nonlinear' ):
                    c1 = pyhm.Gaussian( ck[0], mu=ldc[0], sigma=ldsig )
                    c2 = pyhm.Gaussian( ck[1], mu=ldc[1], sigma=ldsig )
                    c3 = pyhm.Gaussian( ck[2], mu=ldc[2], sigma=ldsig )
                    c4 = pyhm.Gaussian( ck[3], mu=ldc[3], sigma=ldsig )
                    self.initvals.update( { ck[0]:ldc[0], ck[1]:ldc[1], \
                                            ck[2]:ldc[2], ck[3]:ldc[3] } )
            elif ( self.ld.find( 'fixed' )>=0 ):
                if ( self.ldbat=='quadratic' ):
                    gam1, gam2 = ldc
                elif ( self.ldbat=='nonlinear' ):
                    c1, c2, c3, c4 = ldc
            else:
                pdb.set_trace() # shouldn't happen
            if self.ldbat=='quadratic':
                self.mbundle.update( { gamk[0]:gam1, gamk[1]:gam2 } )
                ldpars = { 'gam1':gam1, 'gam2':gam2 }
            elif self.ldbat=='nonlinear':
                self.mbundle.update( { ck[0]:c1, ck[1]:c2, ck[2]:c3, ck[3]:c4 } )
                ldpars = { 'c1':c1, 'c2':c2, 'c3':c3, 'c4':c4 }
            else:
                pdb.set_trace()
        return ldpars
    
    def AddVisitMBundles( self, parents ):
        """
        Before calling this routine, any shared parameters have been defined.
        This routine then defines parameters specific to each visit, including
        parameters for the planet signal and systematics.
        """
        self.evalmodels = {}
        self.keepixs = {}
        dsets = list( self.slcs.keys() )
        nvisits = len( dsets )
        self.ndat = {}
        for j in range( nvisits ):
            k = dsets[j]
            parentsk = parents.copy()
            jd = self.slcs[k].jd
            self.ndat[k] = len( jd )
            if ( self.syspars['tr_type']=='primary' ):
                if self.RpRs_shared==False:
                    RpRslab = 'RpRs_{0}'.format( self.slcs[k].dsetname )
                    RpRs = pyhm.Uniform( RpRslab, lower=0, upper=1 )
                    self.mbundle[RpRslab] = RpRs
                    parentsk['RpRs'] = RpRs
                    self.initvals[RpRslab] = self.wmles[k]['RpRs']
            elif ( self.syspars['tr_type']=='secondary' ):
                if self.EcDepth_shared==False:
                    EcDepthlab = 'EcDepth_{0}'.format( self.slcs[k].dsetname )
                    EcDepth = pyhm.Uniform( EcDepthlab, lower=-1, upper=1 )
                    self.mbundle[EcDepthlab] = EcDepth
                    parents['EcDepth'] = EcDepth
                    self.initvals[EcDepthlab] = self.syspars['EcDepth'][0]
            else:
                pdb.set_trace() # shouldn't happen
            self.GPMBundle( k, parentsk )
        return None


    def BasisMatrix( self, dset, ixs ):
        phi = self.slcs[dset].auxvars[self.analysis]['hstphase'][ixs]
        tv = self.slcs[dset].auxvars[self.analysis]['tv'][ixs]
        x = self.slcs[dset].auxvars[self.analysis]['wavshift_pix'][ixs]
        phiv = ( phi-np.mean( phi ) )/np.std( phi )
        xv = ( x-np.mean( x ) )/np.std( x )
        offset = np.ones( self.ndat[dset] )[ixs]
        B = np.column_stack( [ offset, tv, xv, phiv, phiv**2., phiv**3., phiv**4. ] )
        return B

    
    def PolyFitCullixs( self, dset, config, ixs ):
        """
        Quick polynomial systematics model fit to identify remaining outliers.
        This routine could probably be broken into smaller pieces.
        """
        B = self.BasisMatrix( dset, ixs )
        syspars = self.syspars
        syspars['aRs'] = self.orbpars['aRs']
        syspars['b'] = self.orbpars['b']
        syspars['incl'] = self.orbpars['incl']
        jd = self.slcs[dset].jd[ixs]
        flux = self.slcs[dset].lc_flux[self.lctype][:,self.chix][ixs]
        uncs = self.slcs[dset].lc_uncs[self.lctype][:,self.chix][ixs]
        batpar, pmodel = self.GetBatmanObject( jd, dset, config )
        batpar.limb_dark = 'quadratic'
        batpar.u = self.slcs[dset].ld['quad1d'][self.chix,:]
        batpar.a = self.orbpars['aRs'] # where do these orbpars come from?
        batpar.inc = self.orbpars['incl'] # we want them to be whatever whitelc fit had...
        ntrials = 15
        if self.syspars['tr_type']=='primary':
            batpar.limb_dark = 'quadratic'
            batpar.u = self.slcs[dset].ld['quad1d'][self.chix,:]
            zstart = self.PolyFitPrimary( batpar, pmodel, B, flux, uncs, ntrials )
        elif self.syspars['tr_type']=='secondary':
            zstart = self.PolyFitSecondary( batpar, pmodel, B, flux, uncs, ntrials )
        else:
            pdb.set_trace()
        pinit, parkeys, mod_eval, neglogp = zstart
        pfits = []
        logps = np.zeros( ntrials )
        for i in range( ntrials ):
            print( i+1, ntrials )
            pfiti = scipy.optimize.fmin( neglogp, pinit[i], xtol=1e-5, \
                                         ftol=1e-5, maxfun=10000, maxiter=10000 )
            pfits += [ pfiti ]
            logps[i] = -neglogp( pfiti )
        pfit = pfits[np.argmax( logps )]
        psignal, polyfit = mod_eval( pfit )
        mfit = psignal*polyfit
        nsig = np.abs( flux-mfit )/uncs
        ixskeep = ixs[nsig<=5]
        self.nculled_poly = len( ixs )-len( ixskeep )
        if self.nculled_poly>0:
            print( '\nCulled {0:.0f} outliers\n'.format( self.nculled_poly ) )
        else:
            print( 'No outliers culled' )
        pfitdict = {}
        for i in range( len( parkeys ) ):
            pfitdict[parkeys[i]] = pfit[i]        
        return ixskeep, pfitdict

    def PolyFitPrimary( self, batpar, pmodel, B, flux, uncs, ntrials ):
        ndat = flux.size
        rperturb = np.random.random( ntrials )
        RpRs0 = self.syspars['RpRs'][0]*( 1+0.1*rperturb ) # want to come from whitelc fit
        parkeys = [ 'RpRs' ]
        def mod_eval( pars ):
            batpar.rp = pars[0]
            psignal = pmodel.light_curve( batpar )
            fluxc = flux/psignal
            coeffs = np.linalg.lstsq( B, fluxc, rcond=None )[0]
            polyfit = np.dot( B, coeffs )
            return psignal, polyfit
        def neglogp( pars ):
            psignal, polyfit = mod_eval( pars )
            resids = flux-psignal*polyfit
            return -UR.MVNormalWhiteNoiseLogP( resids, uncs, ndat )
        pinit = RpRs0
        return pinit, parkeys, mod_eval, neglogp
    
    def PolyFitSecondary( self, batpar, pmodel, B, flux, uncs, ntrials ):
        ndat = flux.size
        rperturb = np.random.random( ntrials )
        delT0 = ( rperturb-0.5 )/24.
        EcDepth0 = self.syspars['EcDepth'][0]*( 1+rperturb ) # want to come from whitelc fit
        parkeys = [ 'EcDepth' ]
        def mod_eval( pars ):
            batpar.fp = pars[0]
            psignal = pmodel.light_curve( batpar )
            fluxc = flux/psignal
            coeffs = np.linalg.lstsq( B, fluxc, rcond=None )[0]
            polyfit = np.dot( B, coeffs )
            return psignal, polyfit
        def neglogp( pars ):
            psignal, polyfit = mod_eval( pars )
            resids = flux-psignal*polyfit
            return -UR.MVNormalWhiteNoiseLogP( resids, uncs, ndat )
        pinit = EcDepth0
        return pinit, parkeys, mod_eval, neglogp
    

    

    def GPMBundle( self, dset, parents ):
        self.evalmodels[dset] = {}
        self.keepixs[dset] = {}
        ixs0 = np.arange( self.ndat[dset] )
        scanixs = {}
        scanixs['f'] = ixs0[self.slcs[dset].scandirs==1]
        scanixs['b'] = ixs0[self.slcs[dset].scandirs==-1]
        for k in self.scankeys[dset]:
            self.GetModelComponents( dset, parents, scanixs, k )
        return None

    
    def GetModelComponents( self, dset, parents, scanixs, scankey ):
        """
        Takes planet parameters in pars0, which have been defined separately
        to handle variety of cases with separate/shared parameters across
        visits etc. Then defines the systematics model for this visit+scandir
        combination, including the log-likelihood function. Returns complete 
        mbundle for current visit, with initvals and evalmodel.
        """
        slcs = self.slcs[dset]
        config = slcs.config
        ixs = scanixs[scankey]
        ixs, pfit0 = self.PolyFitCullixs( dset, config, ixs )
        self.keepixs[dset][scankey] = ixs
        idkey = '{0}{1}'.format( dset, scankey )
        gpinputs = self.gpinputs[dset]
        gpkernel = self.gpkernels[dset]
        betalabel = 'beta_{0}'.format( idkey )
        if self.beta_free==True:
            parents['beta'] = pyhm.Gaussian( betalabel, mu=1.0, sigma=0.2 )
            self.initvals[betalabel] = 1.0
        else:
            beta = 1
        self.mbundle[betalabel] = parents['beta']
        if self.syspars['tr_type']=='primary':
            RpRsk = parents['RpRs'].name
            self.initvals[RpRsk] = self.syspars['RpRs'][0]
        elif self.syspars['tr_type']=='secondary':
            EcDepthk = parents['EcDepth'].name
            self.initvals[EcDepthk] = self.syspars['EcDepth'][0]
        else:
            pdb.set_trace()
        batpar, pmodel = self.GetBatmanObject( slcs.jd[ixs], dset, slcs.config )
        z = self.GPLogLike( dset, parents, batpar, pmodel, ixs, idkey )
        loglikename = 'loglike_{0}'.format( idkey )
        self.mbundle[loglikename] = z['loglikefunc']
        self.mbundle[loglikename].name = loglikename
        evalmodelfunc = self.GetEvalModel( z, batpar, pmodel )
        self.evalmodels[dset][scankey] = [ evalmodelfunc, ixs ]
        return None

    
    
    
    def GetBatmanObject( self, jd, dset, config ):
        # Define the batman planet object:
        batpar = batman.TransitParams()
        batpar.per = self.syspars['P'][0]
        if self.syspars['tr_type']=='primary':
            batpar.rp = self.wmles[dset]['RpRs']#self.syspars['RpRs'][0]
            batpar.t0 = self.Tmids[dset]  # this is the white MLE value
        else:
            batpar.rp = self.syspars['RpRs'][0]
        if self.syspars['tr_type']=='secondary':
            batpar.fp = self.wmles[dset]['EcDepth']
            #batpar.t_secondary = self.syspars['Tmid'][0]
            batpar.t_secondary = self.Tmids[dset]  # this is the white MLE value
        batpar.a = self.orbpars['aRs']
        batpar.inc = self.orbpars['incl']
        batpar.ecc = self.syspars['ecc'][0] # in future, ecc and w could be in orbparrs
        batpar.w = self.syspars['omega'][0]
        batpar.limb_dark = self.ldbat
        batpar.u = self.ldpars[config][self.chix,:]
        pmodel = batman.TransitModel( batpar, jd, transittype=self.syspars['tr_type'] )
        # Following taken from here:
        # https://www.cfa.harvard.edu/~lkreidberg/batman/trouble.html#help-batman-is-running-really-slowly-why-is-this
        # Hopefully it works... but fac==None it seems... not sure why?
        fac = pmodel.fac
        pmodel = batman.TransitModel( batpar, jd, fac=fac, \
                                      transittype=self.syspars['tr_type'] )
        return batpar, pmodel

    def GetEvalModel( self, z, batpar, pmodel ):
        tr_type = self.syspars['tr_type']
        k = z['parlabels']
        def EvalModel( fitvals ):
            nf = 500
            jdf = np.r_[ z['jd'].min():z['jd'].max():1j*nf ]
            tvf = np.r_[ z['tv'].min():z['tv'].max():1j*nf ]
            ttrendf = fitvals[k['a0']] + fitvals[k['a1']]*tvf
            ttrend = fitvals[k['a0']] + fitvals[k['a1']]*z['tv']
            if tr_type=='primary':
                batpar.rp = fitvals[k['RpRs']]
                if ( self.ld.find( 'quad' )>=0 )*( self.ld.find( 'free' )>=0 ):
                    ldpars = np.array( [ fitvals[k['gam1']], fitvals[k['gam2']] ] )
                    batpar.u = ldpars
            elif tr_type=='secondary':
                batpar.fp = fitvals[k['EcDepth']]
            pmodelf = batman.TransitModel( batpar, jdf, transittype=tr_type )
            fac = pmodelf.fac
            pmodelf = batman.TransitModel( batpar, jdf, transittype=tr_type, \
                                           fac=fac )
            psignalf = pmodelf.light_curve( batpar )
            psignal = pmodel.light_curve( batpar )
            resids = z['flux']/( psignal*ttrend )-1. # model=psignal*ttrend*(1+GP)
            
            gp = z['zgp']['gp']
            Alabel = z['zgp']['Alabel_global']
            logiLlabels = z['zgp']['logiLlabels_global']
            logiL = []
            for i in logiLlabels:
                logiL += [ fitvals[i] ]
            iL = np.exp( np.array( logiL ) )
            gp.cpars = { 'amp':fitvals[Alabel], 'iscale':iL }
            # Currently the GP(t) baseline is hacked in; may be possible to improve:
            if 'Alabel_baset' in z['zgp']:
                pdb.set_trace() # this probably needs to be updated
                Alabel_baset = z['zgp']['Alabel_baset']
                iLlabel_baset = z['zgp']['iLlabel_baset']
                gp.cpars['amp_baset'] = fitvals[Alabel_baset]
                gp.cpars['iscale_baset'] = fitvals[iLlabel_baset]
            if self.beta_free==True:
                beta = fitvals[k['beta']]
            else:
                beta = 1
            gp.etrain = z['uncs']*beta
            gp.dtrain = np.reshape( resids, [ resids.size, 1 ] )
            mu, sig = gp.predictive( xnew=gp.xtrain, enew=gp.etrain )
            systematics = ttrend#+mu.flatten()#*( mu.flatten() + 1 )
            bestfits = { 'psignal':psignal, 'ttrend':ttrend, 'mu':mu.flatten(), \
                         'jdf':jdf, 'psignalf':psignalf, 'ttrendf':ttrendf }
            zout = { 'psignal':psignal, 'ttrend':ttrend, 'mu':mu.flatten(), \
                     'jdf':jdf, 'psignalf':psignalf, 'ttrendf':ttrendf }
            return { 'arrays':zout, 'batpar':batpar, 'pmodel':pmodel }
        return EvalModel
    
    def PrepPlanetVarsPrimary( self, dset, RpRs ):
        """
        Returns the free parameter objects, initial values and 
        information required by batman for limb darkening.
        """
        slcs = self.slcs[dset]
        pars0 = { 'RpRs':RpRs }
        initvals = { RpRs.name:self.wmles[dset]['RpRs'] }
        return pars0, initvals, ldbat, ldpars

        
    def PrepPlanetVarsSecondary( self, dset, EcDepth ):
        """
        Returns the free parameter objects, initial values and 
        information required by batman for limb darkening.
        """
        slcs = self.slcs[dset]
        pars0 = { 'EcDepth':EcDepth }
        initvals = { EcDepth.name:self.wmles[dset]['EcDepth'] }
        ldbat = 'quadratic'
        ldpars = [ 0, 0 ] # no stellar limb darkening
        return pars0, initvals, ldbat, ldpars


    def GPLogLike( self, dset, parents, batpar, pmodel, ixs, idkey ):
        slcs = self.slcs[dset]
        jd = slcs.jd[ixs]
        tv = slcs.auxvars[self.analysis]['tv'][ixs]
        flux = slcs.lc_flux[self.lctype][ixs,self.chix]
        uncs = slcs.lc_uncs[self.lctype][ixs,self.chix]
        lintcoeffs = UR.LinTrend( jd, tv, flux )
        ldbat = self.ldbat
        #pars = {}
        #initvals = {}
        a0k = 'a0_{0}'.format( idkey )
        parents['a0'] = pyhm.Uniform( a0k, lower=0.5, upper=1.5 )
        self.mbundle[a0k] = parents['a0']
        self.initvals[a0k] = lintcoeffs[0]
        if self.lineartbase[dset]==True:
            a1k = 'a1_{0}'.format( idkey )
            parents['a1'] = pyhm.Uniform( a1k, lower=-0.1, upper=0.1 )
            self.mbundle[a1k] = parents['a1']
            self.initvals[a1k] = lintcoeffs[1]
        zgp = self.PrepGP( dset, ixs, idkey )        
        for k in zgp['gpvars'].keys():
            parents[k] = zgp['gpvars'][k]
        n0 = 30
        print( 'Model parameters for {0}'.format( dset ).center( 2*n0+1 ) )
        print( '{0} {1}'.format( 'Local'.rjust( n0 ),'Global'.rjust( n0 ) ) )
        for k in list( parents.keys() ):
            try:
                print( '{0} {1} (free)'\
                       .format( k.rjust( n0 ), parents[k].name.rjust( n0 ) ) )
            except:
                print( '{0} {1} (fixed)'.format( k.rjust( n0 ), k.rjust( n0 ) ) )
        @pyhm.stochastic( observed=True )
        def loglikefunc( value=flux, parents=parents ):
            def logp( value, parents=parents ):
                logp_val = self.GetGPLogLikelihood( jd, flux, uncs, tv, parents, \
                                                    zgp, batpar, pmodel, \
                                                    self.lineartbase[dset] ) # TODO
                return logp_val
        for k in list( zgp['gpvars'].keys() ):
            l = zgp['gpvars'][k].name
            self.mbundle[l] = zgp['gpvars'][k]
            self.initvals[l] = zgp['gpinitvals'][k]
        parlabels = {}
        for k in list( parents.keys() ):
            try:
                parlabels[k] = parents[k].name
            except:
                pass
        #zout = { 'pars':pars, 'initvals':initvals, 'loglikefunc':loglikefunc, \
        #         'batpar':batpar, 'pmodel':pmodel, 'jd':jd, 'tv':tv, \
        #         'flux':flux, 'uncs':uncs, 'parlabels':parlabels, 'zgp':zgp }
        zout = { 'loglikefunc':loglikefunc, 'batpar':batpar, 'pmodel':pmodel, \
                 'jd':jd, 'tv':tv, 'flux':flux, 'uncs':uncs, \
                 'parlabels':parlabels, 'zgp':zgp }
        return zout
    
    def PrepGP( self, dset, ixs, idkey ):
        gp = gp_class.gp( which_type='full' )
        gp.mfunc = None
        gp.cfunc = self.gpkernels[dset]
        gp.mpars = {}
        gpinputs = self.gpinputs[dset]
        auxvars = self.slcs[dset].auxvars[self.analysis]

        #auxvars = wlc.whitelc[analysis]['auxvars']
        cond1 = ( gp.cfunc==kernels.sqexp_invL_ard )
        cond2 = ( gp.cfunc==kernels.matern32_invL_ard )
        cond3 = ( gp.cfunc==kernels.sqexp_invL )
        cond4 = ( gp.cfunc==kernels.matern32_invL )
        cond5 = ( gp.cfunc==Systematics.custom_kernel_sqexp_invL_ard )
        cond6 = ( gp.cfunc==Systematics.custom_kernel_mat32_invL_ard )
        cond7 = ( gp.cfunc==kernels.sqexp_ard )
        cond8 = ( gp.cfunc==kernels.matern32_ard )
        if cond1+cond2+cond3+cond4: # implies logiL_prior==True
            #z = PrepGP_invL( gp, self.gpinputs[dset], self.auxvars, ixs, idkey )
            #z = UR.GPinvL( gp, gpinputs, auxvars, ixs, idkey )
            z = self.GPinvL( dset, gp, ixs, idkey )
        elif cond5+cond6: # implieslogiL_prior==True
            z = self.GPinvLbaset( dset, gp, ixs, idkey )
            #pdb.set_trace() # todo PrepGP_ard( gp, auxvars, idkey )
        elif cond7+cond8: # implieslogiL_prior==False also
            pdb.set_trace() # todo PrepGP_ard( gp, auxvars, idkey )
        return z

    def GPinvL( self, dset, gp, ixs, idkey ):
        """
        Define GP parameterized in terms of inverse correlation length
        scales. Although it's not tested, this routine is designed to
        handle 1D or ND input variables.
        """ 
        gpvars = {}
        gpinitvals = {}
        Alabel_global = 'Amp_{0}'.format( idkey )        
        gpvars['Amp'] = pyhm.Gamma( Alabel_global, alpha=1, beta=1e2 )
        #gpvars[Alabel] = pyhm.Uniform( Alabel, lower=0, upper=1 )
        gpinitvals['Amp'] = 1e-5
        xtrain = []
        logiLlabels_global = []
        logiLlabels_local = []
        for i in self.gpinputs[dset]:
            k, label = UR.GetVarKey( i )
            v = self.slcs[dset].auxvars[self.analysis][k]
            vs = ( v-np.mean( v ) )/np.std( v )
            pname = 'logiL{0}'.format( label )
            mlabel = '{0}_{1}'.format( pname, idkey )
            gpvari = UR.DefineLogiLprior( vs[ixs], i, mlabel, \
                                          priortype='uniform' )
            gpvars[pname] = gpvari
            logiLlow = gpvars[pname].parents['lower']
            logiLupp = gpvars[pname].parents['upper']
            gpinitvals[pname] = 1e-6
            xtrain += [ vs[ixs] ]
            logiLlabels_global += [ mlabel ]
            logiLlabels_local += [ pname ]
        gp.xtrain = np.column_stack( xtrain )
        zout = { 'gp':gp, 'gpvars':gpvars, 'gpinitvals':gpinitvals, \
                 'Alabel_global':Alabel_global, 'Alabel_local':'Amp', \
                 'logiLlabels_global':logiLlabels_global, \
                 'logiLlabels_local':logiLlabels_local }
        return zout

    def GPinvLbaset( self, dset, gp, ixs, idkey ):
        # TODO = adapt to UtilityRoutines like GPinvL().
        return None
    
    def RunMLE( self ):
        if self.prelim_fit==True:
            mp = pyhm.MAP( self.mbundle )
            for k in list( self.initvals.keys() ):
                mp.model.free[k].value = self.initvals[k]
            print( '\nRunning MLE fit...' )
            print( '\nFree parameters and initial values:' )
            for k in mp.model.free.keys():
                print( k, mp.model.free[k].value )
            print( '\noptmising...' )
            mp.fit( xtol=1e-5, ftol=1e-5, maxfun=10000, maxiter=10000 )
            print( 'Done.' )
            print( '\nMLE results:' )
            self.mle = {}
            for k in mp.model.free.keys():
                self.mle[k] = mp.model.free[k].value
        else:
            prelim_fpaths = self.GetFilePaths( prelim_fit=True )
            print( '\nReading in preliminary MLE fit:' )
            print( prelim_fpaths[1] )
            ifile = open( prelim_fpaths[1], 'rb' )
            prelim = pickle.load( ifile )
            ifile.close()
            self.mle = prelim['mle']
        for k in list( self.mle.keys() ):
            print( k, self.mle[k] )
        print( 'Done.\n' )                
        return None
    
    def RunMCMC( self ):
        # Initialise the emcee sampler:
        mcmc = pyhm.MCMC( self.mbundle )
        self.freepars = list( mcmc.model.free.keys() )
        mcmc.assign_step_method( pyhm.BuiltinStepMethods.AffineInvariant )
        # Define ranges to randomly sample the initial walker values from
        # (Note: GetParRanges is a function provided by user during setup):
        self.init_par_ranges = self.GetParRanges( self.mle )
        # Initial emcee burn-in with single walker group:
        #init_walkers = self.GetInitWalkers( mcmc )
        init_walkers = UR.GetInitWalkers( mcmc, self.nwalkers, self.init_par_ranges )
        mcmc.sample( nsteps=self.nburn1, init_walkers=init_walkers, verbose=False )
        mle_refined = UR.RefineMLE( mcmc.walker_chain, self.mbundle )
        #self.init_par_ranges = self.GetParRanges( self.mle )
        self.init_par_ranges = self.GetParRanges( mle_refined )
        #init_walkers = self.GetInitWalkers( mcmc )
        init_walkers = UR.GetInitWalkers( mcmc, self.nwalkers, self.init_par_ranges )
        # Sample for each chain, i.e. group of walkers:
        self.walker_chains = []
        print( '\nRunning the MCMC sampling:' )
        for i in range( self.ngroups ):
            t1 = time.time()
            print( '\n... group {0} of {1}'.format( i+1, self.ngroups ) )
            # Run the burn-in:
            print( '... running burn-in for {0} steps'.format( self.nburn2 ) )
            mcmc.sample( nsteps=self.nburn2, init_walkers=init_walkers, \
                         verbose=False )
            burn_end_state = UR.GetWalkerState( mcmc )
            # Run the main chain:
            print( '... running main chain for {0} steps'.format( self.nsteps ) )
            mcmc.sample( nsteps=self.nsteps, init_walkers=burn_end_state, \
                         verbose=False )
            self.walker_chains += [ mcmc.walker_chain ]
            t2 = time.time()
        # Refine the MLE solution using MCMC output:
        self.mle = UR.RefineMLEfromGroups( self.walker_chains, self.mbundle )
        self.ExtractMCMCOutput( nburn=0 )
        self.Save()
        #self.Plot()
        return None

    def Save( self ):
        mcmc_fpath, mle_fpath = self.GetFilePaths( prelim_fit=self.prelim_fit )
        self.specfit_mcmc_fpath_pkl = mcmc_fpath
        self.specfit_mcmc_fpath_txt = mcmc_fpath.replace( '.pkl', '.txt' )
        self.specfit_mle_fpath_pkl = mle_fpath
        bestfits, batpars, pmodels = UR.BestFitsEval( self.mle, self.evalmodels )
        self.bestfits = bestfits
        self.batpars = batpars
        self.pmodels = pmodels
        outp = {}
        outp['slcs'] = self.slcs
        outp['wmles'] = self.wmles
        outp['gpkernels'] = self.gpkernels
        outp['gpinputs'] = self.gpinputs
        outp['analysis'] = self.analysis
        outp['keepixs'] = self.keepixs
        #pdb.set_trace()
        outp['batpars'] = self.batpars
        outp['pmodels'] = self.pmodels
        outp['bestfits'] = bestfits
        outp['mle'] = self.mle
        outp['freepars'] = self.freepars
        outp['orbpars'] = self.orbpars
        outp['syspars'] = self.syspars
        ofile = open( self.specfit_mle_fpath_pkl, 'wb' )
        pickle.dump( outp, ofile )
        # Add in the bulky MCMC output:
        outp['chain'] = self.chain
        outp['walker_chains'] = self.walker_chains
        outp['grs'] = self.grs
        outp['chain_properties'] = self.chain_properties
        outp['ngroups'] = self.ngroups
        outp['nwalkers'] = self.nwalkers
        outp['nsteps'] = self.nsteps
        outp['nburn'] = self.nburn2
        ofile = open( self.specfit_mcmc_fpath_pkl, 'wb' )
        pickle.dump( outp, ofile )
        ofile.close()
        # Write to the text file:
        self.TxtOut()
        print( '\nSaved:\n{0}\n{1}\n{2}\n'.format( self.specfit_mcmc_fpath_pkl, \
                                                   self.specfit_mcmc_fpath_txt, \
                                                   self.specfit_mle_fpath_pkl ) )
        return None

    def TxtOut( self ):
        chp = self.chain_properties
        text_str = '#\n# Sample properties: parameter, median, l34, u34, gr\n#\n'
        keys = chp['median'].keys()
        for key in keys:
            if key!='logp':
                text_str += '{0} {1:.6f} -{2:.6f} +{3:.6f} {4:.3f}\n'\
                             .format( key, chp['median'][key], \
                             np.abs( chp['l34'][key] ), chp['u34'][key], \
                             self.grs[key] )
        ofile = open( self.specfit_mcmc_fpath_txt, 'w' )
        ofile.write( text_str )
        ofile.close()
        return text_str


    def GetODir( self ):
        dirbase = os.path.join( self.results_dir, 'spec' )
        if self.syspars['tr_type']=='primary':        
            dirbase = os.path.join( dirbase, self.ld )
        else:
            dirbase = os.path.join( dirbase, 'ldoff' )
        dsets = list( self.slcs.keys() )
        dsets = UR.NaturalSort( dsets )
        dirext = ''
        for k in dsets:
            dirext += '+{0}'.format( k )
        dirext = dirext[1:]
        if len( dsets )>1:
            if self.syspars['tr_type']=='primary':
                if self.RpRs_shared==True:
                    dirext += '.RpRs_shared'
                else:
                    dirext += '.RpRs_individ'
            elif self.syspars['tr_type']=='secondary':
                if self.EcDepth_shared==True:
                    dirext += '.EcDepth_shared'
                else:
                    dirext += '.EcDepth_individ'
            else:
                pdb.set_trace()
        dirbase = os.path.join( dirbase, dirext )
        if self.akey=='':
            print( '\n\nMust set akey to create output folder for this particular analysis\n\n' )
            pdb.set_trace()
        else:
            odir = os.path.join( dirbase, self.akey )
        self.odir = os.path.join( odir, 'nchan{0:.0f}'.format( self.nchannels ) )
        # Don't bother with the reduction parameters in the filenames.
        # That can be done separately with a custom routine defined by
        # the user if it's really important.
        return None

    
    def GetFilePaths( self, prelim_fit=True ):
        self.prelimstr, self.betastr = UR.GetStrs( prelim_fit, self.beta_free )
        self.GetODir()
        if os.path.isdir( self.odir )==False:
            os.makedirs( self.odir )
        oname = 'spec.{0}.{1}.{2}.mcmc.{3}.ch{4:.0f}.pkl'\
                .format( self.analysis, self.betastr, self.lctype, \
                         self.prelimstr, self.chix )
                                                              
        mcmc_fpath = os.path.join( self.odir, oname )
        mle_fpath = mcmc_fpath.replace( 'mcmc', 'mle' )
        return mcmc_fpath, mle_fpath

    def ExtractMCMCOutput( self, nburn=0 ):
        chaindict, grs = UR.GetChainFromWalkers( self.walker_chains, nburn=nburn )
        logp_arr = chaindict['logp']
        logp = chaindict.pop( 'logp' )
        keys_fitpars = list( chaindict.keys() )
        npar = len( keys_fitpars )
        nsamples = len( logp_arr )
        chain = np.zeros( [ nsamples, npar ] )
        for j in range( npar ):
            chain[:,j] = chaindict[keys_fitpars[j]]
        chainprops = pyhm.chain_properties( chaindict, nburn=0, thin=None, \
                                            print_to_screen=True )
        self.chain_properties = chainprops
        self.grs = grs
        self.chain = chaindict
        return None

    
    def EvalPsignalPrimary( self, jd, parents, batpar, pmodel ):
        batpar.rp = parents['RpRs']
        if batpar.limb_dark=='quadratic':
            ldpars = np.array( [ parents['gam1'], parents['gam2'] ] )
        elif batpar.limb_dark=='nonlinear':
            ldpars = np.array( [ parents['c1'], parents['c2'], \
                                 parents['c3'], parents['c4'] ] )
        batpar.u = ldpars
        psignal = pmodel.light_curve( batpar )
        return psignal

    
    def EvalPsignalSecondary( self, jd, parents, batpar, pmodel ):
        batpar.fp = parents['EcDepth']
        psignal = pmodel.light_curve( batpar )
        return psignal

    def GetGPLogLikelihood( self, jd, flux, uncs, tv, parents, \
                            zgp, batpar, pmodel, lineartbase ):
        if lineartbase==True:
            ttrend = parents['a0'] + parents['a1']*tv#[ixs]
        else:
            ttrend = parents['a0']
        if self.syspars['tr_type']=='primary':
            if batpar.limb_dark=='quadratic':
                batpar.u = np.array( [ parents['gam1'], parents['gam2'] ] )
            elif batpar.limb_dark=='nonlinear':
                batpar.u = np.array( [ parents['c1'], parents['c2'], \
                                       parents['c3'], parents['c4'] ] )
            psignal = self.EvalPsignalPrimary( jd, parents, batpar, pmodel )
        elif self.syspars['tr_type']=='secondary':
            psignal = self.EvalPsignalSecondary( jd, parents, batpar, pmodel )
        else:
            pdb.set_trace()
        #resids = flux - psignal*ttrend
        resids = flux/( psignal*ttrend )-1. # model=psignal*ttrend*(1+GP)
        logiL = []
        for i in zgp['logiLlabels_local']:
            logiL += [ parents[i] ]
        iL = np.exp( np.array( logiL ) )
        gp = zgp['gp']
        gp.cpars = { 'amp':parents[zgp['Alabel_local']], 'iscale':iL }
        if 'Alabel_baset' in zgp:
            gp.cpars['amp_baset'] = parents[zgp['Alabel_baset']]
            gp.cpars['iscale_baset'] = parents[zgp['iLlabel_baset']]
        gp.etrain = uncs*parents['beta']
        gp.dtrain = np.reshape( resids, [ resids.size, 1 ] )
        logp_val = gp.logp_builtin()
        return logp_val

class WFC3SpecFitLM():
    
    def __init__( self ):
        self.results_dir = ''
        self.akey = ''
        self.analysis = 'rdiff'
        self.scankeys = {}
        self.ld = ''
        self.orbpars = {}
        self.Tmids = {}
        self.RpRs_shared = True
        self.EcDepth_shared = True
        self.slcs = {}
        self.syspars = {}
        self.ttrend = 'linear'
        self.ntrials = 20
        self.chix = 0

    def PrepData( self ):
        """
        For a collection of spectroscopic light curves and channel index, 
        returns a single concatenated array for all the data, along with a 
        dictionary ixs containing the indices of each dataset within that
        big data array.
        """
        self.dsets = list( self.slcs.keys() )
        ndsets = len( self.dsets )
        analysis = self.analysis
        lctype = self.lctype
        self.SetupLDPars()
        data = []
        ixs = {} # indices to split vstacked data
        self.keepixs = {} # indices to map vstacked data back to original
        self.pmodels = {} # pmodels for each data configuration
        self.batpars = {} # batpars for each data configuration
        #self.pmodelfs = {}
        i1 = 0
        for i in range( ndsets ):
            dset = self.dsets[i]
            #Tmidi = self.Tmids[dset]
            slcs = self.slcs[dset]
            self.wavedgesmicr = slcs.wavedgesmicr[self.chix]
            ixsi = np.arange( slcs.jd.size )
            scanixs = {}
            scanixs['f'] = ixsi[slcs.scandirs==1]
            scanixs['b'] = ixsi[slcs.scandirs==-1]
            ixs[dset] = {}
            self.keepixs[dset] = []
            nf = 500
            for k in self.scankeys[dset]:
                if scanixs[k].size==0:
                    print( '\nNo {0}-scan in {1} dataset. Remove from scankeys.\n'\
                           .format( k, dset ) )
                    pdb.set_trace()
                jdi = slcs.jd[scanixs[k]]
                jdf = np.linspace( jdi.min(), jdi.max(), nf )
                thrsi = 24*( jdi-slcs.jd[0] ) # time since start of visit in hours
                torbi = slcs.auxvars[analysis]['torb'][scanixs[k]]
                fluxi = slcs.lc_flux[lctype][scanixs[k],self.chix]
                uncsi = slcs.lc_uncs[lctype][scanixs[k],self.chix]
                data += [ np.column_stack( [ jdi, thrsi, torbi, fluxi, uncsi ] ) ]
                i2 = i1+len( fluxi )
                ixs[dset][k] = np.arange( i1, i2 )
                self.keepixs[dset] += [ np.arange( slcs.jd.size )[scanixs[k]] ]
                batparik, pmodelik = self.GetBatmanObject( dset, jdi, slcs.config )
                #batparifk, pmodelifk = self.GetBatmanObject( jdif, wlc.config )
                idkey = '{0}{1}'.format( dset, k )
                self.pmodels[idkey] = pmodelik # TODO = change to [dset][k]?
                self.batpars[idkey] = batparik # TODO = change to [dset][k]?
                #self.pmodelfs[idkey] = pmodelifk # TODO = change to [dset][k]?
                # Slide the index along for next visit:
                i1 = i2
            keepixsd = np.concatenate( self.keepixs[dset] )
            ixsk = np.argsort( keepixsd )
            self.keepixs[dset] = keepixsd[ixsk]
        # Package data together in single array for mpfit:
        self.data = np.vstack( data ) 
        self.data_ixs = ixs
        return None
    
    def PrepModelParams( self ):
        """
        Sets up the arrays organizing the parameters in a format that can
        then be used with mpfit. Returns:
        'labels' - A list of strings giving the names of each parameter.
        'fixed' - An array of 1's and 0's indicating which parameters are
        held fixed and which are allowed to vary, as per the 'case' input.
        'pars_init' - An array containing the default starting values
        for each parameter.
        'ixs' - A dictionary containing indices that map the parameters from
        each individual dataset onto the joint parameter list that gets
        passed to mpfit.
        """
        #dsets = list( self.wlcs.keys() )
        ndsets = len( self.dsets )
        # Determine preliminary values for baseline and planet parameters:
        p, b = self.PrepPlanetPars( self.syspars['tr_type'] )
        # Combine into global parameter list:
        nppar_total = len( p['pars_init'] )
        self.pars_init = np.concatenate( [ p['pars_init'], b['pars_init'] ] )
        self.par_labels = np.concatenate( [ p['labels'], b['labels'] ] )
        self.fixed = np.concatenate( [ p['fixed'], b['fixed'] ] )
        ixs = {}
        c = 0
        for i in range( ndsets ):
            dset = self.dsets[i]
            ixsi = []
            for j in range( len( self.scankeys[dset] ) ):
                idkey = '{0}{1}'.format( dset, self.scankeys[dset][j] )
                bixsij = nppar_total + b['ixs'][idkey]
                ixs[idkey] = np.concatenate( [ p['ixs'][dset], bixsij ] )
        self.par_ixs = ixs
        return None
    
    def PrepPlanetPars( self, transittype ):
        """        
        Returns dictionaries for the planet and baseline parameters, containing:
          ixs = dictionary with indices for each dataset
          labels = list of parameter labels
          fixed = list of which parameters are fixed and free
          init = list of initial values for each parameter
        """
        plabels, pinit, pfixed = self.InitialPPars( transittype )
        blabels0, binit0, bfixed0 = self.InitialBPars()        
        ng = len( pinit )
        pixsg = np.arange( ng ) # global (across visits) planet parameters
        pixs = {}
        bixs = {}
        ndsets = len( self.dsets )
        blabels = []
        bfixed = []
        binit = []
        c = 0 # counter
        for k in range( ndsets ):
            pixs[self.dsets[k]] = pixsg
            bparsk = self.PrelimBPars( self.dsets[k] )
            blabels += [ bparsk['blabels'] ]
            bfixed = np.concatenate( [ bfixed, bparsk['bfixed'] ] )
            binit = np.concatenate( [ binit, bparsk['bpars_init'] ] )
            for i in list( bparsk['bixs'].keys() ):
                bixs[i] = bparsk['bixs'][i]+c
            c += len( bparsk['blabels'] )
        plabels = np.array( plabels )
        #blabels = np.array( blabels ).flatten()
        blabels = np.concatenate( blabels ).flatten()
        p = { 'labels':plabels, 'fixed':pfixed, 'pars_init':pinit, 'ixs':pixs }
        b = { 'labels':blabels, 'fixed':bfixed, 'pars_init':binit, 'ixs':bixs }
        return p, b

    def InitialBPars( self ):
        """
        Returns clean starting arrays for baseline trend arrays.
        """
        if self.ttrend=='linear':
            binit0 = [ 1, 0 ]
            bfixed0 = [ 0, 0 ]
            blabels0 = [ 'b0', 'b1' ]
        elif self.ttrend=='quadratic':
            binit0 = [ 1, 0, 0 ]
            bfixed0 = [ 0, 0, 0 ]
            blabels0 = [ 'b0', 'b1', 'b2' ]
        return blabels0, binit0, bfixed0
    
    def PrelimBPars( self, dataset ):
        """
        """
        blabels0, binit0, bfixed0 = self.InitialBPars()
        nbpar = len( binit0 )
        slcs = self.slcs[dataset]
        ixs = np.arange( slcs.jd.size )
        #scanixs = {}
        #scanixs['f'] = ixs[slcs.scandirs==1]
        #scanixs['b'] = ixs[slcs.scandirs==-1]
        bpars_init = []
        bfixed = []
        blabels = []
        bixs = {}
        c = 0 # counter
        for k in self.scankeys[dataset]:
            idkey = '{0}{1}'.format( dataset, k )
            ixsk = self.data_ixs[dataset][k]
            thrsk = self.data[:,1][ixsk]#[scanixs[k]]
            fluxk = self.data[:,3][ixsk]#[scanixs[k]]
            orbixs = UR.SplitHSTOrbixs( thrsk )
            t1 = np.median( thrsk[0] )
            t2 = np.median( thrsk[-1] )
            f1 = np.median( fluxk[0] )
            f2 = np.median( fluxk[-1] )
            grad = ( f2-f1 )/( t2-t1 )
            offs = f1-grad*t1
            if self.ttrend=='linear':
                binitk = [ offs, grad ]
            elif self.ttrend=='quadratic':
                binitk = [ offs, grad, 0 ]
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

    def InitialPPars( self, transittype ):
        """
        Returns clean starting arrays for planet parameter arrays.
        """
        #dsets = list( self.wlcs.keys() )
        config = self.slcs[self.dsets[0]].config
        ldpars = self.ldpars[config][self.chix,:]
        pinit0 = []
        if transittype=='primary':
            s = 3
            if self.ld.find( 'quad' )>=0:
                plabels = [ 'RpRs', 'gam1', 'gam2' ]
                try:
                    pinit0 += [ self.ppar_init['RpRs'] ]                        
                except:
                    pinit0 += [ self.syspars['RpRs'][0] ]
                pinit0 += [ ldpars[0], ldpars[1] ]
                pinit0 = np.array( pinit0 )
                if self.ld.find( 'fixed' )>=0:
                    pfixed = np.array( [ 0, 1, 1 ] )
                if self.ld.find( 'free' )>=0:
                    pfixed = np.array( [ 0, 0, 0 ] )
            elif self.ld.find( 'nonlin' )>=0:
                pdb.set_trace() # TODO - implement 4-parameter LD
        elif transittype=='secondary':
            plabels = [ 'EcDepth' ]
            try:
                pinit0 += [ self.ppar_init['EcDepth'] ]
            except:
                pinit0 += [ self.syspars['EcDepth'][0] ]
            pinit0 = np.array( pinit0 )
            pfixed = np.array( [ 0 ] )
        return plabels, pinit0, pfixed
    
    def PreFitting( self, niter=2, sigcut=10 ):
        ixsd = self.data_ixs
        ixsm = self.keepixs
        ixsp = self.par_ixs
        batp = self.batpars
        syspars = self.syspars
        data = self.data
        npar = len( self.pars_init )
        print( '\nRunning initial fit for full model:' )
        self.pars_init = self.RunTrials( 5 )
        print( 'Done.\n' )
        ncull = 0
        ndat = 0
        print( '\nIterating over multiple trials to flag outliers:' )
        tt = syspars['tr_type']
        for g in range( niter ):
            pars_fit = self.RunTrials( self.ntrials )
            mfit = self.CalcModel( pars_fit )
            ffit = mfit['psignal']*mfit['ttrend']
            ixsmg = {}
            for dset in self.dsets:
                scandirs = self.slcs[dset].scandirs[ixsm[dset]]
                ixsmg['f'] = ixsm[dset][scandirs==1]
                ixsmg['b'] = ixsm[dset][scandirs==-1]
                ixsmz = []
                for k in self.scankeys[dset]:
                    ixsdk = ixsd[dset][k]
                    ixsmzk = ixsmg[k]
                    idkey = '{0}{1}'.format( dset, k )
                    residsk = self.data[ixsdk,3]-ffit[ixsdk]
                    uncsk = self.data[ixsdk,4]
                    nsig = np.abs( residsk )/uncsk                           
                    ixs = ( nsig<sigcut ) # within threshold
                    ncull += int( nsig.size-ixs.sum() )
                    self.pmodels[idkey] = batman.TransitModel( batp[idkey], \
                                                               data[ixsdk,0][ixs], \
                                                               transittype=tt )
                    ixsd[dset][k] = ixsdk[ixs]
                    ixsmz += [ ixsmzk[ixs] ]
                    ndat += len( residsk )
                ixsmz = np.concatenate( ixsmz )
                ixs0 = np.argsort( ixsmz )
                ixsm[dset] = ixsmz[ixs0]
            print( 'Iteration={0:.0f}, Nculled={1:.0f}'.format( g+1, ncull ) )
            self.pars_init = pars_fit
        print( '\n{0}\nRescaling measurement uncertainties by:\n'.format( 50*'#' ) )
        rescale = {}
        for dset in self.dsets:
            rescale[dset] = {}
            for k in self.scankeys[dset]:
                idkey = '{0}{1}'.format( dset, k )
                ixsdk = ixsd[dset][k]
                zk = ( self.data[ixsdk,3]-ffit[ixsdk] )/self.data[ixsdk,4]
                chi2k = np.sum( zk**2. )
                rchi2k = chi2k/float( len( residsk )-len( pars_fit[ixsp[idkey]] ) )
                rescale[dset][k] = np.sqrt( rchi2k )
                print( '{0:.2f} for {1}{2}'.format( rescale[dset][k], dset, k ) )
        for dset in self.dsets:
            for k in self.scankeys[dset]:
                self.data[ixsd[dset][k],4] *= rescale[dset][k]
        self.uncertainties_rescale = rescale
        print( '{0}\n'.format( 50*'#' ) )    
        self.model_fit = None # reset for main FitModel() run
        self.pars_fit = None # reset for main FitModel() run
        self.data_ixs = ixsd
        self.keepixs = ixsm
        return None

    def CalcModel( self, pars ):
        """
        For a parameter array for a specific dataset, the parameters
        are *always* the following (at least for now):
           RpRs, ldcoeff1, ..., ldcoeffN, b1, b2, (b3)
        or:
           EcDepth, b1, b2, (b3)
        So you can always unpack these in this order and send them as
        inputs to their appropriate functions. 
        """
        if self.ttrend=='linear':
            def bfunc( thrs, pars ):
                return pars[0] + pars[1]*thrs
        elif self.ttrend=='quadratic':
            def bfunc( thrs, pars ):
                return pars[0] + pars[1]*thrs + pars[2]*( thrs**2. )
        else:
            pdb.set_trace()
        ndat, nvar = np.shape( self.data )
        batp = self.batpars
        pmod = self.pmodels
        psignal = np.zeros( ndat )
        ttrend = np.zeros( ndat )
        jd = self.data[:,0]
        thrs = self.data[:,1]
        torb = self.data[:,2]
        flux = self.data[:,3]
        uncs = self.data[:,4]
        ndsets = len( self.dsets )
        self.UpdateBatpars( pars )

        for i in range( ndsets ):
            dset = self.dsets[i]
            for k in self.scankeys[dset]:
                idkey = '{0}{1}'.format( dset, k )
                ixsdk = self.data_ixs[dset][k]
                parsk = pars[self.par_ixs[idkey]]
                if pmod[idkey].transittype==1:
                    if batp[idkey].limb_dark=='quadratic':
                        m = 2
                    elif batp[idkey].limb_dark=='nonlinear':
                        m = 4
                    else:
                        pdb.set_trace()
                    s = 1+m # RpRs
                else:
                    s = 1 # EcDepth
                psignal[ixsdk] = pmod[idkey].light_curve( batp[idkey] )
                # Evaluate the systematics signal:
                ttrend[ixsdk] = bfunc( thrs[ixsdk], parsk[s:] )
        return { 'psignal':psignal, 'ttrend':ttrend }

    def CalcChi2( self ):
        pfit = self.model_fit['psignal'] # planet signal
        sfit = self.model_fit['ttrend'] # systematics
        ffit = pfit*sfit
        chi2 = 0
        for dset in list( self.data_ixs.keys() ):
            for k in list( self.data_ixs[dset].keys() ):
                ixsdk = self.data_ixs[dset][k]
                residsk = ( self.data[ixsdk,3]-ffit[ixsdk] )
                uncsk = self.data[ixsdk,4]
                chi2 += np.sum( ( residsk/uncsk )**2. )
        return chi2
    
    def SetupLDPars( self ):
        ldkey = UR.GetLDKey( self.ld )
        if ldkey.find( 'nonlin' )>=0:
            self.ldbat = 'nonlinear'
            k = 'nonlin1d'
        elif ldkey.find( 'quad' )>=0:
            self.ldbat = 'quadratic'
            k = 'quad1d'
        else:
            pdb.set_trace()
        configs = []
        self.ldpars = {}
        for dset in self.dsets:
            configs += [ self.slcs[dset].config ]
            self.ldpars[configs[-1]] = self.slcs[dset].ld[k]
        return None
    

    def GetBatmanObject( self, dset, jd, config ):
        # Define the batman planet object:
        batpar = batman.TransitParams()
        batpar.per = self.syspars['P'][0]
        batpar.ecc = self.syspars['ecc'][0]
        batpar.w = self.syspars['omega'][0]
        batpar.limb_dark = self.ldbat
        batpar.u = self.ldpars[config][self.chix,:]
        batpar.a = self.orbpars['aRs']
        batpar.inc = self.orbpars['incl']
        self.ppar_init = {}
        if self.syspars['tr_type']=='primary':
            batpar.rp = self.wmles[dset]['RpRs']
            batpar.t0 = self.Tmids[dset]
            self.ppar_init['RpRs'] = batpar.rp
        if self.syspars['tr_type']=='secondary':
            batpar.rp = self.syspars['RpRs'][0]
            batpar.fp = self.mles[dset]['EcDepth']
            batpar.t0 = self.syspars['T0'][0]
            batpar.t_secondary = self.Tmids[dset]
            self.ppar_init['EcDepth'] = batpar.rp
        pmodel = batman.TransitModel( batpar, jd, transittype=self.syspars['tr_type'] )
        # Following taken from here:
        # https://www.cfa.harvard.edu/~lkreidberg/batman/trouble.html#help-batman-is-running-really-slowly-why-is-this
        # Hopefully it works... but fac==None it seems... not sure why?
        fac = pmodel.fac
        pmodel = batman.TransitModel( batpar, jd, fac=fac, \
                                      transittype=self.syspars['tr_type'] )
        return batpar, pmodel

    def UpdateBatpars( self, pars ):
        batp = self.batpars
        pmod = self.pmodels
        ndsets = len( self.dsets )
        for i in range( ndsets ):
            dset = self.dsets[i]
            for k in self.scankeys[dset]:
                idkey = '{0}{1}'.format( dset, k )
                ixsdk = self.data_ixs[dset][k]
                parsk = pars[self.par_ixs[idkey]]
                # Evaluate the planet signal:
                if pmod[idkey].transittype==1:
                    # Primary transits have RpRs and optionally limb darkening:
                    batp[idkey].rp = parsk[0]
                    if batp[idkey].limb_dark=='quadratic':
                        m = 2
                    elif batp[idkey].limb_dark=='nonlinear':
                        m = 4
                    else:
                        pdb.set_trace()
                    batp[idkey].u = parsk[1:1+m]
                elif pmod[idkey].transittype==2:
                    # Secondary eclipses only have the eclipse depth:
                    batp[idkey].fp = parsk[0]
                else:
                    pdb.set_trace()
        self.batpars = batp
        return None
    
    def BestFitsOut( self ):
        jd = self.data[:,0]
        thrs = self.data[:,1]
        self.bestfits = {}
        nf = 500
        for dset in list( self.data_ixs.keys() ):
            self.bestfits[dset] = {}
            for k in list( self.data_ixs[dset].keys() ):
                idkey = '{0}{1}'.format( dset, k )
                ixsdk = self.data_ixs[dset][k]
                pfitdk = self.model_fit['psignal'][ixsdk] # planet signal
                tfitdk = self.model_fit['ttrend'][ixsdk] # baseline trend
                jddk = jd[ixsdk]
                jdfdk = np.linspace( jddk.min(), jddk.max(), nf )
                pmodfk = batman.TransitModel( self.batpars[idkey], jdfdk, \
                                              transittype=self.syspars['tr_type'] )
                pfitfdk = pmodfk.light_curve( self.batpars[idkey] )
                # Evaluate ttrend - would be better to have a more elegant way of doing this:
                thrsdk = thrs[ixsdk]
                thrsfdk = np.linspace( thrsdk.min(), thrsdk.max(), nf )
                pfit = self.pars_fit['pvals'][self.par_ixs[idkey]]
                if self.ttrend=='linear':
                    ttrendfdk = pfit[-2]+pfit[-1]*thrsfdk
                elif self.ttrend=='quadratic':
                    ttrendfdk = pfit[-3]+pfit[-2]*thrsfdk+( pfit[-1]*( thrsfdk**2. ) )
                self.bestfits[dset][k] = {}
                self.bestfits[dset][k]['jd'] = jddk
                self.bestfits[dset][k]['psignal'] = pfitdk
                self.bestfits[dset][k]['ttrend'] = tfitdk
                self.bestfits[dset][k]['jdf'] = jdfdk
                self.bestfits[dset][k]['psignalf'] = pfitfdk
                self.bestfits[dset][k]['ttrendf'] = ttrendfdk
        return None
    
    def TxtOut( self, save_to_file=True ):
        ostr = ''
        ostr +=  '{0}\n# status = {1}'.format( 50*'#', self.pars_fit['status'] )
        try:
            ostr1 = '\n# Uncertainty rescaling factors:'
            for k in list( self.data_ixs.keys() ):
                ostr1 += '\n#  {0} = {1:.4f}'.format( k, self.uncertainties_rescale[k] )
            ostr += ostr1
        except:
            pass
        ostr += '\n# Fit results:\n#{0}'.format( 49*'-' )
        npar = len( self.fixed )
        pvals = self.pars_fit['pvals']
        puncs = self.pars_fit['puncs']
        for i in range( npar ):
            col1 = '{0}'.format( self.par_labels[i].rjust( 15 ) )
            col2 = '{0:20f}'.format( pvals[i] ).replace( ' ', '' ).rjust( 20 )
            col3 = '{0:20f}'.format( puncs[i] ).replace( ' ', '' ).ljust( 20 )
            if self.fixed[i]==0:
                ostr += '\n{0} = {1} +/- {2} (free)'.format( col1, col2, col3 )
            else:
                ostr += '\n{0} = {1} +/- {2} (fixed)'.format( col1, col2, col3 )
        ostr += '\n# Tmid assumed for each dataset:'
        for d in list( self.slcs.keys() ):
            ostr += '\n#  {0} = {1}'.format( d, self.Tmids[d] )
        if save_to_file==True:
            ofile = open( self.specfit_fpath_txt, 'w' )
            ofile.write( ostr )
            ofile.close()
        return ostr

    def GetODir( self ):
        dirbase = os.path.join( self.results_dir, 'spec' )
        if self.syspars['tr_type']=='primary':
            dirbase = os.path.join( dirbase, self.ld )
        else:
            dirbase = os.path.join( dirbase, 'ldoff' )
        dsets = UR.NaturalSort( self.dsets )
        dirext = ''
        for k in dsets:
            dirext += '+{0}'.format( k )
        dirext = dirext[1:]
        if len( dsets )>1:
            if self.syspars['tr_type']=='primary':
                if self.RpRs_shared==True:
                    dirext += '.RpRs_shared'
                else:
                    dirext += '.RpRs_individ'
            elif self.syspars['tr_type']=='secondary':
                if self.EcDepth_shared==True:
                    dirext += '.EcDepth_shared'
                else:
                    dirext += '.EcDepth_individ'
            else:
                pdb.set_trace()
        dirbase = os.path.join( dirbase, dirext )
        if self.akey=='':
            print( '\n\nMust set akey to create output folder for this particular analysis\n\n' )
            pdb.set_trace()
        else:
            self.odir = os.path.join( dirbase, self.akey )
        self.odir = os.path.join( self.odir, 'nchan{0:.0f}'.format( self.nchannels ) )
        # Don't bother with the reduction parameters in the filenames.
        # That can be done separately with a custom routine defined by
        # the user if it's really important.
        return None
    
    def GetFilePath( self ):
        self.GetODir()
        if os.path.isdir( self.odir )==False:
            os.makedirs( self.odir )
        #if self.beta_free==True:
        #    betastr = 'beta_free'
        #else:
        #    betastr = 'beta_fixed'
        oname = 'spec.{0}.{1}.mpfit.{2}base.ch{3:.0f}.pkl'\
            .format( self.analysis, self.lctype, self.ttrend, self.chix )
        opath = os.path.join( self.odir, oname )
        return opath
    
    def Save( self ):
        outp = {}
        outp['slcs'] = self.slcs
        outp['wmles'] = self.wmles        
        outp['analysis'] = self.analysis
        outp['lctype'] = self.lctype
        outp['chix'] = self.chix
        # vstacked data:
        outp['data_ixs'] = self.data_ixs
        outp['data'] = self.data
        #outp['cullixs_init'] = self.cullixs
        outp['keepixs_final'] = self.keepixs
        outp['par_ixs'] = self.par_ixs
        outp['model_fit'] = self.model_fit
        self.BestFitsOut()
        outp['bestfits'] = self.bestfits
        outp['uncertainties_rescale'] = self.uncertainties_rescale
        outp['par_labels'] = self.par_labels
        outp['pars_fit'] = self.pars_fit
        outp['wavedgesmicr'] = self.wavedgesmicr
        outp['mle'] = {}
        for i in range( self.npar ):
            outp['mle'][self.par_labels[i]] = self.pars_fit['pvals'][i]
        outp['fixed'] = self.fixed
        outp['batpars'] = self.batpars
        outp['pmodels'] = self.pmodels
        outp['syspars'] = self.syspars
        outp['orbpars'] = self.orbpars
        outp['Tmids'] = self.Tmids
        opath_pkl = self.GetFilePath()
        ofile = open( opath_pkl, 'wb' )
        pickle.dump( outp, ofile )
        ofile.close()
        opath_txt = opath_pkl.replace( '.pkl', '.txt' )
        self.specfit_fpath_pkl = opath_pkl
        self.specfit_fpath_txt = opath_txt
        # Write to the text file:
        self.TxtOut( save_to_file=True )
        print( '\nSaved:\n{0}\n{1}\n'.format( self.specfit_fpath_pkl, \
                                              self.specfit_fpath_txt ) )
        return None
    
    
    def FitModel( self, save_to_file=False, verbose=True ):
        def NormDeviates( pars, fjac=None, data=None ):
            """
            Function defined in format required by mpfit.
            """
            m = self.CalcModel( pars )
            fullmodel = m['psignal']*m['ttrend']
            resids = data[:,3]-fullmodel
            status = 0
            rms = np.sqrt( np.mean( resids**2. ) )
            return [ status, resids/data[:,4] ]
        self.npar = len( self.par_labels )
        parinfo = []
        for i in range( self.npar ):
            parinfo += [ { 'value':self.pars_init[i], 'fixed':int( self.fixed[i] ), \
                           'parname':self.par_labels[i], \
                           'limited':[0,0], 'limits':[0.,0.] } ]
        fa = { 'data':self.data }
        m = mpfit( NormDeviates, self.pars_init, functkw=fa, parinfo=parinfo, \
                   maxiter=1e3, ftol=1e-5, quiet=True )
        if (m.status <= 0): print( 'error message = ', m.errmsg )
        self.pars_fit = { 'pvals':m.params, 'puncs':m.perror, 'pcov':m.covar, \
                          'ndof':m.dof, 'status':m.status }
        self.model_fit = self.CalcModel( m.params )
        if save_to_file==True:
            self.Save()
            #self.Plot() # TODO
        ostr = self.TxtOut( save_to_file=save_to_file )
        if verbose==True:
            print( ostr )
        return None

    
    def RunTrials( self, ntrials ):
        """
        Fit the light curve model using multiple randomized starting parameter values.
        """
        npar = len( self.pars_init )
        chi2 = np.zeros( ntrials )
        trials = []
        print( '\nTrials with randomly perturbed starting positions:' )
        for i in range( ntrials ):
            print( i+1, ntrials )
            for j in range( npar ):
                if self.fixed[j]==0:
                    if i>0: # Note: first trial is unperturbed
                        v = self.pars_init[j]
                        dv = 0.05*np.random.randn()*np.abs( v )
                        self.pars_init[j] = v + dv
            self.FitModel( save_to_file=False, verbose=False )
            chi2[i] = self.CalcChi2()
            trials += [ self.pars_fit['pvals'] ]
        return trials[np.argmin(chi2)]



    
class WFC3WhiteFitLM():
    """
    Uses Levenberg-Marquardt as implemented by mpfit.
    Fits the systematics with double exponential ramp
    model described in de Wit et al (2018).
    
    Routines:
    PrepData()
    PrepModelParams()
    PreFitting()
    FitModel()
    """
    def __init__( self ):
        self.wlcs = None
        self.results_dir = ''
        self.akey = ''
        self.analysis = 'rdiff_zap'
        self.gpkernels = ''
        self.gpinputs = []
        self.scankeys = {}
        self.syspars = {}
        self.ld = ''
        self.ldbat = ''
        self.ldpars = []
        self.orbpars = ''
        self.beta_free = True
        self.Tmid0 = {}
        self.Tmid_free = True
        self.ntrials = 10
        #self.batpar = {} # maybe have a dict of these for each dset
        #self.pmodel = {}
        self.lineartbase = {} # set True/False for each visit
        self.tr_type = ''
        self.ngroups = 5
        self.nwalkers = 100
        self.nburn1 = 100
        self.nburn2 = 250
        self.nsteps = 250
        self.RpRs_shared = True
        self.EcDepth_shared = True

    def PrepData( self ):
        """
        For a collection of white light curves, returns a single
        concatenated array for all the data, along with a dictionary
        ixs containing the indices of each dataset within that
        big data array.
        """
        self.dsets = list( self.wlcs.keys() )
        ndsets = len( self.dsets )
        analysis = self.analysis
        self.SetupLDPars()
        data = []
        ixs = {} # indices to split vstacked data
        self.keepixs = {} # indices to map vstacked data back to original
        self.pmodels = {} # pmodels for each data configuration
        self.batpars = {} # batpars for each data configuration
        #self.pmodelfs = {}
        self.Tmid0 = {} # literature mid-times for each data configuration
        i1 = 0
        for i in range( ndsets ):
            dset = self.dsets[i]
            wlc = self.wlcs[dset]
            ixsc = self.cullixs[dset]
            # Define scanixs to already be culled before steps below:
            scanixs = {}
            scanixs['f'] = ixsc[wlc.scandirs[ixsc]==1]
            scanixs['b'] = ixsc[wlc.scandirs[ixsc]==-1]
            Tmidi = self.syspars['Tmid'][0]
            while Tmidi<wlc.jd.min():
                Tmidi += self.syspars['P'][0]
            while Tmidi>wlc.jd.max():
                Tmidi -= self.syspars['P'][0]
            self.Tmid0[dset] = Tmidi
            ixs[dset] = {}
            self.keepixs[dset] = []
            nf = 500
            for k in self.scankeys[dset]:
                if scanixs[k].size==0:
                    print( '\nNo {0}-scan in {1} dataset. Remove from scankeys.\n'\
                           .format( k, dset ) )
                    pdb.set_trace()
                jdi = wlc.jd[scanixs[k]]
                jdf = np.linspace( jdi.min(), jdi.max(), nf )
                thrsi = 24*( jdi-wlc.jd[0] ) # time since start of visit in hours
                torbi = wlc.whitelc[analysis]['auxvars']['torb'][scanixs[k]]
                fluxi = wlc.whitelc[analysis]['flux'][scanixs[k]]
                uncsi = wlc.whitelc[analysis]['uncs'][scanixs[k]]
                data += [ np.column_stack( [ jdi, thrsi, torbi, fluxi, uncsi ] ) ]
                i2 = i1+len( fluxi )
                ixs[dset][k] = np.arange( i1, i2 )
                self.keepixs[dset] += [ np.arange( wlc.jd.size )[scanixs[k]] ]
                batparik, pmodelik = self.GetBatmanObject( jdi, wlc.config )
                #batparifk, pmodelifk = self.GetBatmanObject( jdif, wlc.config )
                idkey = '{0}{1}'.format( dset, k )
                self.pmodels[idkey] = pmodelik # TODO = change to [dset][k]?
                self.batpars[idkey] = batparik # TODO = change to [dset][k]?
                #self.pmodelfs[idkey] = pmodelifk # TODO = change to [dset][k]?
                # Slide the index along for next visit:
                i1 = i2
            keepixsd = np.concatenate( self.keepixs[dset] )
            ixsk = np.argsort( keepixsd )
            self.keepixs[dset] = keepixsd[ixsk]
        # Package data together in single array for mpfit:
        self.data = np.vstack( data ) 
        self.data_ixs = ixs
        #keepixs = np.concatenate( ixsm )
        #pdb.set_trace()
        return None


    def PrepModelParams( self ):
        """
        Sets up the arrays organizing the parameters in a format that can
        then be used with mpfit. Returns:
        'labels' - A list of strings giving the names of each parameter.
        'fixed' - An array of 1's and 0's indicating which parameters are
        held fixed and which are allowed to vary, as per the 'case' input.
        'pars_init' - An array containing the default starting values
        for each parameter.
        'ixs' - A dictionary containing indices that map the parameters from
        each individual dataset onto the joint parameter list that gets
        passed to mpfit.
        """
        ndsets = len( self.dsets )
        # Determine preliminary values for ramp parameters:
        r, fluxc = self.PrepRampPars()
        # Determine preliminary values for baseline and planet parameters:
        p, b = self.PrepPlanetPars( self.syspars['tr_type'], fluxc )
        nppar_total = len( p['pars_init'] )
        # Combine the ramp and baseline parameters:
        s = { 'labels':[], 'fixed':[], 'pars_init':[], 'ixs':{} }
        c = 0
        for i in list( r['ixs'].keys() ):
            rixs = r['ixs'][i]
            bixs = b['ixs'][i]
            s['labels'] += [ np.concatenate( [ r['labels'][rixs], \
                                               b['labels'][bixs] ] ) ]
            s['fixed'] += [ np.concatenate( [ r['fixed'][rixs], \
                                              b['fixed'][bixs] ] ) ]
            s['pars_init'] += [ np.concatenate( [ r['pars_init'][rixs], \
                                                  b['pars_init'][bixs] ] ) ]
            nspar = len( rixs )+len( bixs )
            s['ixs'][i] = np.arange( c, c+nspar )
            c += nspar
        for j in ['labels','fixed','pars_init']:
            s[j] = np.concatenate( s[j] )
        # Combine into global parameter list:
        self.pars_init = np.concatenate( [ p['pars_init'], s['pars_init'] ] )
        self.par_labels = np.concatenate( [ p['labels'], s['labels'] ] )
        self.fixed = np.concatenate( [ p['fixed'], s['fixed'] ] )
        ixs = {}
        c = 0
        for i in range( ndsets ):
            dset = self.dsets[i]
            ixsi = []
            for j in range( len( self.scankeys[dset] ) ):
                idkey = '{0}{1}'.format( dset, self.scankeys[dset][j] )
                sixsij = nppar_total + s['ixs'][idkey]
                ixs[idkey] = np.concatenate( [ p['ixs'][dset], sixsij ] )
        self.par_ixs = ixs
        self.npar = len( self.par_labels )
        print( '\nInitial parameter values:' )
        for i in range( self.npar ):
            print( '  {0} = {1}'.format( self.par_labels[i], self.pars_init[i] ) )
        return None


    def PrepPlanetPars( self, transittype, fluxc ):
        """        
        Returns dictionaries for the planet and baseline parameters, containing:
          ixs = dictionary with indices for each dataset
          labels = list of parameter labels
          fixed = list of which parameters are fixed and free
          init = list of initial values for each parameter
        """
        # DE code currently hard-wired for single Rp/Rs and limb darkening
        # parameters across all datasets, which implicity assumes same
        # config, i.e. passband, for all datasets; TODO = generalize.
        plabels, pinit0, pfixed = self.InitialPPars( transittype )
        ng = len( pinit0 )
        pixsg = np.arange( ng ) # global (across visits) planet parameters
        pixs = {}
        bixs = {}
        # Set up the visit-specific baseline and planet parameters:
        #dsets = list( self.wlcs.keys() )
        ndsets = len( self.dsets )
        blabels = []
        bfixed = []
        binit = []
        c = 0 # counter
        pinit = pinit0
        self.delTixs = {}
        for k in range( ndsets ):
            try:
                delTk = self.ppar_init['delT_{0}'.format( self.dsets[k] )]
            except:
                delTk = 0            
            if self.Tmid_free==True:
                pfixed = np.concatenate( [ pfixed, [0] ] ) # delT free for each visit
            else:
                pfixed = np.concatenate( [ pfixed, [1] ] ) # delT fixed to zero
            pinitk = np.concatenate( [ pinit0, [delTk] ] )
            # Get initial values for delT and baseline parameters:
            delTk, bparsk = self.PrelimPFit( fluxc, pinitk, transittype, self.dsets[k] )
            # Add a delT parameter for each dataset:
            pinit = np.concatenate( [ pinit, [ delTk ] ] ) 
            pixs[self.dsets[k]] = np.concatenate( [ pixsg, [ ng+k ] ] )
            delTlab = 'delT_{0}'.format( self.dsets[k] )
            plabels += [ delTlab ]
            blabels += [ bparsk['blabels'] ]
            self.delTixs[self.dsets[k]] = ng+k
            bfixed = np.concatenate( [ bfixed, bparsk['bfixed'] ] )
            binit = np.concatenate( [ binit, bparsk['bpars_init'] ] )
            for i in list( bparsk['bixs'].keys() ):
                bixs[i] = bparsk['bixs'][i]+c
            c += len( bparsk['blabels'] )
        plabels = np.array( plabels )
        blabels = np.array( blabels ).flatten()
        p = { 'labels':plabels, 'fixed':pfixed, 'pars_init':pinit, 'ixs':pixs }
        b = { 'labels':blabels, 'fixed':bfixed, 'pars_init':binit, 'ixs':bixs }
        self.nppar = len( plabels )
        self.nbpar = len( blabels )
        #pdb.set_trace()
        return p, b

    def InitialBPars( self ):
        """
        Returns clean starting arrays for baseline trend arrays.
        """
        if self.ttrend=='linear':
            binit0 = [ 1, 0 ]
            bfixed0 = [ 0, 0 ]
            blabels0 = [ 'b0', 'b1' ]
        elif self.ttrend=='quadratic':
            binit0 = [ 1, 0, 0 ]
            bfixed0 = [ 0, 0, 0 ]
            blabels0 = [ 'b0', 'b1', 'b2' ]
        return blabels0, binit0, bfixed0
    
    def InitialPPars( self, transittype ):
        """
        Returns clean starting arrays for planet parameter arrays.
        """
        #dsets = list( self.wlcs.keys() )
        config = self.wlcs[self.dsets[0]].config
        ldpars = self.ldpars[config]
        pinit0 = []
        if transittype=='primary':
            s = 5
            if self.ld.find( 'quad' )>=0:
                plabels = [ 'RpRs', 'aRs', 'b', 'gam1', 'gam2' ]
                for l in [ 'RpRs', 'aRs', 'b' ]:
                    try:
                        pinit0 += [ self.ppar_init[l] ]
                    except:
                        pinit0 += [ self.syspars[l] ]
                pinit0 += [ ldpars[0], ldpars[1] ]
                pinit0 = np.array( pinit0 )
                if self.ld.find( 'fixed' )>=0:
                    if self.orbpars.find( 'free' )>=0:
                        pfixed = np.array( [ 0, 0, 0, 1, 1 ] )
                    else:
                        pfixed = np.array( [ 0, 1, 1, 1, 1 ] )
                if self.ld.find( 'free' )>=0:
                    if self.orbpars.find( 'free' )>=0:
                        pfixed = np.array( [ 0, 0, 0, 0, 0 ] )
                    else:
                        pfixed = np.array( [ 0, 1, 1, 0, 0 ] )
            elif self.ld.find( 'nonlin' )>=0:
                plabels = [ 'RpRs', 'aRs', 'b', 'c1', 'c2', 'c3', 'c4' ]
                for l in [ 'RpRs', 'aRs', 'b' ]:
                    try:
                        pinit0 += [ self.ppar_init[l] ]
                    except:
                        pinit0 += [ self.syspars[l] ]
                #pinit0 += [ ldpars[0], ldpars[1] ]
                pinit0 += [ ldpars[0], ldpars[1], ldpars[2], ldpars[3] ]
                pinit0 = np.array( pinit0 )
                if self.ld.find( 'fixed' )>=0:
                    if self.orbpars.find( 'free' )>=0:
                        pfixed = np.array( [ 0, 0, 0, 1, 1, 1, 1 ] )
                    else:
                        pfixed = np.array( [ 0, 1, 1, 1, 1, 1, 1 ] )
                if self.ld.find( 'free' )>=0:
                    if self.orbpars.find( 'free' )>=0:
                        pfixed = np.array( [ 0, 0, 0, 0, 0, 0, 0 ] )
                    else:
                        pfixed = np.array( [ 0, 1, 1, 0, 0, 0, 0 ] )
        elif transittype=='secondary':
            plabels = [ 'EcDepth', 'aRs', 'b' ]
            for l in [ 'EcDepth', 'aRs', 'b' ]:
                try:
                    pinit0 += [ self.ppar_init[l] ]
                except:
                    pinit0 += [ self.syspars[l][0] ]
            pinit0 = np.array( pinit0 )
            if self.orbpars.find( 'fixed' )>=0:
                pfixed = np.array( [ 0, 1, 1 ] )
            else:
                pfixed = np.array( [ 0, 0, 0 ] )
        return plabels, pinit0, pfixed

    
    def PrelimPFit( self, fluxc, pinit, transittype, dataset ):
        """
        After dividing the data by an initial fit for the systematics ramp,
        this routine performs a preliminary fit for the transit signal and 
        baseline trend. For the transit signal, the sole aim is to obtain a
        starting value for the delT parameters.
        """
        if transittype=='primary':
            if self.ld.find( 'quad' )>=0:
                RpRs, aRs, b, gam1, gam2, delT = pinit
                pinit = [ RpRs, delT ]
            elif self.ld.find( 'nonlin' )>=0:
                RpRs, aRs, b, c1, c2, c3, c4, delT = pinit
                pinit = [ RpRs, delT ]
        elif transittype=='secondary':
                EcDepth, aRs, b, delT = pinit
                pinit = [ EcDepth, delT ]
        blabels0, binit0, bfixed0 = self.InitialBPars()
        nbpar = len( binit0 )        
        # Sort out the baseline indices for each scan direction
        # and normalize the data using the preliminary ramp fit:
        thrs = self.data[:,1]
        fluxn = {} # normalized flux
        binit = []
        bixs = {}
        c = 0 # counter
        for k in self.scankeys[dataset]:
            idkey = '{0}{1}'.format( dataset, k )
            orbixs = UR.SplitHSTOrbixs( thrs[self.data_ixs[dataset][k]] )
            nmed = min( [ int( len( orbixs[-1] )/2. ), 3 ] )
            fluxn[k] = fluxc[idkey]/np.median( fluxc[idkey][orbixs[-1]][-nmed:] )
            binit += binit0
            bixs[idkey] = np.arange( c, c+nbpar )
            c += nbpar
        if self.Tmid_free==True:
            nppar = len( pinit )
            pfiti = self.OptimizeDelTBaseline( dataset, thrs, fluxn, pinit, \
                                               binit, bixs, transittype )
            delT = pfiti[1]
        else:
            nppar = 1 # because delT excluded
            pfiti = self.OptimizeBaselineOnly( dataset, thrs, fluxn, pinit, \
                                               binit, bixs, transittype )
            
        bpars_init = []
        bfixed = []
        blabels = []
        for k in self.scankeys[dataset]:
            idkey = '{0}{1}'.format( dataset, k )
            bpars_init = np.concatenate( [ bpars_init, pfiti[nppar+bixs[idkey]] ] )
            bfixed = np.concatenate( [ bfixed, bfixed0 ] )
            blabels_k = []
            for j in range( nbpar ):
                blabels_k += [ '{0}_{1}'.format( blabels0[j], idkey ) ]
            blabels += [ np.array( blabels_k, dtype=str ) ]
        blabels = np.concatenate( blabels )
        b = { 'blabels':blabels, 'bfixed':bfixed, 'bpars_init':bpars_init, 'bixs':bixs }
        return delT, b

    def OptimizeDelTBaseline( self, dataset, thrs, fluxn, pinit, binit, bixs, transittype ):
        nppar = len( pinit )
        pars0 = np.concatenate( [ pinit, binit ] )
        Tmid0 = self.Tmid0[dataset]
        def CalcModelBasic( p ):
            """
            Takes the mid-time and either RpRs or EcDepth, along with
            the baseline parameters, and returns the planet signal
            and baseline trend as output.
            """
            psignal = {}
            baseline = {}
            for k in self.scankeys[dataset]:
                idkey = '{0}{1}'.format( dataset, k )
                if transittype=='primary':
                    self.batpars[idkey].rp = p[0]
                    self.batpars[idkey].t0 = Tmid0 + p[1]
                elif transittype=='secondary':
                    self.batpars[idkey].fp = p[0]                    
                    self.batpars[idkey].t_secondary = Tmid0 + p[1]
                psignal[idkey] = self.pmodels[idkey].light_curve( self.batpars[idkey] )
                bparsk = p[nppar+bixs[idkey]]
                thrsk = thrs[self.data_ixs[dataset][k]]
                if self.ttrend=='linear':
                    baseline[idkey] = bparsk[0] + bparsk[1]*thrsk
                elif self.ttrend=='quadratic':
                    baseline[idkey] = bparsk[0] + bparsk[1]*thrsk + bparsk[2]*( thrsk**2. )
                else:
                    pdb.set_trace()
            return psignal, baseline
        def CalcRMS( pars ):
            m = CalcModelBasic( pars )
            resids = []
            for k in self.scankeys[dataset]:
                idkey = '{0}{1}'.format( dataset, k )
                mk = m[0][idkey]*m[1][idkey]
                resids += [ fluxn[k]-mk ]
            return np.sqrt( np.mean( np.concatenate( resids )**2. ) )
        return scipy.optimize.fmin( CalcRMS, pars0 )
        
    
    def OptimizeBaselineOnly( self, dataset, thrs, fluxn, pinit, binit, bixs, transittype ):
        nppar = 1 # because delT is excluded
        pars0 = np.concatenate( [ [ pinit[0] ], binit ] ) # exclude delT here
        Tmid0 = self.Tmid0[dataset]
        def CalcModelBasic( p ):
            """
            Takes the mid-time and either RpRs or EcDepth, along with
            the baseline parameters, and returns the planet signal
            and baseline trend as output.
            """
            psignal = {}
            baseline = {}
            for k in self.scankeys[dataset]:
                idkey = '{0}{1}'.format( dataset, k )
                if transittype=='primary':
                    self.batpars[idkey].rp = p[0]
                    self.batpars[idkey].t0 = Tmid0 #+ p[1]
                elif transittype=='secondary':
                    self.batpars[idkey].fp = p[0]                    
                    self.batpars[idkey].t_secondary = Tmid0 #+ p[1]
                psignal[idkey] = self.pmodels[idkey].light_curve( self.batpars[idkey] )
                bparsk = p[nppar+bixs[idkey]]
                thrsk = thrs[self.data_ixs[dataset][k]]
                if self.ttrend=='linear':
                    baseline[idkey] = bparsk[0] + bparsk[1]*thrsk
                elif self.ttrend=='quadratic':
                    baseline[idkey] = bparsk[0] + bparsk[1]*thrsk + bparsk[2]*( thrsk**2. )
                else:
                    pdb.set_trace()
            return psignal, baseline
        def CalcRMS( pars ):
            m = CalcModelBasic( pars )
            resids = []
            for k in self.scankeys[dataset]:
                idkey = '{0}{1}'.format( dataset, k )
                mk = m[0][idkey]*m[1][idkey]
                resids += [ fluxn[k]-mk ]
            return np.sqrt( np.mean( np.concatenate( resids )**2. ) )
        return scipy.optimize.fmin( CalcRMS, pars0 )
        
    
    def PrepRampPars( self ):
        thrs = self.data[:,1]
        torb = self.data[:,2]
        flux = self.data[:,3]
        ixsd = self.data_ixs
        # For each scan direction, the systematics model consists of a
        # double-exponential ramp (a1,a2,a3,a4,a5):
        rlabels0 = [ 'a1', 'a2', 'a3', 'a4', 'a5' ]
        #rlabels0 = [ 'a0', 'a1', 'a2', 'a3', 'a4', 'a5' ] #DELETE
        # Initial values for systematics parameters:
        rlabels = []
        rfixed = []
        rinit = []
        rixs = {}
        fluxc = {}
        c = 0 # counter
        #dsets = list( self.wlcs.keys() )
        ndsets = len( self.dsets )
        for i in range( ndsets ):
            dset = self.dsets[i]
            for k in self.scankeys[dset]:
                ixsdk = ixsd[dset][k]
                idkey = '{0}{1}'.format( dset, k )
                # Run a quick double-exponential ramp fit on the first
                # and last HST orbits to get reasonable starting values
                # for the parameters:
                rpars0, fluxcik = self.PrelimDEFit( thrs[ixsdk], torb[ixsdk], \
                                                    flux[ixsdk] )
                rinit = np.concatenate( [ rinit, rpars0 ] )
                nrpar = len( rpars0 )
                rixs[idkey] = np.arange( c*nrpar, (c+1)*nrpar )
                fluxc[idkey] = fluxcik
                rfixed = np.concatenate( [ rfixed, np.zeros( nrpar ) ] )
                rlabels_ik = []
                for j in range( nrpar ):
                    rlabels_ik += [ '{0}_{1}{2}'.format( rlabels0[j], dset, k ) ]
                rlabels += [ np.array( rlabels_ik, dtype=str ) ]
                c += 1
        rlabels = np.concatenate( rlabels )
        r = { 'labels':rlabels, 'fixed':rfixed, 'pars_init':rinit, 'ixs':rixs }
        #print( rinit )
        #pdb.set_trace()
        return r, fluxc

    
    def PrelimDEFit( self, thrs, torb, flux ):
        """
        Performs preliminary fit for the ramp systematics, only
        fitting to the first and last HST orbits.
        """
        if self.ttrend=='linear':
            rfunc = UR.DERampLinBase
            nbase = 2
        elif self.ttrend=='quadratic':
            rfunc = UR.DERampQuadBase
            nbase = 3
        else:
            pdb.set_trace()
        orbixs = UR.SplitHSTOrbixs( thrs )
        ixs = np.concatenate( [ orbixs[0], orbixs[-1] ] )
        def CalcRMS( pars ):
            ttrend, ramp = rfunc( thrs[ixs], torb[ixs], pars )
            resids = flux[ixs]-ttrend*ramp
            rms = np.sqrt( np.mean( resids**2. ) )
            return rms
        ntrials = 30
        rms = np.zeros( ntrials )
        pfit = []
        for i in range( ntrials ):
            print( i )
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
            if self.ttrend=='quadratic': pinit += [ 0 ]
            pfiti = scipy.optimize.fmin( CalcRMS, pinit, maxiter=1e4, xtol=1e-3, ftol=1e-4 )
            rms[i] = CalcRMS( pfiti )
            pfit += [ pfiti ]
        pbest = pfit[np.argmin(rms)]            
        a1, a2, a3, a4, a5 = pbest[:-nbase]
        rpars = [ a1, a2, a3, a4, a5 ]
        tfit, rfit = rfunc( thrs, torb, pbest )
        fluxc = flux/( tfit*rfit )
        return rpars, fluxc


    def RunTrials( self, ntrials ):
        """
        Fit the light curve model using multiple randomized starting parameter values.
        """
        ndsets = len( self.dsets )
        chi2 = np.zeros( ntrials )
        trials = []
        print( '\nTrials with randomly perturbed starting positions:' )
        for i in range( ntrials ):
            print( i+1, ntrials )
            for j in range( self.npar ):
                if i>0: # Note: first trial is unperturbed
                    if self.fixed[j]==0:
                        v = self.pars_init[j]
                        # Perturbations for planet parameters excluding delT:
                        if j<self.nppar-ndsets: 
                            dv = (1e-2)*np.random.randn()*np.abs( v )
                        elif j<self.nppar: # smaller perturbations for delT
                            dv = (1e-4)*np.random.randn()*np.abs( v )
                        elif j<self.npar-self.nbpar: # small perturbations for ramp parameters
                            dv = (1e-4)*np.random.randn()*np.abs( v )
                        else: # no perturbation for baseline parameters
                            dv = 0 
                        self.pars_init[j] = v + dv
            self.FitModel( save_to_file=False, verbose=False )
            chi2[i] = self.CalcChi2()
            trials += [ self.pars_fit['pvals'] ]
        return trials[np.argmin(chi2)]
    
    def PreFitting( self, niter=2, sigcut=10 ):
        """
        Performs an initial fit with 5x trial starting positions.
        Then uses the best fit as a starting location for subsequent
        trials, the best fit of which is used to flag outliers. This
        process is typically iterated multiple times, where the number 
        of iterations is specified as an attribute. At the final
        iteration, the measurement uncertainties are rescaled such
        that the best-fit model has a reduced chi^2 of 1. This is 
        done to get more realistic uncertainties for the model parameters
        in the final model fit, which is performed outside this routine.
        """
        print( '\nIterating over multiple trials to flag outliers:' )
        syspars = self.syspars
        tt = syspars['tr_type']
        finished = False
        nattempts = 0
        nattempts_max = 5
        while ( finished==False )*( nattempts<nattempts_max ):
            ixsd = self.data_ixs.copy()
            ixsm = self.keepixs.copy()
            ixsp = self.par_ixs
            batp = self.batpars
            #pdb.set_trace()
            data = self.data
            ncull = 0
            ndat = 0
            print( '\nRunning initial fit for full model:' )
            self.pars_init = self.RunTrials( 5 )
            print( 'Done.\n' )
            for g in range( niter ):
                pars_fit = self.RunTrials( self.ntrials )
                mfit = self.CalcModel( pars_fit )
                ffit = mfit['psignal']*mfit['ttrend']*mfit['ramp']
                ixsmg = {}
                for dset in self.dsets:
                    scandirs = self.wlcs[dset].scandirs[ixsm[dset]]
                    ixsmg['f'] = ixsm[dset][scandirs==1] #testing
                    ixsmg['b'] = ixsm[dset][scandirs==-1] #testing
                    ixsmz = [] #testing
                    for k in self.scankeys[dset]:
                        ixsdk = ixsd[dset][k]
                        ixsmzk = ixsmg[k] #testing
                        idkey = '{0}{1}'.format( dset, k )
                        residsk = self.data[ixsdk,3]-ffit[ixsdk]
                        uncsk = self.data[ixsdk,4]
                        nsig = np.abs( residsk )/uncsk                           
                        ixs = ( nsig<sigcut ) # within threshold
                        ncull += int( nsig.size-ixs.sum() )
                        self.pmodels[idkey] = batman.TransitModel( batp[idkey], \
                                                                   data[ixsdk,0][ixs], \
                                                                   transittype=tt )
                        ixsd[dset][k] = ixsdk[ixs]
                        ixsmz += [ ixsmzk[ixs] ] #testing
                        ndat += len( residsk )
                    ixsmz = np.concatenate( ixsmz ) #testing
                    ixs0 = np.argsort( ixsmz ) #testing
                    ixsm[dset] = ixsmz[ixs0] #testing
                print( 'Iteration={0:.0f}, Nculled={1:.0f}'.format( g+1, ncull ) )
                self.pars_init = pars_fit
            if ncull<0.1*ndat:
                finished = True
            else:
                warnstr = '\n{0:.0f}% of data points culled indicating something'\
                          .format( 100*float( ncull )/float( ndat ) )
                warnstr += ' went wrong. Re-attempting...'
                print( warnstr )
                nattempts += 1
        if ( finished==False )*( nattempts>nattempts_max ):
            print( '\nOver 10% of data flagged as outliers - something probably wrong' )
            print( 'Re-attempted {0:.0f} times but always had the same problem!\n'\
                   .format( nattempts_max ) )
            pdb.set_trace()
        print( '\n{0}\nRescaling measurement uncertainties by:\n'.format( 50*'#' ) )
        rescale = {}
        self.sigw = {}
        for dset in self.dsets:
            rescale[dset] = {}
            self.sigw[dset] = {}
            for k in self.scankeys[dset]:
                idkey = '{0}{1}'.format( dset, k )
                ixsdk = ixsd[dset][k]
                #ixsmk = ixsm[dset][k]
                zk = ( self.data[ixsdk,3]-ffit[ixsdk] )/self.data[ixsdk,4]
                chi2k = np.sum( zk**2. )
                rchi2k = chi2k/float( zk.size-len( pars_fit[ixsp[idkey]] ) )
                rescale[dset][k] = np.sqrt( rchi2k )
                self.sigw[dset][k] = np.median( self.data[ixsdk,4] )
                print( '{0:.2f} for {1}{2}'.format( rescale[dset][k], dset, k ) )
                if np.isfinite( rescale[dset][k] )==False: pdb.set_trace()
        for dset in self.dsets:
            for k in self.scankeys[dset]:
                self.data[ixsd[dset][k],4] *= rescale[dset][k]
        self.uncertainties_rescale = rescale
        print( '{0}\n'.format( 50*'#' ) )    
        self.model_fit = None # reset for main FitModel() run
        self.pars_fit = None # reset for main FitModel() run
        self.data_ixs = ixsd
        self.keepixs = ixsm
        return None


    def BestFitsOut( self ):
        jd = self.data[:,0]
        self.bestfits = {}
        nf = 500
        for dset in list( self.data_ixs.keys() ):
            self.bestfits[dset] = {}
            for k in list( self.data_ixs[dset].keys() ):
                idkey = '{0}{1}'.format( dset, k )
                ixsdk = self.data_ixs[dset][k]
                pfitdk = self.model_fit['psignal'][ixsdk] # planet signal
                tfitdk = self.model_fit['ttrend'][ixsdk] # baseline trend
                rfitdk = self.model_fit['ramp'][ixsdk] # ramp
                jddk = jd[ixsdk]
                jdfdk = np.linspace( jddk.min(), jddk.max(), nf )
                pmodfk = batman.TransitModel( self.batpars[idkey], jdfdk, \
                                              transittype=self.syspars['tr_type'] )
                pfitfdk = pmodfk.light_curve( self.batpars[idkey] )
                self.bestfits[dset][k] = {}
                self.bestfits[dset][k]['jd'] = jddk
                self.bestfits[dset][k]['psignal'] = pfitdk
                self.bestfits[dset][k]['ttrend'] = tfitdk
                self.bestfits[dset][k]['ramp'] = rfitdk
                self.bestfits[dset][k]['jdf'] = jdfdk
                self.bestfits[dset][k]['psignalf'] = pfitfdk
        return None
    
    def CalcChi2( self ):
        pfit = self.model_fit['psignal'] # planet signal
        sfit = self.model_fit['ttrend']*self.model_fit['ramp'] # systematics
        ffit = pfit*sfit
        chi2 = 0
        for dset in list( self.data_ixs.keys() ):
            for k in list( self.data_ixs[dset].keys() ):
                ixsdk = self.data_ixs[dset][k]
                residsk = ( self.data[ixsdk,3]-ffit[ixsdk] )
                uncsk = self.data[ixsdk,4]
                chi2 += np.sum( ( residsk/uncsk )**2. )
        return chi2

    
    def FitModel( self, save_to_file=False, verbose=True ):
        def NormDeviates( pars, fjac=None, data=None ):
            """
            Function defined in format required by mpfit.
            """
            m = self.CalcModel( pars )
            fullmodel = m['psignal']*m['ttrend']*m['ramp']
            resids = data[:,3]-fullmodel
            status = 0
            return [ status, resids/data[:,4] ]
        self.npar = len( self.par_labels )
        parinfo = []
        for i in range( self.npar ):
            parinfo += [ { 'value':self.pars_init[i], 'fixed':int( self.fixed[i] ), \
                           'parname':self.par_labels[i], \
                           'limited':[0,0], 'limits':[0.,0.] } ]
        fa = { 'data':self.data }
        m = mpfit( NormDeviates, self.pars_init, functkw=fa, parinfo=parinfo, \
                   maxiter=1e3, ftol=1e-5, quiet=True )
        if (m.status <= 0): print( 'error message = ', m.errmsg )
        self.pars_fit = { 'pvals':m.params, 'puncs':m.perror, 'pcov':m.covar, \
                          'ndof':m.dof, 'status':m.status }
        self.Tmids = {}
        for k in self.dsets:
            self.Tmids[k] = self.Tmid0[k] + m.params[self.delTixs[k]]
        self.model_fit = self.CalcModel( m.params )
        if save_to_file==True:
            self.Save()
            self.Plot()
        ostr = self.TxtOut( save_to_file=save_to_file )
        if verbose==True:
            print( ostr )
        return None

    def TxtOut( self, save_to_file=True ):
        ostr = ''
        ostr +=  '{0}\n# status = {1}'.format( 50*'#', self.pars_fit['status'] )
        try:
            ostr1 = '\n# Uncertainty rescaling factors:'
            for k in list( self.data_ixs.keys() ):
                ostr1 += '\n#  {0} = {1:.4f}'.format( k, self.uncertainties_rescale[k] )
            ostr += ostr1
        except:
            pass
        ostr += '\n# Fit results:\n#{0}'.format( 49*'-' )
        npar = len( self.fixed )
        pvals = self.pars_fit['pvals']
        puncs = self.pars_fit['puncs']
        for i in range( npar ):
            col1 = '{0}'.format( self.par_labels[i].rjust( 15 ) )
            col2 = '{0:20f}'.format( pvals[i] ).replace( ' ', '' ).rjust( 20 )
            col3 = '{0:20f}'.format( puncs[i] ).replace( ' ', '' ).ljust( 20 )
            if self.fixed[i]==0:
                ostr += '\n{0} = {1} +/- {2} (free)'.format( col1, col2, col3 )
            else:
                ostr += '\n{0} = {1} +/- {2} (fixed)'.format( col1, col2, col3 )
        ostr += '\n# Tmid assumed for each dataset:'
        for d in list( self.dsets ):
            ostr += '\n#  {0} = {1}'.format( d, self.Tmid0[d] )
        if save_to_file==True:
            ostr += '\n# Photon noise rescaling factors:'
            for d in list( self.dsets ):
                for k in self.scankeys[d]:
                    idkey = '{0}{1}'.format( d, k )
                    beta = self.uncertainties_rescale[d][k]
                    ostr += '\n#  {0} = {1:.2f}'.format( idkey, beta )
            ofile = open( self.whitefit_fpath_txt, 'w' )
            ofile.write( ostr )
            ofile.close()
        return ostr
    
    def Save( self ):
        """
        In terms of what still needs to be done to make this output
        be compatible with WFC3PrepSpeclcs():
          - create cullixs_final in proper format
          - create bestfits structure in proper format
        Next step = Look at what structure these have in a GP whitefit file.
        """
        outp = {}
        outp['wlcs'] = self.wlcs
        outp['analysis'] = self.analysis
        # vstacked data:
        outp['data_ixs'] = self.data_ixs
        outp['data'] = self.data
        outp['cullixs_init'] = self.cullixs
        outp['keepixs_final'] = self.keepixs
        outp['par_ixs'] = self.par_ixs
        outp['model_fit'] = self.model_fit
        self.BestFitsOut()
        outp['bestfits'] = self.bestfits
        outp['uncertainties_rescale'] = self.uncertainties_rescale
        outp['par_labels'] = self.par_labels
        outp['pars_fit'] = self.pars_fit
        outp['mle'] = {}
        for i in range( self.npar ):
            outp['mle'][self.par_labels[i]] = self.pars_fit['pvals'][i]
        outp['fixed'] = self.fixed
        outp['batpars'] = self.batpars
        outp['pmodels'] = self.pmodels
        outp['syspars'] = self.syspars
        outp['orbpars'] = { 'fittype':self.orbpars }
        if ( self.syspars['tr_type']=='primary' ):
            ix_aRs = self.par_labels=='aRs'
            ix_b = self.par_labels=='b'
            aRs = float( self.pars_fit['pvals'][ix_aRs] )
            b = float( self.pars_fit['pvals'][ix_b] )
        else:
            aRs = float( self.syspars['aRs'][0] )
            b = float( self.syspars['b'][0] )
        outp['orbpars']['aRs'] = aRs
        outp['orbpars']['b'] = b
        outp['orbpars']['incl'] = np.rad2deg( np.arccos( b/aRs ) )
        outp['Tmids'] = self.Tmids
        outp['Tmid0'] = self.Tmid0
        opath_pkl = self.GetFilePath()
        ofile = open( opath_pkl, 'wb' )
        pickle.dump( outp, ofile )
        ofile.close()
        opath_txt = opath_pkl.replace( '.pkl', '.txt' )
        self.whitefit_fpath_pkl = opath_pkl
        self.whitefit_fpath_txt = opath_txt
        # Write to the text file:
        self.TxtOut( save_to_file=True )
        print( '\nSaved:\n{0}\n{1}\n'.format( self.whitefit_fpath_pkl, \
                                              self.whitefit_fpath_txt ) )
        return None
    
               
    def Plot( self ):
        plt.ioff()
        self.UpdateBatpars( self.pars_fit['pvals'] )
        analysis = self.analysis
        #dsets = list( self.wlcs.keys() )
        nvisits = len( self.dsets )
        jd = self.data[:,0]
        thrs = self.data[:,1]
        flux = self.data[:,3]
        uncs = self.data[:,4]
        ixsd = self.data_ixs
        pfit = self.model_fit['psignal']
        sfit = self.model_fit['ramp']*self.model_fit['ttrend']
        ffit = pfit*sfit
        resids = flux-ffit
        cjoint = 'Cyan'#0.8*np.ones( 3 )
        label_fs = 12
        text_fs = 8
        for i in range( nvisits ):
            fig = plt.figure( figsize=[6,9] )
            xlow = 0.15
            axh12 = 0.35
            axh3 = axh12*0.5
            axw = 0.83
            ylow1 = 1-0.04-axh12
            ylow2 = ylow1-0.015-axh12
            ylow3 = ylow2-0.015-axh3
            ax1 = fig.add_axes( [ xlow, ylow1, axw, axh12 ] )
            ax2 = fig.add_axes( [ xlow, ylow2, axw, axh12 ], sharex=ax1 )
            ax3 = fig.add_axes( [ xlow, ylow3, axw, axh3 ], sharex=ax1 )
            scankeys = self.scankeys[self.dsets[i]]
            nscans = len( scankeys )
            tv = []
            tvf = []
            psignalf = []
            print( '\n\nResidual scatter:' )
            Tmidlit = self.Tmid0[self.dsets[i]]
            for s in range( len( self.scankeys[self.dsets[i]] ) ):
                k = self.scankeys[self.dsets[i]][s]
                idkey = '{0}{1}'.format( self.dsets[i], k )
                ixsik = ixsd[self.dsets[i]][k]
                if k=='f':
                    mfc = np.array( [217,240,211] )/256.
                    mec = np.array( [27,120,55] )/256.
                elif k=='b':
                    mfc = np.array( [231,212,232] )/256.
                    mec = np.array( [118,42,131] )/256.
                #jdfk = np.linspace( jd[ixsik].min(), jd[ixsik].max(), 300 )
                jdfk = self.bestfits[self.dsets[i]][k]['jdf']
                if self.syspars['tr_type']=='primary':
                    Tmid = self.batpars[idkey].t0
                elif self.syspars['tr_type']=='secondary':
                    Tmid = self.batpars[idkey].t_secondary
                else:
                    pdb.set_trace()
                tvk = 24*( jd[ixsik]-Tmid )
                tvfk = 24*( jdfk-Tmid )
                tv += [ tvk ]
                tvf += [ tvfk ]
                ax1.plot( tvk, 100*( flux[ixsik]-1 ), 'o', mec=mec, mfc=mfc )
                oixs = UR.SplitHSTOrbixs( thrs[ixsik] )
                norb = len( oixs )
                for j in range( norb ):
                    ax1.plot( tvk[oixs[j]], 100*( ffit[ixsik][oixs[j]]-1 ), \
                              '-', color=cjoint )
                if k==self.scankeys[self.dsets[i]][-1]:
                    delTmin = 24*60*( Tmidlit-Tmid )
                    lstr = 'predicted mid-time (delT={0:.2f}min)'.format( delTmin )
                    ax1.axvline( delTmin/60., ls='--', c='r', zorder=0, label=lstr )
                    ax1.legend( loc='lower right', bbox_to_anchor=[1,1.005] )
                ax2.plot( tvk, 100*( flux[ixsik]/sfit[ixsik]-1 ), 'o', mec=mec, mfc=mfc )
                pmodfk = batman.TransitModel( self.batpars[idkey], jdfk, \
                                              transittype=self.syspars['tr_type'] )
                #psignalfk = pmodfk.light_curve( self.batpars[idkey] )
                psignalfk = self.bestfits[self.dsets[i]][k]['psignalf']
                psignalf += [ psignalfk ]
                ax3.errorbar( tvk, (1e6)*resids[ixsik], \
                              yerr=(1e6)*uncs[ixsik], fmt='o', \
                              mec=mec, ecolor=mec, mfc=mfc )
                ax3.axhline( 0, ls='-', color=cjoint, zorder=0 )
                rms_ppm = (1e6)*( np.sqrt( np.mean( resids[ixsik]**2. ) ) )
                print( '  {0} = {1:.0f} ppm ({2:.2f}x photon noise)'\
                       .format( idkey, rms_ppm, \
                                self.uncertainties_rescale[self.dsets[i]][k] ) )
                sig0_ppm = (1e6)*self.sigw[self.dsets[i]][k]
                noisestr = '$\sigma_0$={0:.0f}ppm, rms={1:.0f}ppm'\
                           .format( sig0_ppm, rms_ppm )
                fig.text( xlow+0.01*axw, ylow2+( 0.02+0.05*s )*axh12, noisestr, \
                          fontsize=text_fs, rotation=0, color=mec, \
                          horizontalalignment='left', verticalalignment='bottom' )
                          
            tvf = np.concatenate( tvf )
            psignalf = np.concatenate( psignalf )
            ixs = np.argsort( tvf )
            ax2.plot( tvf[ixs], 100*( psignalf[ixs]-1 ), '-', color=cjoint, zorder=0 )
            plt.setp( ax1.xaxis.get_ticklabels(), visible=False )
            plt.setp( ax2.xaxis.get_ticklabels(), visible=False )
            titlestr = '{0}'.format( self.dsets[i] )
            fig.text( xlow+0.05*axw, ylow1+axh12*1.01, titlestr, rotation=0, fontsize=18, \
                      verticalalignment='bottom', horizontalalignment='left' )
            fig.text( xlow+0.5*axw, 0.005, 'Time from mid-transit (h)', \
                      rotation=0, fontsize=label_fs, \
                      verticalalignment='bottom', horizontalalignment='center' )
            fig.text( 0.01, ylow1+0.5*axh12, 'Flux change (%)', \
                      rotation=90, fontsize=label_fs, \
                      verticalalignment='center', horizontalalignment='left' )
            fig.text( 0.01, ylow2+0.5*axh12, 'Flux change (%)', \
                      rotation=90, fontsize=label_fs, \
                      verticalalignment='center', horizontalalignment='left' )
            fig.text( 0.01, ylow3+0.5*axh3, 'Residuals (ppm)', \
                      rotation=90, fontsize=label_fs, \
                      verticalalignment='center', horizontalalignment='left' )
            if self.syspars['tr_type']=='primary':
                fitstr = '$R_p/R_\star = {0:.5f} \pm {1:.5f}$'\
                         .format( self.pars_fit['pvals'][0], self.pars_fit['puncs'][0] )
            elif self.syspars['tr_type']=='secondary':
                fitstr = '$D = {0:.0f} \pm {1:.0f}$ ppm'\
                         .format( (1e6)*self.pars_fit['pvals'][0], \
                                  (1e6)*self.pars_fit['puncs'][0] )
            else:
                pdb.set_trace()
            fig.text( xlow+0.99*axw, ylow2+0.02*axh12, fitstr, \
                      fontsize=text_fs, rotation=0, color='Black', \
                      horizontalalignment='right', verticalalignment='bottom' )
            
            #fig.suptitle( ofigpath, fontsize=16 )
            opath = self.whitefit_fpath_pkl\
                    .replace( '.pkl', '.{0}.pdf'.format( self.dsets[i] ) )
            #ofigpath = os.path.basename( opath ).replace( '.pdf', '' )
            fig.savefig( opath )
            print( '\nSaved:\n{0}\n'.format( opath ) )
            plt.close()
        return None


    def GetODir( self ):
        dirbase = os.path.join( self.results_dir, 'white' )
        if self.orbpars=='free':
            dirbase = os.path.join( dirbase, 'orbpars_free' )
        elif self.orbpars=='fixed':
            dirbase = os.path.join( dirbase, 'orbpars_fixed' )
        else:
            print( '\n\n\norbpars must be "free" or "fixed"\n\n\n' )
            pdb.set_trace() # haven't implemented other cases yet
        if self.Tmid_free==True:
            dirbase = os.path.join( dirbase, 'Tmid_free' )
        else:
            dirbase = os.path.join( dirbase, 'Tmid_fixed' )
        if self.syspars['tr_type']=='primary':
            dirbase = os.path.join( dirbase, self.ld )
        else:
            dirbase = os.path.join( dirbase, 'ldoff' )
        #dsets = list( self.wlcs.keys() )
        dsets = UR.NaturalSort( self.dsets )
        dirext = ''
        for k in dsets:
            dirext += '+{0}'.format( k )
        dirext = dirext[1:]
        if len( dsets )>1:
            if self.syspars['tr_type']=='primary':
                if self.RpRs_shared==True:
                    dirext += '.RpRs_shared'
                else:
                    dirext += '.RpRs_individ'
            elif self.syspars['tr_type']=='secondary':
                if self.EcDepth_shared==True:
                    dirext += '.EcDepth_shared'
                else:
                    dirext += '.EcDepth_individ'
            else:
                pdb.set_trace()
        dirbase = os.path.join( dirbase, dirext )
        if self.akey=='':
            print( '\n\nMust set akey to create output folder for this particular analysis\n\n' )
            pdb.set_trace()
        else:
            self.odir = os.path.join( dirbase, self.akey )
        # Don't bother with the reduction parameters in the filenames.
        # That can be done separately with a custom routine defined by
        # the user if it's really important.
        return None


    def GetFilePath( self ):
        self.GetODir()
        if os.path.isdir( self.odir )==False:
            os.makedirs( self.odir )
        if self.beta_free==True:
            betastr = 'beta_free'
        else:
            betastr = 'beta_fixed'
        oname = 'white.{0}.mpfit.{1}base.pkl'.format( self.analysis, self.ttrend )
        opath = os.path.join( self.odir, oname )
        return opath
    

    
    def CalcModel( self, pars ):#, data, ixsp, ixsd, pmodels, batpars, Tmidlits ):
        """
        For a parameter array for a specific dataset, the parameters
        are *always* the following (at least for now):
           RpRs, aRs, b, ldcoeff1, ..., ldcoeffN, delT, a1, a2, a3, a4, a5, b1, b2, (b3)
        or:
           EcDepth, delT, a1, a2, a3, a4, a5, b1, b2, (b3)
        So you can always unpack these in this order and send them as
        inputs to their appropriate functions. 
        """
        if self.ttrend=='linear':
            rfunc = UR.DERampLinBase
        elif self.ttrend=='quadratic':
            rfunc = UR.DERampQuadBase
        else:
            pdb.set_trace()
        ndat, nvar = np.shape( self.data )
        batp = self.batpars
        pmod = self.pmodels
        psignal = np.zeros( ndat )
        ttrend = np.zeros( ndat )
        ramp = np.zeros( ndat )
        jd = self.data[:,0]
        thrs = self.data[:,1]
        torb = self.data[:,2]
        flux = self.data[:,3]
        uncs = self.data[:,4]
        #dsets = list( self.wlcs.keys() )
        ndsets = len( self.dsets )
        self.UpdateBatpars( pars )

        for i in range( ndsets ):
            dset = self.dsets[i]
            Tmid0k = self.Tmid0[dset]
            for k in self.scankeys[dset]:
                idkey = '{0}{1}'.format( dset, k )
                ixsdk = self.data_ixs[dset][k]
                parsk = pars[self.par_ixs[idkey]]
                if pmod[idkey].transittype==1:
                    if batp[idkey].limb_dark=='quadratic':
                        m = 2
                    elif batp[idkey].limb_dark=='nonlinear':
                        m = 4
                    else:
                        pdb.set_trace()
                    s = 4+m # RpRs, aRs, b, delT
                else:
                    s = 2 # EcDepth, delT
                psignal[ixsdk] = pmod[idkey].light_curve( batp[idkey] )
                # Evaluate the systematics signal:
                tfit, rfit = rfunc( thrs[ixsdk], torb[ixsdk], parsk[s:] )
                ttrend[ixsdk] = tfit
                ramp[ixsdk] = rfit
        return { 'psignal':psignal, 'ttrend':ttrend, 'ramp':ramp }

    def UpdateBatpars( self, pars ):
        batp = self.batpars
        pmod = self.pmodels
        dsets = list( self.wlcs.keys() )
        ndsets = len( self.dsets )
        for i in range( ndsets ):
            dset = self.dsets[i]
            Tmid0k = self.Tmid0[dset]
            for k in self.scankeys[dset]:
                idkey = '{0}{1}'.format( dset, k )
                ixsdk = self.data_ixs[dset][k]
                parsk = pars[self.par_ixs[idkey]]
                # Evaluate the planet signal:
                if pmod[idkey].transittype==1:
                    # Primary transits have RpRs, aRs, inc and optionally limb darkening:
                    batp[idkey].rp = parsk[0]
                    batp[idkey].a = parsk[1]
                    batp[idkey].inc = np.rad2deg( np.arccos( parsk[2]/batp[idkey].a ) )
                    if batp[idkey].limb_dark=='quadratic':
                        m = 2
                    elif batp[idkey].limb_dark=='nonlinear':
                        m = 4
                    else:
                        pdb.set_trace()
                    batp[idkey].u = parsk[3:3+m]
                    batp[idkey].t0 = Tmid0k + parsk[3+m]                    
                elif pmod[idkey].transittype==2:
                    # Secondary eclipses only have the eclipse depth and mid-time:
                    batp[idkey].fp = parsk[0]
                    batp[idkey].t_secondary = Tmid0k + parsk[1]
                else:
                    pdb.set_trace()
        self.batpars = batp
        return None
    
    def SetupLDPars( self ):
        #dsets = list( self.wlcs.keys() )
        ldkey = UR.GetLDKey( self.ld )
        if ldkey.find( 'nonlin' )>=0:
            self.ldbat = 'nonlinear'
            k = 'nonlin1d'
        elif ldkey.find( 'quad' )>=0:
            self.ldbat = 'quadratic'
            k = 'quad1d'
        else:
            pdb.set_trace()
        configs = []
        self.ldpars = {}
        for dset in self.dsets:
            configs += [ self.wlcs[dset].config ]
            self.ldpars[configs[-1]] = self.wlcs[dset].ld[k]
        #configs = list( np.unique( np.array( configs ) ) )
        return None

    def GetBatmanObject( self, jd, config ):
        # Define the batman planet object:
        batpar = batman.TransitParams()
        batpar.t0 = self.syspars['T0'][0]
        batpar.per = self.syspars['P'][0]
        batpar.rp = self.syspars['RpRs'][0]
        batpar.a = self.syspars['aRs'][0]
        try: # See if inclination has been provided directly
            batpar.inc = self.syspars['incl'][0]
        except: # otherwise, derive from impact parameter:
            b = self.syspars['b'][0]
            batpar.inc = np.rad2deg( np.arccos( b/batpar.a ) )
        batpar.ecc = self.syspars['ecc'][0]
        batpar.w = self.syspars['omega'][0]
        batpar.limb_dark = self.ldbat
        batpar.u = self.ldpars[config]
        if self.syspars['tr_type']=='secondary':
            batpar.fp = self.syspars['EcDepth']
            batpar.t_secondary = self.syspars['Tmid'][0]
        pmodel = batman.TransitModel( batpar, jd, transittype=self.syspars['tr_type'] )
        # Following taken from here:
        # https://www.cfa.harvard.edu/~lkreidberg/batman/trouble.html#help-batman-is-running-really-slowly-why-is-this
        # Hopefully it works... but fac==None it seems... not sure why?
        fac = pmodel.fac
        pmodel = batman.TransitModel( batpar, jd, fac=fac, \
                                      transittype=self.syspars['tr_type'] )
        return batpar, pmodel
    

class WFC3WhiteFitGP():
    """
    Main routines for setting up model:
      1. GenerateMBundle() ** called directly
      2. SetupLDPars()
      3. AddVisitMBundles()
      4. GPMBundle()
      5. GetModelComponents()
      6. GetBatmanObject()
      7. GPLogLike()
      8. GetEvalModel()
    """
    def __init__( self ):
        self.wlcs = None
        self.results_dir = ''
        self.akey = ''
        self.analysis = 'rdiff_zap'
        self.gpkernels = ''
        self.gpinputs = []
        self.scankeys = {}
        self.syspars = {}
        self.ld = ''
        self.ldbat = ''
        self.ldpars = []
        self.orbpars = ''
        self.beta_free = True
        self.Tmid0 = {}
        #self.batpar = {} # maybe have a dict of these for each dset
        #self.pmodel = {}
        self.lineartbase = {} # set True/False for each visit
        self.tr_type = ''
        self.prelim_fit = False
        self.ngroups = 5
        self.nwalkers = 100
        self.nburn1 = 100
        self.nburn2 = 250
        self.nsteps = 250
        self.RpRs_shared = True
        self.EcDepth_shared = True        

        
    def GenerateMBundle( self ):
        """
        TODO - Rename this GenerateGPMBundle()...
        This routine starts by defining parameters shared across
        multiple visits then calls the AddVisitMBundles() routine
        to define parameters specific to each visit.
        """
        self.dsets = list( self.wlcs.keys() )
        # Define the model parameters shared across all lightcurves:
        print( '\n{0}\nGenerating model parameters:'.format( 50*'#' ) )
        parents = {}
        self.initvals = {}
        self.mbundle = {}
        if self.orbpars=='free':
            aRs = pyhm.Uniform( 'aRs', lower=0, upper=100 )
            b = pyhm.Uniform( 'b', lower=0, upper=1 )
            self.mbundle.update( { 'aRs':aRs, 'b':b } )
            self.initvals.update( { 'aRs':self.syspars['aRs'][0], \
                                    'b':self.syspars['b'][0] } )
        elif self.orbpars=='fixed':
            aRs = self.syspars['aRs'][0]
            b = self.syspars['b'][0]
            self.mbundle.update( { 'aRs':aRs, 'b':b } )
        else:
            pdb.set_trace()
        parents.update( { 'aRs':aRs, 'b':b } )
        if ( self.syspars['tr_type']=='primary' ):
            if ( self.RpRs_shared==True ):
                RpRs = pyhm.Uniform( 'RpRs', lower=0, upper=1 )
                self.mbundle['RpRs'] = RpRs
                parents['RpRs'] = RpRs
                self.initvals['RpRs'] = self.syspars['RpRs'][0]
            else:
                pdb.set_trace() # have not implemented yet
            ldpars = self.SetupLDPars()
            parents.update( ldpars )
        if ( self.syspars['tr_type']=='secondary' ):
            if ( self.EcDepth_shared==True ):
                EcDepth = pyhm.Uniform( 'EcDepth', lower=0, upper=1 )
                self.mbundle['EcDepth'] = EcDepth
                parents['EcDepth'] = EcDepth
                self.initvals['EcDepth'] = self.syspars['EcDepth'][0]
            else:
                pdb.set_trace() # have not implemented yet                
        self.AddVisitMBundles( parents )
        print( '\nGlobal list of model parameters:' )
        for k in list( self.mbundle.keys() ):
            try:
                print( '{0} (free)'.format( self.mbundle[k].name.rjust( 30 ) ) )
            except:
                print( '{0}={1} (fixed)'.format( k, self.mbundle[k] ).rjust( 30 ) )
        return None
    
        
    def SetupLDPars( self ):
        #dsets = list( self.wlcs.keys() )
        ldkey = UR.GetLDKey( self.ld )
        if ldkey.find( 'nonlin' )>=0:
            self.ldbat = 'nonlinear'
            k = 'nonlin1d'
        elif ldkey.find( 'quad' )>=0:
            self.ldbat = 'quadratic'
            k = 'quad1d'
        else:
            pdb.set_trace()
        configs = []
        self.ldpars = {}
        for dset in self.dsets:
            configs += [ self.wlcs[dset].config ]
            self.ldpars[configs[-1]] = self.wlcs[dset].ld[k]
        configs = list( np.unique( np.array( configs ) ) )
        for c in configs:
            ldc = self.ldpars[c]
            gamk = [ 'gam1_{0}'.format( c ), 'gam2_{0}'.format( c ) ]
            ck = [ 'c1_{0}'.format( c ), 'c2_{0}'.format( c ), \
                   'c3_{0}'.format( c ), 'c4_{0}'.format( c ) ]
            if ( self.ld.find( 'free' )>=0 ):
                ldsig = 0.6
                if ( self.ldbat=='quadratic' ):
                    gam1 = pyhm.Gaussian( gamk[0], mu=ldc[0], sigma=ldsig )
                    gam2 = pyhm.Gaussian( gamk[1], mu=ldc[1], sigma=ldsig )
                    self.initvals.update( { gamk[0]:ldc[0], gamk[1]:ldc[1] } )
                    self.mbundle.update( { gamk[0]:gam1, gamk[1]:gam2 } )
                if ( self.ldbat=='nonlinear' ):
                    c1 = pyhm.Gaussian( ck[0], mu=ldc[0], sigma=ldsig )
                    c2 = pyhm.Gaussian( ck[1], mu=ldc[1], sigma=ldsig )
                    c3 = pyhm.Gaussian( ck[2], mu=ldc[2], sigma=ldsig )
                    c4 = pyhm.Gaussian( ck[3], mu=ldc[3], sigma=ldsig )
                    self.initvals.update( { ck[0]:ldc[0], ck[1]:ldc[1], \
                                            ck[2]:ldc[2], ck[3]:ldc[3] } )
                    self.mbundle.update( { ck[0]:c1, ck[1]:c2, ck[2]:c3, ck[3]:c4 } )
            elif ( self.ld.find( 'fixed' )>=0 ):
                if ( self.ldbat=='quadratic' ):
                    gam1, gam2 = ldc
                    self.mbundle.update( { gamk[0]:ldc[0], gamk[1]:ldc[1] } )
                elif ( self.ldbat=='nonlinear' ):
                    c1, c2, c3, c4 = ldc
                    self.mbundle.update( { ck[0]:ldc[0], ck[1]:ldc[1], \
                                           ck[2]:ldc[2], ck[3]:ldc[3] } )
            else:
                pdb.set_trace() # shouldn't happen
            if self.ldbat=='quadratic':
                ldpars = { 'gam1':gam1, 'gam2':gam2 }
            elif self.ldbat=='nonlinear':
                ldpars = { 'c1':c1, 'c2':c2, 'c3':c3, 'c4':c4 }
            else:
                pdb.set_trace()
        return ldpars

    def GetBatmanObject( self, jd, config ):
        # Define the batman planet object:
        batpar = batman.TransitParams()
        batpar.t0 = self.syspars['T0'][0]
        batpar.per = self.syspars['P'][0]
        batpar.rp = self.syspars['RpRs'][0]
        batpar.a = self.syspars['aRs'][0]
        try: # See if inclination has been provided directly
            batpar.inc = self.syspars['incl'][0]
        except: # otherwise, derive from impact parameter:
            b = self.syspars['b'][0]
            batpar.inc = np.rad2deg( np.arccos( b/batpar.a ) )
        batpar.ecc = self.syspars['ecc'][0]
        batpar.w = self.syspars['omega'][0]
        batpar.limb_dark = self.ldbat
        batpar.u = self.ldpars[config]
        if self.syspars['tr_type']=='secondary':
            batpar.fp = self.syspars['EcDepth']
            batpar.t_secondary = self.syspars['Tmid'][0]
        pmodel = batman.TransitModel( batpar, jd, transittype=self.syspars['tr_type'] )
        # Following taken from here:
        # https://www.cfa.harvard.edu/~lkreidberg/batman/trouble.html#help-batman-is-running-really-slowly-why-is-this
        # Hopefully it works... but fac==None it seems... not sure why?
        fac = pmodel.fac
        pmodel = batman.TransitModel( batpar, jd, fac=fac, \
                                      transittype=self.syspars['tr_type'] )
        return batpar, pmodel
    
        
    def GetTmid( self, j, ixsf0, ixsb0 ):
        if self.syspars['tr_type']=='primary':
            if ( ixsf0.sum()>0 )*( ixsb0.sum()>0 ):
                if self.batpars[j]['f'].t0!=self.batpars[j]['b'].t0:
                    pdb.set_trace()
                else:
                    tmid = self.batpars[j]['f'].t0
            elif ixsf0.sum()>0:
                tmid = self.batpars[j]['f'].t0
            else:
                tmid = self.batpars[j]['b'].t0
        elif self.syspars['tr_type']=='secondary':
            if ( ixsf0.sum()>0 )*( ixsb0.sum()>0 ):
                tmidf = self.batpars[j]['f'].t_secondary
                tmidb = self.batpars[j]['b'].t_secondary
                if tmidf!=tmidb:
                    pdb.set_trace()
                else:
                    tmid = tmidf
            elif ixsf0.sum()>0:
                tmid = self.batpars[j]['f'].t_secondary
            else:
                tmid = self.batpars[j]['b'].t_secondary
        return tmid

        
    def EvalPsignalPrimary( self, jd, parents, batpar, pmodel, Tmid0 ):
        batpar.rp = parents['RpRs']
        batpar.t0 = Tmid0 + parents['delT']
        batpar.a = parents['aRs']
        batpar.inc = np.rad2deg( np.arccos( parents['b']/parents['aRs'] ) )
        if batpar.limb_dark=='quadratic':
            ldpars = np.array( [ parents['gam1'], parents['gam2'] ] )
        elif batpar.limb_dark=='nonlinear':
            ldpars = np.array( [ parents['c1'], parents['c2'], \
                                 parents['c3'], parents['c4'] ] )
        batpar.u = ldpars
        psignal = pmodel.light_curve( batpar )
        return psignal

    
    def EvalPsignalSecondary( self, jd, parents, batpar, pmodel, Tmid0 ):
        batpar.fp = parents['EcDepth']
        batpar.t_secondary = Tmid0 + parents['delT']
        batpar.a = parents['aRs']
        batpar.inc = np.rad2deg( np.arccos( parents['b']/parents['aRs'] ) )
        
        psignal = pmodel.light_curve( batpar )
        return psignal

    
    def AddVisitMBundles( self, parents ):
        """
        Before calling this routine, any shared parameters have been defined.
        This routine then defines parameters specific to each visit, including
        parameters for the planet signal and systematics.
        """
        self.Tmid0 = {}
        self.evalmodels = {}
        self.keepixs_final = {} # todo = change to keepixs_final so consistent w/speclcs
        #dsets = list( self.wlcs.keys() )
        nvisits = len( self.dsets )
        for j in range( nvisits ):
            k = self.dsets[j]
            parentsk = parents.copy()
            config = self.wlcs[k].config
            delTlab = 'delT_{0}'.format( k )
            delTk = pyhm.Uniform( delTlab, lower=-0.3, upper=0.3 )
            self.mbundle[delTlab] = delTk
            self.initvals[delTlab] = 0
            parentsk['delT'] = delTk
            jd = self.wlcs[k].jd
            Tmidk = self.syspars['Tmid'][0]
            while Tmidk<jd.min():
                Tmidk += self.syspars['P'][0]
            while Tmidk>jd.max():
                Tmidk -= self.syspars['P'][0]
            if ( Tmidk<jd.min() )+( Tmidk>jd.max() ):
                pdb.set_trace() # mid-time outside data range
            self.Tmid0[k] = [ Tmidk, delTlab ]
            if ( self.syspars['tr_type']=='primary' ):
                if self.RpRs_shared==False:
                    RpRslab = 'RpRs_{0}'.format( self.wlcs[k].dsetname )
                    RpRs = pyhm.Uniform( RpRslab, lower=0, upper=1 )
                    self.mbundle[RpRslab] = RpRs
                    parentsk['RpRs'] = RpRs
                    self.initvals[RpRslab] = self.syspars['RpRs'][0]
            elif ( self.syspars['tr_type']=='secondary' ):
                if self.EcDepth_shared==False:
                    EcDepthlab = 'EcDepth_{0}'.format( self.wlcs[k].dsetname )
                    EcDepth = pyhm.Uniform( EcDepthlab, lower=0, upper=1 )
                    self.mbundle[EcDepthlab] = EcDepth
                    parentsk['EcDepth'] = EcDepth
                    self.initvals[EcDepthlab] = self.syspars['EcDepth'][0]
            else:
                pdb.set_trace() # shouldn't happen
            self.GPMBundle( k, parentsk, Tmidk )
        return None

    
    def GPMBundle( self, dset, parents, Tmid ):
        wlc = self.wlcs[dset]
        ixsc = self.cullixs[dset]         
        scanixs = {}
        scanixs['f'] = ixsc[wlc.scandirs[ixsc]==1]
        scanixs['b'] = ixsc[wlc.scandirs[ixsc]==-1]
        keepixs_final = []
        for k in self.scankeys[dset]:
            self.GetModelComponents( dset, parents, scanixs, k, Tmid )
            keepixs_final += [ self.evalmodels[dset][k][1] ]
        keepixs_final = np.concatenate( keepixs_final )
        ixs = np.argsort( keepixs_final )
        self.keepixs_final[dset] = keepixs_final[ixs]
        return None
    
        
    def GetModelComponents( self, dset, parents, scanixs, scankey, Tmid ):
        """
        Takes planet parameters in pars0, which have been defined separately
        to handle variety of cases with separate/shared parameters across
        visits etc. Then defines the systematics model for this visit+scandir
        combination, including the log-likelihood function. Returns complete 
        mbundle for current visit, with initvals and evalmodel.
        """
        self.evalmodels[dset] = {}
        wlc = self.wlcs[dset]
        #idkey = '{0}{1}'.format( wlc.dsetname, scankey ) # this was what it was...
        idkey = '{0}{1}'.format( dset, scankey ) # haven't tested yet but should be right?
        gpinputs = self.gpinputs[dset]
        gpkernel = self.gpkernels[dset]
        ixs, pfit0 = self.PolyFitCullixs( dset, Tmid, scanixs[scankey] )
        betalabel = 'beta_{0}'.format( idkey )
        if self.beta_free==True:
            parents['beta'] = pyhm.Gaussian( betalabel, mu=1.0, sigma=0.2 )
            self.initvals[betalabel] = 1.0
        else:
            parents['beta'] = 1
        self.mbundle[betalabel] = parents['beta']
        if self.syspars['tr_type']=='primary':
            RpRsk = parents['RpRs'].name
            self.initvals[RpRsk] = self.syspars['RpRs'][0]
        elif self.syspars['tr_type']=='secondary':
            EcDepthk = parents['EcDepth'].name
            self.initvals[EcDepthk] = self.syspars['EcDepth'][0]
        else:
            pdb.set_trace()
        if self.orbpars=='free':
            self.initvals['aRs'] = self.syspars['aRs'][0]
            self.initvals['b'] = self.syspars['b'][0]
        batpar, pmodel = self.GetBatmanObject( wlc.jd[ixs], wlc.config )
        z = self.GPLogLike( dset, parents, batpar, pmodel, Tmid, ixs, idkey )
        loglikename = 'loglike_{0}'.format( idkey )
        self.mbundle[loglikename] = z['loglikefunc']
        self.mbundle[loglikename].name = loglikename
        evalmodelfunc = self.GetEvalModel( z, batpar, pmodel, Tmid )
        self.evalmodels[dset][scankey] = [ evalmodelfunc, ixs ]
        return None

    
    def GetEvalModel( self, z, batpar, pmodel, Tmid0 ):
        tr_type = self.syspars['tr_type']
        k = z['parlabels']
        def EvalModel( fitvals ):
            nf = 500
            jdf = np.linspace( z['jd'].min(), z['jd'].max(), nf )
            tvf = np.linspace( z['tv'].min(), z['tv'].max(), nf )
            ttrendf = fitvals[k['a0']] + fitvals[k['a1']]*tvf
            ttrend = fitvals[k['a0']] + fitvals[k['a1']]*z['tv']
            if self.orbpars=='free':
                batpar.a = fitvals[k['aRs']]
                batpar.inc = np.rad2deg( np.arccos( fitvals[k['b']]/batpar.a ) )
            if tr_type=='primary':
                batpar.rp = fitvals[k['RpRs']]
                batpar.t0 = Tmid0 + fitvals[k['delT']]
                if ( self.ld.find( 'quad' )>=0 )*( self.ld.find( 'free' )>=0 ):
                    ldpars = np.array( [ fitvals[k['gam1']], fitvals[k['gam2']] ] )
                    batpar.u = ldpars
            elif tr_type=='secondary':
                batpar.fp = fitvals[k['EcDepth']]
                batpar.t_secondary = Tmid0 + fitvals[k['delT']]
            
            pmodelf = batman.TransitModel( batpar, jdf, transittype=tr_type )
            fac = pmodelf.fac
            pmodelf = batman.TransitModel( batpar, jdf, transittype=tr_type, \
                                           fac=fac )
            psignalf = pmodelf.light_curve( batpar )
            psignal = pmodel.light_curve( batpar )
            resids = z['flux']/( psignal*ttrend )-1. # model=psignal*ttrend*(1+GP)
            
            gp = z['zgp']['gp']
            Alabel = z['zgp']['Alabel_global']
            logiLlabels = z['zgp']['logiLlabels_global']
            logiL = []
            for i in logiLlabels:
                logiL += [ fitvals[i] ]
            iL = np.exp( np.array( logiL ) )
            gp.cpars = { 'amp':fitvals[Alabel], 'iscale':iL }
            # Currently the GP(t) baseline is hacked in; may be possible to improve:
            if 'Alabel_baset' in z['zgp']:
                pdb.set_trace() # this probably needs to be updated
                Alabel_baset = z['zgp']['Alabel_baset']
                iLlabel_baset = z['zgp']['iLlabel_baset']
                gp.cpars['amp_baset'] = fitvals[Alabel_baset]
                gp.cpars['iscale_baset'] = fitvals[iLlabel_baset]
            if self.beta_free==True:
                beta = fitvals[k['beta']]
            else:
                beta = 1
            gp.etrain = z['uncs']*beta
            gp.dtrain = np.reshape( resids, [ resids.size, 1 ] )
            mu, sig = gp.predictive( xnew=gp.xtrain, enew=gp.etrain )
            systematics = ttrend#+mu.flatten()#*( mu.flatten() + 1 )
            bestfits = { 'psignal':psignal, 'ttrend':ttrend, 'mu':mu.flatten(), \
                         'jdf':jdf, 'psignalf':psignalf, 'ttrendf':ttrendf }
            zout = { 'psignal':psignal, 'ttrend':ttrend, 'mu':mu.flatten(), \
                     'jdf':jdf, 'psignalf':psignalf, 'ttrendf':ttrendf }
            return { 'arrays':zout, 'batpar':batpar, 'pmodel':pmodel }
        return EvalModel
                
    
    def GetGPLogLikelihood( self, jd, flux, uncs, tv, parents, zgp, \
                            batpar, pmodel, Tmid0, lineartbase ):
        if lineartbase==True:
            ttrend = parents['a0'] + parents['a1']*tv#[ixs]
        else:
            ttrend = parents['a0']
        if self.syspars['tr_type']=='primary':
            if batpar.limb_dark=='quadratic':
                batpar.u = np.array( [ parents['gam1'], parents['gam2'] ] )
            elif batpar.limb_dark=='nonlinear':
                batpar.u = np.array( [ parents['c1'], parents['c2'], \
                                       parents['c3'], parents['c4'] ] )
            psignal = self.EvalPsignalPrimary( jd, parents, batpar, pmodel, Tmid0 )
        elif self.syspars['tr_type']=='secondary':
            psignal = self.EvalPsignalSecondary( jd, parents, batpar, pmodel, Tmid0 )
        else:
            pdb.set_trace()
        #resids = flux - psignal*ttrend
        resids = flux/( psignal*ttrend )-1. # model=psignal*ttrend*(1+GP)
        logiL = []
        for i in zgp['logiLlabels_local']:
            logiL += [ parents[i] ]
        iL = np.exp( np.array( logiL ) )
        gp = zgp['gp']
        gp.cpars = { 'amp':parents[zgp['Alabel_local']], 'iscale':iL }
        if 'Alabel_baset' in zgp:
            gp.cpars['amp_baset'] = parents[zgp['Alabel_baset']]
            gp.cpars['iscale_baset'] = parents[zgp['iLlabel_baset']]
        gp.etrain = uncs*parents['beta']
        gp.dtrain = np.reshape( resids, [ resids.size, 1 ] )
        logp_val = gp.logp_builtin()
        return logp_val

    
    def GPLogLike( self, dset, parents, batpar, pmodel, Tmid0, ixs, idkey ):
        wlc = self.wlcs[dset]
        config = wlc.config
        jd = wlc.jd[ixs]
        tv = wlc.whitelc[self.analysis]['auxvars']['tv'][ixs]
        flux = wlc.whitelc[self.analysis]['flux'][ixs]
        uncs = wlc.whitelc[self.analysis]['uncs'][ixs]
        lintcoeffs = UR.LinTrend( jd, tv, flux )
        ldbat = self.ldbat
        #pars = {}
        #initvals = {}
        a0k = 'a0_{0}'.format( idkey )
        parents['a0'] = pyhm.Uniform( a0k, lower=0.5, upper=1.5 )
        self.mbundle[a0k] = parents['a0']
        self.initvals[a0k] = lintcoeffs[0]
        #initvals[a0k] = 1 # DELETE? REVERT?
        if self.lineartbase[dset]==True:
            a1k = 'a1_{0}'.format( idkey )
            parents['a1'] = pyhm.Uniform( a1k, lower=-0.1, upper=0.1 )
            self.mbundle[a1k] = parents['a1']
            self.initvals[a1k] = lintcoeffs[1]
            #initvals[a1k] = 0 # DELETE? REVERT?
        zgp = self.PrepGP( dset, ixs, idkey )        
        for k in list( zgp['gpvars'].keys() ):
            parents[k] = zgp['gpvars'][k]
        n0 = 30
        print( 'Model parameters for {0}'.format( dset ).center( 2*n0+1 ) )
        print( '{0} {1}'.format( 'Local'.rjust( n0 ),'Global'.rjust( n0 ) ) )
        for k in list( parents.keys() ):
            try:
                print( '{0} {1} (free)'\
                       .format( k.rjust( n0 ), parents[k].name.rjust( n0 ) ) )
            except:
                print( '{0} {1} (fixed)'.format( k.rjust( n0 ), k.rjust( n0 ) ) )
        @pyhm.stochastic( observed=True )
        def loglikefunc( value=flux, parents=parents ):
            def logp( value, parents=parents ):
                logp_val = self.GetGPLogLikelihood( jd, flux, uncs, tv, parents, \
                                                    zgp, batpar, pmodel, Tmid0, \
                                                    self.lineartbase[dset] )
                return logp_val
        for k in list( zgp['gpvars'].keys() ):
            l = zgp['gpvars'][k].name
            self.mbundle[l] = zgp['gpvars'][k]
            self.initvals[l] = zgp['gpinitvals'][k]
        parlabels = {}
        for k in list( parents.keys() ):
            try:
                parlabels[k] = parents[k].name
            except:
                pass
        #zout = { 'pars':pars, 'initvals':initvals, 'loglikefunc':loglikefunc, \
        #         'batpar':batpar, 'pmodel':pmodel, 'jd':jd, 'tv':tv, \
        #         'flux':flux, 'uncs':uncs, 'parlabels':parlabels, 'zgp':zgp }
        zout = { 'loglikefunc':loglikefunc, 'batpar':batpar, 'pmodel':pmodel, \
                 'jd':jd, 'tv':tv, 'flux':flux, 'uncs':uncs, \
                 'parlabels':parlabels, 'zgp':zgp }
        return zout
    
    def PrepGP( self, dset, ixs, idkey ):

        gp = gp_class.gp( which_type='full' )
        gp.mfunc = None
        gp.cfunc = self.gpkernels[dset]
        gp.mpars = {}

        #auxvars = wlc.whitelc[analysis]['auxvars']
        cond1 = ( gp.cfunc==kernels.sqexp_invL_ard )
        cond2 = ( gp.cfunc==kernels.matern32_invL_ard )
        cond3 = ( gp.cfunc==kernels.sqexp_invL )
        cond4 = ( gp.cfunc==kernels.matern32_invL )
        cond5 = ( gp.cfunc==Systematics.custom_kernel_sqexp_invL_ard )
        cond6 = ( gp.cfunc==Systematics.custom_kernel_mat32_invL_ard )
        cond7 = ( gp.cfunc==kernels.sqexp_ard )
        cond8 = ( gp.cfunc==kernels.matern32_ard )
        if cond1+cond2+cond3+cond4: # implies logiL_prior==True
            #z = PrepGP_invL( gp, self.gpinputs[dset], self.auxvars, ixs, idkey )
            z = self.GPinvL( dset, gp, ixs, idkey )
        elif cond5+cond6: # implieslogiL_prior==True
            z = self.GPinvLbaset( dset, gp, ixs, idkey )
            #pdb.set_trace() # todo PrepGP_ard( gp, auxvars, idkey )
        elif cond7+cond8: # implieslogiL_prior==False also
            pdb.set_trace() # todo PrepGP_ard( gp, auxvars, idkey )
        return z
        
    def GPinvL( self, dset, gp, ixs, idkey ):
        """
        Define GP parameterized in terms of inverse correlation length
        scales. Although it's not tested, this routine is designed to
        handle 1D or ND input variables.
        """ 
        gpvars = {}
        gpinitvals = {}
        Alabel_global = 'Amp_{0}'.format( idkey )        
        gpvars['Amp'] = pyhm.Gamma( Alabel_global, alpha=1, beta=1e2 )
        #gpvars[Alabel] = pyhm.Uniform( Alabel, lower=0, upper=1 )
        gpinitvals['Amp'] = 1e-5
        xtrain = []
        logiLlabels_global = []
        logiLlabels_local = []
        for i in self.gpinputs[dset]:
            #v = auxvars[gpinputs[k]]
            k, label = UR.GetVarKey( i )
            #v = auxvars[k]
            v = self.wlcs[dset].whitelc[self.analysis]['auxvars'][k]
            #ext = '{0}_{1}'.format( label, idkey )
            vs = ( v-np.mean( v ) )/np.std( v )
            #logiLlabel = 'logiL{0}'.format( ext )
            #labeli = ''
            pname = 'logiL{0}'.format( label )
            mlabel = '{0}_{1}'.format( pname, idkey )
            gpvari = UR.DefineLogiLprior( vs[ixs], i, mlabel, \
                                          priortype='uniform' )
            gpvars[pname] = gpvari
            logiLlow = gpvars[pname].parents['lower']
            logiLupp = gpvars[pname].parents['upper']
            gpinitvals[pname] = 1e-6# 0.5*(logiLlow+logiLupp)#-1e-8#iLlow + 0.3*( iLupp-iLlow )
            xtrain += [ vs[ixs] ]
            logiLlabels_global += [ mlabel ]
            logiLlabels_local += [ pname ]
        gp.xtrain = np.column_stack( xtrain )
        #zout = { 'gp':gp, 'gpvars':gpvars, 'gpinitvals':gpinitvals, \
        #         'Alabel':Alabel, 'logiLlabels':logiLlabels }
        zout = { 'gp':gp, 'gpvars':gpvars, 'gpinitvals':gpinitvals, \
                 'Alabel_global':Alabel_global, 'Alabel_local':'Amp', \
                 'logiLlabels_global':logiLlabels_global, \
                 'logiLlabels_local':logiLlabels_local }
        return zout

    def GPinvLbaset( self, dset, gp, ixs, idkey ):
        # todo = Should take Alabel and make an Alabel_baset along with
        # iLlabel_baset and return as output. Adapt from GPinvL().
        gpvars = {}
        gpinitvals = {}
        Alabel = 'Amp_{0}'.format( idkey )
        Alabel_baset = 'Amp_baset_{0}'.format( idkey )
        iLlabel_baset = 'iL_baset_{0}'.format( idkey )
        gpvars[Alabel] = pyhm.Gamma( Alabel, alpha=1, beta=1e2 )
        #gpvars[Alabel] = pyhm.Uniform( Alabel, lower=0, upper=1 )
        gpvars[Alabel_baset] = pyhm.Gamma( Alabel_baset, alpha=1, beta=1e3 )
        gpvars[iLlabel_baset] = pyhm.Uniform( iLlabel_baset, lower=0, upper=2 )
        gpinitvals[Alabel] = 1e-5
        gpinitvals[Alabel_baset] = 5e-4
        gpinitvals[iLlabel_baset] = 0.2
        tv = self.wlcs[dset].tv
        xtrain = [ tv[ixs] ]
        logiLlabels = []
        for i in self.gpinputs[dset]:
            #v = auxvars[gpinputs[k]]
            k, label = UR.GetVarKey( i )
            #v = auxvars[k]
            v = self.wlcs[dset].whitelc[self.analysis]['auxvars'][k][ixs]
            ext = '{0}_{1}'.format( label, idkey )
            vs = ( v-np.mean( v ) )/np.std( v )
            logiLlabel = 'logiL{0}'.format( ext )
            gpvari = UR.DefineLogiLprior( vs, i, logiLlabel, \
                                          priortype='uniform' )
            gpvars[logiLlabel] = gpvari
            logiLlow = gpvars[logiLlabel].parents['lower']
            logiLupp = gpvars[logiLlabel].parents['upper']
            #gpinitvals[logiLlabel] = 1e-5#0.5*(logiLlow+logiLupp)#-1e-8#iLlow + 0.3*( iLupp-iLlow )
            gpinitvals[logiLlabel] = 0.5*(logiLlow+logiLupp)#-1e-8#iLlow + 0.3*( iLupp-iLlow )
            xtrain += [ vs[ixs] ]
            logiLlabels += [ logiLlabel ]
        gp.xtrain = np.column_stack( xtrain )
        zout = { 'gp':gp, 'gpvars':gpvars, 'gpinitvals':gpinitvals, \
                 'Alabel':Alabel, 'logiLlabels':logiLlabels, \
                 'Alabel_baset':Alabel_baset, 'iLlabel_baset':iLlabel_baset }
        return zout

    
    def PolyFitCullixs( self, dset, Tmid, ixs ):
        """
        Quick polynomial systematics model fit to identify remaining outliers.
        Note that ixs contains the indices for a single scan direction after
        removing the pre-defined data points defined by cullixs. So the aim
        of this routine is to identify additional statistical outliers.
        """
        wlc = self.wlcs[dset]
        syspars = self.syspars
        jd = wlc.jd[ixs]
        #tv = wlc.tv[ixs]
        phi = wlc.whitelc[self.analysis]['auxvars']['hstphase'][ixs]
        tv = wlc.whitelc[self.analysis]['auxvars']['tv'][ixs]
        x = wlc.whitelc[self.analysis]['auxvars']['wavshift_pix'][ixs]
        phiv = ( phi-np.mean( phi ) )/np.std( phi )
        xv = ( x-np.mean( x ) )/np.std( x )
        ndat = tv.size
        offset = np.ones( ndat )
        
        B = np.column_stack( [ offset, tv, xv, phiv, phiv**2., phiv**3., phiv**4. ] )
        flux = wlc.whitelc[self.analysis]['flux'][ixs]
        uncs = wlc.whitelc[self.analysis]['uncs'][ixs]
        batpar, pmodel = self.GetBatmanObject( jd, wlc.config )
        ntrials = 15        
        if self.syspars['tr_type']=='primary':
            batpar.limb_dark = 'quadratic'
            batpar.u = wlc.ld['quad1d']
            zstart = self.PolyFitPrimary( batpar, pmodel, Tmid, B, flux, uncs, ntrials )
        elif self.syspars['tr_type']=='secondary':
            zstart = self.PolyFitSecondary( batpar, pmodel, Tmid, B, flux, uncs, ntrials )
        else:
            pdb.set_trace()
        pinit, parkeys, mod_eval, neglogp = zstart
        pfits = []
        logps = np.zeros( ntrials )
        print( '\nRunning quick outlier cull for dataset {0}...'.format( dset ) )
        for i in range( ntrials ):
            pfiti = scipy.optimize.fmin( neglogp, pinit[i,:], xtol=1e-5, \
                                         disp=False, ftol=1e-5, \
                                         maxfun=10000, maxiter=10000 )
            pfits += [ pfiti ]
            logps[i] = -neglogp( pfiti )
        pfit = pfits[np.argmax( logps )]
        psignal, polyfit = mod_eval( pfit )
        mfit = psignal*polyfit
        nsig = np.abs( flux-mfit )/uncs
        ixskeep = ixs[nsig<=5]
        self.nculled_poly = len( ixs )-len( ixskeep )
        if self.nculled_poly>0:
            print( 'Culled {0:.0f} outliers'.format( self.nculled_poly ) )
        else:
            print( 'No outliers culled' )
        pfitdict = {}
        for i in range( len( parkeys ) ):
            pfitdict[parkeys[i]] = pfit[i]        
        return ixskeep, pfitdict

    def PolyFitPrimary( self, batpar, pmodel, Tmid, B, flux, uncs, ntrials ):
        ndat = flux.size
        delT0 = ( 2.*np.random.random( ntrials )-1. )/24.
        RpRs0 = self.syspars['RpRs'][0]*( 1+0.1*np.random.randn( ntrials ) )
        if self.orbpars=='free':
            aRs0 = self.syspars['aRs'][0] + np.zeros( ntrials )
            b0 = self.syspars['b'][0] + np.zeros( ntrials )
            aRsp = self.syspars['aRs']
            bp = self.syspars['b']
            parkeys = [ 'RpRs', 'delT', 'aRs', 'b' ]
            def mod_eval( pars ):
                batpar.rp = pars[0]
                batpar.t0 = Tmid + pars[1]
                batpar.a = pars[2]
                batpar.inc = np.rad2deg( np.arccos( pars[3]/batpar.a ) )
                psignal = pmodel.light_curve( batpar )
                fluxc = flux/psignal
                coeffs = np.linalg.lstsq( B, fluxc, rcond=None )[0]
                polyfit = np.dot( B, coeffs )
                return psignal, polyfit
            def neglogp( pars ):
                psignal, polyfit = mod_eval( pars )
                resids = flux-psignal*polyfit
                llike = UR.MVNormalWhiteNoiseLogP( resids, uncs, ndat )
                aRslp = UR.NormalLogP( pars[2], aRsp[0], aRsp[1] )
                blp = UR.NormalLogP( pars[3], bp[0], bp[1] )
                return -( llike+aRslp+blp )
            pinit = np.column_stack( [ RpRs0, delT0, aRs0, b0 ] )
        elif ( self.orbpars=='fixed' ):
            batpar.a = self.syspars['aRs'][0]
            batpar.inc = np.rad2deg( np.arccos( self.syspars['b'][0]/batpar.a ) )
            parkeys = [ 'RpRs', 'delT' ]
            def mod_eval( pars ):
                batpar.rp = pars[0]
                batpar.t0 = Tmid + pars[1]
                psignal = pmodel.light_curve( batpar )
                fluxc = flux/psignal
                coeffs = np.linalg.lstsq( B, fluxc, rcond=None )[0]
                polyfit = np.dot( B, coeffs )
                return psignal, polyfit
            def neglogp( pars ):
                psignal, polyfit = mod_eval( pars )
                resids = flux-psignal*polyfit
                return -UR.MVNormalWhiteNoiseLogP( resids, uncs, ndat )
            pinit = np.column_stack( [ RpRs0, delT0 ] )
        else:
            pdb.set_trace() # need to work out when to install aRs, b for wmeanfixed
        return pinit, parkeys, mod_eval, neglogp
    
    def PolyFitSecondary( self, batpar, pmodel, Tmid, B, flux, uncs, ntrials ):
        ndat = flux.size
        rperturb = np.random.random( ntrials )
        delT0 = ( rperturb-0.5 )/24.
        EcDepth0 = self.syspars['EcDepth'][0]*( 1+0.1*np.random.randn( ntrials ) )
        if self.orbpars=='free':
            aRs0 = self.syspars['aRs'][0] + np.zeros( ntrials )
            b0 = self.syspars['b'][0] + np.zeros( ntrials )
            aRsp = self.syspars['aRs']
            bp = self.syspars['b']
            parkeys = [ 'EcDepth', 'delT', 'aRs', 'b' ]
            def mod_eval( pars ):
                batpar.fp = pars[0]
                batpar.t_secondary = Tmid + pars[1]
                batpar.a = pars[2]
                batpar.inc = np.rad2deg( np.arccos( pars[3]/batpar.a ) )
                psignal = pmodel.light_curve( batpar )
                fluxc = flux/psignal
                coeffs = np.linalg.lstsq( B, fluxc, rcond=None )[0]
                polyfit = np.dot( B, coeffs )
                return psignal, polyfit
            def neglogp( pars ):
                psignal, polyfit = mod_eval( pars )
                resids = flux-psignal*polyfit
                llike = UR.MVNormalWhiteNoiseLogP( resids, uncs, ndat )
                aRslp = UR.NormalLogP( pars[2], aRsp[0], aRsp[1] )
                blp = UR.NormalLogP( pars[3], bp[0], bp[1] )
                return -( llike+aRslp+blp )
            pinit = np.column_stack( [ EcDepth0, delT0, aRs0, b0 ] )
        elif ( self.orbpars=='fixed' ):
            batpar.a = self.syspars['aRs'][0]
            batpar.inc = np.rad2deg( np.arccos( self.syspars['b'][0]/batpar.a ) )
            parkeys = [ 'EcDepth', 'delT' ]
            def mod_eval( pars ):
                batpar.fp = pars[0]
                batpar.t_secondary = Tmid + pars[1]
                psignal = pmodel.light_curve( batpar )
                fluxc = flux/psignal
                coeffs = np.linalg.lstsq( B, fluxc, rcond=None )[0]
                polyfit = np.dot( B, coeffs )
                return psignal, polyfit
            def neglogp( pars ):
                psignal, polyfit = mod_eval( pars )
                resids = flux-psignal*polyfit
                return -UR.MVNormalWhiteNoiseLogP( resids, uncs, ndat )
            pinit = np.column_stack( [ EcDepth0, delT0 ] )
        else:
            pdb.set_trace() # need to work out when to install aRs, b for wmeanfixed
        return pinit, parkeys, mod_eval, neglogp
    
    
    def RunMLE( self ):
        if self.prelim_fit==True:
            mp = pyhm.MAP( self.mbundle )
            for k in list( self.initvals.keys() ):
                mp.model.free[k].value = self.initvals[k]
            print( '\nRunning MLE fit...' )
            print( '\nFree parameters: name, value, parents, logprior' )
            for k in mp.model.free.keys():
                print( k, mp.model.free[k].value, mp.model.free[k].parents, \
                       mp.model.free[k].logp() )
            print( '\noptmising...' )
            mp.fit( xtol=1e-5, ftol=1e-5, maxfun=10000, maxiter=10000 )
            print( 'Done.' )
            print( '\nMLE results:' )
            self.mle = {}
            for k in mp.model.free.keys():
                self.mle[k] = mp.model.free[k].value
        else:
            prelim_fpaths = self.GetFilePaths( prelim_fit=True )
            print( '\nReading in preliminary MLE fit:' )
            print( prelim_fpaths[1] )
            ifile = open( prelim_fpaths[1], 'rb' )
            prelim = pickle.load( ifile )
            ifile.close()
            self.mle = prelim['mle']
        for k in list( self.mle.keys() ):
            print( k, self.mle[k] )
        print( 'Done.\n' )                
        return None

    
    def RunMCMC( self ):
        # Initialise the emcee sampler:
        mcmc = pyhm.MCMC( self.mbundle )
        self.freepars = list( mcmc.model.free.keys() )
        mcmc.assign_step_method( pyhm.BuiltinStepMethods.AffineInvariant )
        # Define ranges to randomly sample the initial walker values from
        # (Note: GetParRanges is a function provided by user during setup):
        self.init_par_ranges = self.GetParRanges( self.mle )
        # Initial emcee burn-in with single walker group:
        #init_walkers = self.GetInitWalkers( mcmc )
        init_walkers = UR.GetInitWalkers( mcmc, self.nwalkers, self.init_par_ranges )
        mcmc.sample( nsteps=self.nburn1, init_walkers=init_walkers, verbose=False )
        mle_refined = UR.RefineMLE( mcmc.walker_chain, self.mbundle )
        self.init_par_ranges = self.GetParRanges( self.mle )
        init_walkers = UR.GetInitWalkers( mcmc, self.nwalkers, self.init_par_ranges )
        # Sample for each chain, i.e. group of walkers:
        self.walker_chains = []
        print( '\nRunning the MCMC sampling:' )
        for i in range( self.ngroups ):
            t1 = time.time()
            print( '\n... group {0} of {1}'.format( i+1, self.ngroups ) )
            # Run the burn-in:
            print( '... running burn-in for {0} steps'.format( self.nburn2 ) )
            mcmc.sample( nsteps=self.nburn2, init_walkers=init_walkers, \
                         verbose=False )
            burn_end_state = UR.GetWalkerState( mcmc )
            # Run the main chain:
            print( '... running main chain for {0} steps'.format( self.nsteps ) )
            mcmc.sample( nsteps=self.nsteps, init_walkers=burn_end_state, \
                         verbose=False )
            self.walker_chains += [ mcmc.walker_chain ]
            t2 = time.time()
        # Refine the MLE solution using MCMC output:
        #self.RefineMLEfromGroups()
        self.mle = UR.RefineMLEfromGroups( self.walker_chains, self.mbundle )
        self.ExtractMCMCOutput( nburn=0 )
        self.Save()
        self.Plot()
        return None

    
    def Plot( self ):
        plt.ioff()
        #dsets = list( self.evalmodels.keys() )
        nvisits = len( self.dsets )
        dat = {}
        z_thrsf = []
        z_psignalf = []
        for i in range( nvisits ):
            j = self.dsets[i]
            wlc = self.wlcs[j]
            delt = wlc.jd-wlc.jd[0]
            jd = wlc.jd
            # User-defined cullixs:
            ixsc = self.cullixs[j]
            ixsf0 = ixsc[wlc.scandirs[ixsc]==1]
            ixsb0 = ixsc[wlc.scandirs[ixsc]==-1]
            tmid = self.GetTmid( j, ixsf0, ixsb0 )
            thrs = 24.*( jd-tmid )
            flux = wlc.whitelc[self.analysis]['flux']
            uncs = wlc.whitelc[self.analysis]['uncs']
            if ixsf0.sum()>0:
                zf = self.PrepPlotVars( j, delt, flux, uncs, scandir='f' )
                zf['thrsf'] = 24*( zf['jdf']-tmid )
                ixsf = zf['ixs']
                ixsf0 = ixsf0[np.isin(ixsf0,ixsf,invert=True)]
            else:
                zf = None
            if ixsb0.sum()>0:
                zb = self.PrepPlotVars( j, delt, flux, uncs, scandir='b' )
                zb['thrsf'] = 24*( zb['jdf']-tmid )
                ixsb = zb['ixs']
                ixsb0 = ixsb0[np.isin(ixsb0,ixsb,invert=True)]
            else:
                zb = None
            dat[j], thrsfj, psignalfj = self.PlotVisit( j, zf, zb, \
                                                        ixsf0, ixsb0, thrs )
            z_thrsf += [ thrsfj ]
            z_psignalf += [ psignalfj ]
        self.PlotCombined( dat, z_thrsf, z_psignalf )
        return None
    
    def PlotVisit( self, j, zf, zb, ixsf0, ixsb0, thrs ):
        fig, axsl, axsr = self.CreatePlotAxes()
        datj = self.PlotRaw( axsl[0], axsr[0], zf, zb, ixsf0, ixsb0, thrs )
        self.PlotSystematics( axsl[1], axsr[1], zf, zb, ixsf0, ixsb0, thrs )
        #thrsfj, psignalfj = self.PlotCorrected( axsl[2], axsl[3], zf, zb, \
        #                                        ixsf0, ixsb0, thrs )
        thrsfj, psignalfj = self.PlotCorrected( axsl[2], axsl[3], zf, zb, thrs )
        opath = self.whitefit_mle_fpath_pkl\
                .replace( '.pkl', '.{0}.pdf'.format( j ) )
        ofigpath = os.path.basename( opath ).replace( '.pdf', '' )
        fig.suptitle( ofigpath, fontsize=16 )
        fig.savefig( opath )
        print( '\nSaved:\n{0}\n'.format( opath ) )
        return datj, thrsfj, psignalfj

    
    def PlotRaw( self, axl, axr, zf, zb, ixsf0, ixsb0, thrs ):
        lcolor = 'Orange'
        xcolor = 'r'
        dat = {}
        dat['thrs'] = []
        dat['dflux'] = []
        dat['dfluxc'] = []
        dat['uncs_ppm'] = []
        dat['resids_ppm'] = []
        #if ixsf0.sum()>0:
        if zf is not None:
            print( 'zf' )
            ixsf = zf['ixs']
            axl.plot( thrs[ixsf0], zf['dflux'][ixsf0], 'x', \
                          mec=xcolor, zorder=200 )
            axl.plot( thrs[ixsf], zf['dflux'][ixsf], 'o', \
                          mec=zf['mec'], mfc=zf['mfc'], zorder=100 )
            axl.plot( zf['thrsf'], 100*( zf['ttrendf']*zf['psignalf']-zf['f0'] ), \
                      '-', color=lcolor, zorder=0 )
            dat['thrs'] += [ thrs[ixsf] ]
            dat['dflux'] += [ zf['dflux'][ixsf] ]
            dat['dfluxc'] += [ zf['dfluxc'] ]
            dat['uncs_ppm'] += [ zf['uncs_ppm'] ]
            dat['resids_ppm'] += [ zf['resids_ppm'] ]
        #if ixsb0.sum()>0:
        if zb is not None:
            print( 'zb' )
            ixsb = zb['ixs']
            axr.plot( thrs[ixsb0], zb['dflux'][ixsb0], 'x', \
                          mec=xcolor, zorder=200 )
            axr.plot( thrs[ixsb], zb['dflux'][ixsb], 'o', \
                          mec=zb['mec'], mfc=zb['mfc'], zorder=100 )
            axr.plot( zb['thrsf'], 100*( zb['ttrendf']*zb['psignalf']-zb['f0'] ), \
                      '-', color=lcolor, zorder=0 )
            for ixs in zb['orbixs']:
                axr.plot( zb['thrsf'], \
                          100*( zb['ttrendf']*zb['psignalf']-zb['f0'] ), \
                          '-', color=lcolor, zorder=0 )
                axr.plot( zb['thrsf'], \
                          100*( zb['ttrendf']*zb['psignalf']-zb['f0'] ), \
                          '-', color=lcolor, zorder=0 )
            dat['thrs'] += [ thrs[ixsb] ]
            dat['dflux'] += [ zb['dflux'][ixsb] ]
            dat['dfluxc'] += [ zb['dfluxc'] ]
            dat['uncs_ppm'] += [ zb['uncs_ppm'] ]
            dat['resids_ppm'] += [ zb['resids_ppm'] ]
        dat['thrs'] = np.concatenate( dat['thrs'] )
        dat['dflux'] = np.concatenate( dat['dflux'] )
        dat['dfluxc'] = np.concatenate( dat['dfluxc'] )
        dat['uncs_ppm'] = np.concatenate( dat['uncs_ppm'] )
        dat['resids_ppm'] = np.concatenate( dat['resids_ppm'] )
        plt.setp( axl.xaxis.get_ticklabels(), visible=False )
        plt.setp( axr.xaxis.get_ticklabels(), visible=False )
        plt.setp( axr.yaxis.get_ticklabels(), visible=False )
        return dat

    def PlotSystematics( self, axl, axr, zf, zb, ixsf0, ixsb0, thrs ):
        lcolor = 'Orange'
        # Systematics:
        if zf is not None:
            ixsf = zf['ixs']
            axl.plot( thrs[ixsf], zf['syst_ppm'], 'o', \
                      mec=zf['mec'], mfc=zf['mfc'], zorder=100 )
            for ixs in zf['orbixs']:
                t = thrs[ixsf][ixs]
                f = (1e6)*zf['mu'][ixs]
                axl.plot( t, f, '-', color=lcolor, zorder=0 )
        if zb is not None:
            ixsb = zb['ixs']
            axr.plot( thrs[ixsb], zb['syst_ppm'], 'o', \
                      mec=zb['mec'], mfc=zb['mfc'], zorder=100 )
            for ixs in zb['orbixs']:
                t = thrs[ixsb][ixs]
                f = (1e6)*zb['mu'][ixs]
                axr.plot( t, f, '-', color=lcolor, zorder=0 )
        plt.setp( axr.yaxis.get_ticklabels(), visible=False )

    def PlotCorrected( self, axlc, axresids, zf, zb, thrs ):
        lcolor = 'Orange'
        # Corrected flux:
        if zf is not None:
            ixsf = zf['ixs']
            axlc.plot( thrs[ixsf], zf['dfluxc'], 'o', mec=zf['mec'], \
                       mfc=zf['mfc'], zorder=100 )
            axresids.errorbar( thrs[ixsf], zf['resids_ppm'], yerr=zf['uncs_ppm'], \
                               fmt='o', mec=zf['mec'], mfc=zf['mfc'], \
                               ecolor=zf['mec'], zorder=100 )
            #thrsff = 24.*( zf['jdf']-tmid )
            thrsff = zf['thrsf']
            psignalff = zf['psignalf']
            ttrendff = zf['ttrendf']
            psignalf = zf['psignal']
            ttrendf = zf['ttrend']
        else:
            thrsff = []
            psignalff = []
        if zb is not None:
            ixsb = zb['ixs']
            axlc.plot( thrs[ixsb], zb['dfluxc'], 'o', mec=zb['mec'], \
                       mfc=zb['mfc'], zorder=100 )
            axresids.errorbar( thrs[ixsb], zb['resids_ppm'], yerr=zb['uncs_ppm'], \
                               fmt='o', mec=zb['mec'], mfc=zb['mfc'], \
                               ecolor=zb['mec'], zorder=100 )
            #thrsfb = 24.*( zb['jdf']-tmid )
            thrsfb = zb['thrsf']
            psignalfb = zb['psignalf']
            ttrendfb = zb['ttrendf']
            psignalb = zb['psignal']
            ttrendb = zb['ttrend']
        else:
            thrsfb = []
            psignalfb = []
        thrsj = np.concatenate( [ thrsff, thrsfb ] )
        psignalj = np.concatenate( [ psignalff, psignalfb ] )
        axresids.axhline( 0, ls='-', c=lcolor, zorder=0 )
        axlc.plot( thrsj, 100*( psignalj-1 ), '-', color=lcolor, zorder=0 )
        plt.setp( axlc.xaxis.get_ticklabels(), visible=False )
        ixsj = np.argsort( thrsj )        
        return thrsj[ixsj], psignalj[ixsj]

    
    def PlotCombined( self, dat, thrsf, psignalf ):
        thrsf = np.concatenate( thrsf )
        ixs = np.argsort( thrsf )
        thrsf = thrsf[ixs]
        psignalf = np.concatenate( psignalf )[ixs]
        #dsets = list( dat.keys() )
        nvisits = len( self.dsets )
        fig = plt.figure()
        ax1 = fig.add_subplot( 211 )
        ax2 = fig.add_subplot( 212, sharex=ax1 )
        ax2.set_xlabel( 'Time from mid-transit (h)' )
        ax2.set_ylabel( 'Resids (ppm)' )
        ax1.set_ylabel( 'Flux change (%)' )
        cs = UR.MultiColors()
        lc = 'k'
        ax1.plot( thrsf, 100*( psignalf-1 ), '-', c=lc, zorder=0 )
        ax2.axhline( 0, ls='-', c=lc, zorder=0 )
        for i in range( nvisits ):
            j = self.dsets[i]
            ax1.errorbar( dat[j]['thrs'], dat[j]['dfluxc'], \
                          yerr=(1e-4)*dat[j]['uncs_ppm'], \
                          fmt='o', mec=cs[i], mfc=cs[i], ecolor=cs[i], \
                          label=j, alpha=0.6 )
            ax2.errorbar( dat[j]['thrs'], dat[j]['resids_ppm'], \
                          yerr=dat[j]['uncs_ppm'], fmt='o', \
                          mec=cs[i], mfc=cs[i], ecolor=cs[i], alpha=0.6 )
        ax1.legend( loc='lower left', numpoints=1 )
        opath = self.whitefit_mle_fpath_pkl.replace( '.pkl', '.joint.pdf' )
        ofigpath = os.path.basename( opath ).replace( '.pdf', '' )
        fig.suptitle( ofigpath, fontsize=16 )
        fig.savefig( opath )
        print( '\nSaved:\n{0}\n'.format( opath ) )
        return None
        

    def PrepPlotVars( self, dset, delt, flux, uncs, scandir='f' ):
        z = {}
        z['evalmodel'], z['ixs'] = self.evalmodels[dset][scandir]
        print( scandir, z['ixs'] )
        z['mfit'] = z['evalmodel']( self.mle )                
        z['orbixs'] = UR.SplitHSTOrbixs( delt[z['ixs']]*24 )
        z['f0'] = flux[z['ixs']][-1]
        z['dflux'] = 100*( flux-z['f0'] )
        z['ttrend'] = z['mfit']['arrays']['ttrend']
        z['mu'] = z['mfit']['arrays']['mu'].flatten()
        z['systematics'] = z['ttrend']*( 1+z['mu'] )
        z['psignal'] = z['mfit']['arrays']['psignal']
        z['jdf'] = z['mfit']['arrays']['jdf']
        z['ttrendf'] = z['mfit']['arrays']['ttrendf']
        z['psignalf'] = z['mfit']['arrays']['psignalf']
        z['resids_ppm'] = (1e6)*( flux[z['ixs']]-\
                                  z['psignal']*z['systematics'] )
        z['uncs_ppm'] = (1e6)*uncs[z['ixs']]
        z['dfluxc'] = 100*( flux[z['ixs']]/z['systematics']-1 )
        z['syst_ppm'] = (1e6)*( flux[z['ixs']]/(z['psignal']*z['ttrend'])-1 )
        if scandir=='f':
            z['mfc'] = np.array( [217,240,211] )/256.
            z['mec'] = np.array( [27,120,55] )/256.
        elif scandir=='b':
            z['mfc'] = np.array( [231,212,232] )/256.
            z['mec'] = np.array( [118,42,131] )/256.
        else:
            pdb.set_trace()
        return z

    def CreatePlotAxes( self ):
        figw = 12
        figh = 12
        fig = plt.figure( figsize=[figw,figh] )
        axh1 = 0.30
        axh2 = 0.15
        axh3 = axh1
        axh4 = 0.10
        ylow1 = 1-0.05-axh1
        ylow2 = ylow1-axh2
        ylow3 = ylow2-axh3-0.055
        ylow4 = ylow3-axh4
        axw = 0.45
        xlowl = 0.08
        xlowr = xlowl + axw #+ xlowl
        ax1l = fig.add_axes( [ xlowl, ylow1, axw, axh1 ] )
        ax1r = fig.add_axes( [ xlowr, ylow1, axw, axh1 ], sharex=ax1l, \
                             sharey=ax1l )
        ax2l = fig.add_axes( [ xlowl, ylow2, axw, axh2 ], sharex=ax1l )
        ax2r = fig.add_axes( [ xlowr, ylow2, axw, axh2 ], sharex=ax1l, \
                             sharey=ax2l )
        ax3l = fig.add_axes( [ xlowl, ylow3, axw, axh3 ], sharex=ax1l )
        ax4l = fig.add_axes( [ xlowl, ylow4, axw, axh4 ], sharex=ax1l )
        ax1l.set_ylabel( 'Flux change (%)' )
        ax2l.set_ylabel( 'Systematics (ppm)' )
        ax3l.set_ylabel( 'Flux change (%)' )
        ax4l.set_ylabel( 'Residuals (ppm)' )
        ax2l.set_xlabel( 'Time (h)' )
        ax2r.set_xlabel( 'Time (h)' )
        ax4l.set_xlabel( 'Time (h)' )
        axsl = [ ax1l, ax2l, ax3l, ax4l ]
        axsr = [ ax1r, ax2r ]
        return fig, axsl, axsr

    

    def LoadFromFile( self ):
        mcmc_fpath, mle_fpath = self.GetFilePaths( prelim_fit=self.prelim_fit )
        ifile = open( mcmc_fpath, 'rb' )
        z = pickle.load( ifile )
        ifile.close()
        self.whitefit_mcmc_fpath_pkl = mcmc_fpath
        self.whitefit_mle_fpath_pkl = mle_fpath
        self.cullixs = z['cullixs_init']
        self.keepixs_final = z['keepixs_final']
        self.batpars = z['batpars']
        self.pmodels = z['pmodels']
        self.bestfits = z['bestfits']
        self.mle = z['mle']
        self.freepars = z['freepars']
        self.Tmid0 = z['Tmid0']
        self.chain = z['chain']
        self.walker_chains = z['walker_chains']
        self.grs = z['grs']
        self.chain_properties = z['chain_properties']
        self.ngroups = z['ngroups']
        self.nwalkers = z['nwalkers']
        self.nsteps = z['nsteps']
        self.nburn2 = z['nburn']
        return None
        
    def Save( self ):
        mcmc_fpath, mle_fpath = self.GetFilePaths( prelim_fit=self.prelim_fit )
        self.whitefit_mcmc_fpath_pkl = mcmc_fpath
        self.whitefit_mcmc_fpath_txt = mcmc_fpath.replace( '.pkl', '.txt' )
        self.whitefit_mle_fpath_pkl = mle_fpath
        bestfits, batpars, pmodels = UR.BestFitsEval( self.mle, self.evalmodels )
        self.bestfits = bestfits
        self.batpars = batpars
        self.pmodels = pmodels
        outp = {}
        outp['wlcs'] = self.wlcs
        outp['analysis'] = self.analysis
        outp['cullixs_init'] = self.cullixs
        outp['keepixs_final'] = self.keepixs_final
        outp['batpars'] = self.batpars
        outp['pmodels'] = self.pmodels
        outp['bestfits'] = bestfits
        outp['orbpars'] = { 'fittype':self.orbpars }
        if ( self.orbpars=='fixed' ):#+( self.orbpars=='wmeanfixed' ):
            outp['orbpars']['aRs'] = self.mbundle['aRs']
            outp['orbpars']['b'] = self.mbundle['b']
        else:
            outp['orbpars']['aRs'] = self.mle['aRs']
            outp['orbpars']['b'] = self.mle['b']
        outp['mle'] = self.mle
        outp['freepars'] = self.freepars
        outp['Tmid0'] = self.Tmid0
        ofile = open( self.whitefit_mle_fpath_pkl, 'wb' )
        pickle.dump( outp, ofile )
        ofile.close()
        # Add in the bulky MCMC output:
        outp['chain'] = self.chain
        outp['walker_chains'] = self.walker_chains
        outp['grs'] = self.grs
        outp['chain_properties'] = self.chain_properties
        outp['ngroups'] = self.ngroups
        outp['nwalkers'] = self.nwalkers
        outp['nsteps'] = self.nsteps
        outp['nburn'] = self.nburn2
        ofile = open( self.whitefit_mcmc_fpath_pkl, 'wb' )
        pickle.dump( outp, ofile )
        ofile.close()
        # Write to the text file:
        self.TxtOut()
        print( '\nSaved:\n{0}\n{1}\n{2}\n'.format( self.whitefit_mcmc_fpath_pkl, \
                                                   self.whitefit_mcmc_fpath_txt, \
                                                   self.whitefit_mle_fpath_pkl ) )
        return None

    
    def TxtOut( self ):
        chp = self.chain_properties
        text_str = '#\n# Sample properties: parameter, median, l34, u34, gr\n#\n'
        keys = chp['median'].keys()
        for key in keys:
            if key!='logp':
                text_str += '{0} {1:.6f} -{2:.6f} +{3:.6f} {4:.3f}\n'\
                             .format( key, chp['median'][key], \
                             np.abs( chp['l34'][key] ), chp['u34'][key], \
                             self.grs[key] )
        ofile = open( self.whitefit_mcmc_fpath_txt, 'w' )
        ofile.write( text_str )
        ofile.close()
        return text_str
    
    
    def GetODir( self ):
        dirbase = os.path.join( self.results_dir, 'white' )
        if self.orbpars=='free':
            dirbase = os.path.join( dirbase, 'orbpars_free' )
        elif self.orbpars=='fixed':
            dirbase = os.path.join( dirbase, 'orbpars_fixed' )
        else:
            pdb.set_trace() # haven't implemented other cases yet
        if self.syspars['tr_type']=='primary':
            dirbase = os.path.join( dirbase, self.ld )
        else:
            dirbase = os.path.join( dirbase, 'ldoff' )
        #dsets = list( self.wlcs.keys() )
        dsets = UR.NaturalSort( self.dsets )
        dirext = ''
        for k in dsets:
            dirext += '+{0}'.format( k )
        dirext = dirext[1:]
        if len( dsets )>1:
            if self.syspars['tr_type']=='primary':
                if self.RpRs_shared==True:
                    dirext += '.RpRs_shared'
                else:
                    dirext += '.RpRs_individ'
            elif self.syspars['tr_type']=='secondary':
                if self.EcDepth_shared==True:
                    dirext += '.EcDepth_shared'
                else:
                    dirext += '.EcDepth_individ'
            else:
                pdb.set_trace()
        dirbase = os.path.join( dirbase, dirext )
        if self.akey=='':
            print( '\n\nMust set akey to create output folder for this particular analysis\n\n' )
            pdb.set_trace()
        else:
            self.odir = os.path.join( dirbase, self.akey )
        # Don't bother with the reduction parameters in the filenames.
        # That can be done separately with a custom routine defined by
        # the user if it's really important.
        return None

    def GetFilePaths( self, prelim_fit=True ):
        self.GetODir()
        if os.path.isdir( self.odir )==False:
            os.makedirs( self.odir )
        if self.beta_free==True:
            betastr = 'beta_free'
        else:
            betastr = 'beta_fixed'
        if prelim_fit==True:
            prelimstr = 'prelim'
        else:
            prelimstr = 'final'
        oname = 'white.{0}.{1}.mcmc.{2}.pkl'.format( self.analysis, betastr, \
                                                     prelimstr )
        mcmc_fpath = os.path.join( self.odir, oname )
        mle_fpath = mcmc_fpath.replace( 'mcmc', 'mle' )
        return mcmc_fpath, mle_fpath
    
    def ExtractMCMCOutput( self, nburn=0 ):
        chaindict, grs = UR.GetChainFromWalkers( self.walker_chains, nburn=nburn )
        logp_arr = chaindict['logp']
        logp = chaindict.pop( 'logp' )
        keys_fitpars = list( chaindict.keys() )
        npar = len( keys_fitpars )
        nsamples = len( logp_arr )
        chain = np.zeros( [ nsamples, npar ] )
        for j in range( npar ):
            chain[:,j] = chaindict[keys_fitpars[j]]
        chainprops = pyhm.chain_properties( chaindict, nburn=0, thin=None, \
                                            print_to_screen=True )
        self.chain_properties = chainprops
        self.grs = grs
        self.chain = chaindict
        return None


class WFC3SpecLightCurves():
    
    def __init__( self ):
        self.target = ''
        self.dsetname = ''
        self.spec1d_fpath = ''
        self.config = None
        self.ss_dispbound_ixs = []
        self.ss_maxshift_pix = 1
        self.ss_dshift_pix = 0.001
        self.ss_smoothing_fwhm = None
        self.cuton_micron = None
        self.npix_perbin = None
        self.nchannels = None
        self.bandpass_fpath = ''
        self.atlas_fpath = ''
        self.atlas_teff = None
        self.atlas_logg = None
        self.atlas_newgrid = True
        self.whitefit_fpath_pkl = ''
        self.scankeys = { 'f':1, 'b':-1 }
        self.ld = { 'quad':None, 'nonlin':None }
        
    def Create( self, save_to_file=True ):
        print( '\nReading:\n{0}\n{1}\n'.format( self.spec1d_fpath, \
                                                self.whitefit_fpath_pkl ) )
        ifile = open( self.spec1d_fpath, 'rb' )
        spec1d = pickle.load( ifile )
        ifile.close()
        self.config = spec1d.config
        ifile = open( self.whitefit_fpath_pkl, 'rb' )
        whitefit = pickle.load( ifile )
        ifile.close()
        self.analysis = whitefit['analysis']        
        print( 'Done.' )
        self.rkeys = spec1d.rkeys
        #ecounts1d = spec1d.spectra[self.analysis]['ecounts1d']
        # Generate the speclcs:
        self.PrepSpecLCs( spec1d, whitefit )
        self.GetLD( spec1d )
        if save_to_file==True:
            self.Save()
            self.Plot( spec1d )
        return None

    
    def MakeCommonMode( self, bestfits, flux ):
        """
        Generate the common-mode correction for each scan direction
        of each dataset using the white lightcurve fit.
        """        
        self.cmode = {}
        self.cmode = {}
        for j in self.scankeys:
            ixsj = ( self.scandirs==UR.ScanVal( j ) )
            psignalj = bestfits[j]['psignal']
            self.cmode[j] = flux[ixsj]/psignalj
        return None

    
    
    def PrepSpecLCs( self, spec1d, whitefit ):
        # Get ixs to be used for each scan direction:
        self.scankeys = list( whitefit['bestfits'][self.dsetname].keys() )
        ixsc = whitefit['keepixs_final'][self.dsetname]
        self.jd = spec1d.jd[ixsc]
        self.scandirs = spec1d.scandirs[ixsc]
        # Copy auxvars, cull, split into f and b to start:
        self.auxvars = {}
        for k in list( spec1d.spectra.keys() ):
            auxvarsk = spec1d.spectra[self.analysis]['auxvars'].copy()
            self.auxvars[k] = {}
            for i in list( auxvarsk.keys() ):
                self.auxvars[k][i] = auxvarsk[i][ixsc]
        self.analysis = whitefit['analysis']
        ixsc = whitefit['keepixs_final'][self.dsetname]
        wfitarrs = whitefit['bestfits'][self.dsetname]
        wflux = whitefit['wlcs'][self.dsetname].whitelc[self.analysis]['flux']
        self.MakeCommonMode( wfitarrs, wflux[ixsc] )
        wavmicr = spec1d.spectra[self.analysis]['wavmicr']
        ecounts1d = spec1d.spectra[self.analysis]['ecounts1d']
        self.GetChannels( wavmicr )
        self.lc_flux = { 'raw':{}, 'cm':{}, 'ss':{} }
        self.lc_uncs = { 'raw':{}, 'cm':{}, 'ss':{} }
        self.MakeBasic( ecounts1d[ixsc,:] )
        self.MakeShiftStretch( wavmicr, ecounts1d[ixsc,:], wfitarrs )
        self.UnpackArrays()
        return None
    
    
    def UnpackArrays( self ):
        lc_flux = {}
        lc_uncs = {}
        for k in ['raw','cm','ss']:
            jd = []
            fluxk = []
            uncsk = []
            for j in self.scankeys:
                fluxkj = self.lc_flux[k][j]
                uncskj = self.lc_uncs[k][j]
                for i in range( self.nchannels ):
                    fnormkji = np.mean( fluxkj[:,i] )
                    fluxkj[:,i] = fluxkj[:,i]/fnormkji
                    uncskj[:,i] = uncskj[:,i]/fnormkji
                fluxk += [ fluxkj ]
                uncsk += [ uncskj ]
                ixsj = ( self.scandirs==UR.ScanVal( j ) )
                jd += [ self.jd[ixsj] ]
            jd = np.concatenate( jd )
            ixs = np.argsort( jd )
            self.lc_flux[k] = np.concatenate( fluxk )[ixs]
            self.lc_uncs[k] = np.concatenate( uncsk )[ixs]
        return None
    
    
    def GetChannels( self, wavmicr ):
        cutonmicr = self.cuton_micron
        ndisp = wavmicr.size
        nchan = self.nchannels
        nppb = self.npix_perbin
        edges0 = np.arange( ndisp )[np.argmin( np.abs( wavmicr-cutonmicr ) )]
        edges = np.arange( edges0, edges0+( nchan+1 )*nppb, nppb )
        self.chixs = []
        self.wavedgesmicr = []
        for i in range( nchan ):
            self.chixs += [ [ edges[i], edges[i+1] ] ]
            self.wavedgesmicr += [ [ wavmicr[edges[i]], wavmicr[edges[i+1]] ] ]
        return None

    
    def MakeBasic( self, ecounts1d ):
        flux_raw = {}
        uncs_raw = {}
        flux_cm = {}
        uncs_cm = {}
        for j in self.scankeys:
            ixsj = ( self.scandirs==UR.ScanVal( j ) )
            ndat = ixsj.sum()
            flux_raw[j] = np.zeros( [ ndat, self.nchannels ] )
            uncs_raw[j] = np.zeros( [ ndat, self.nchannels ] )
            for i in range( self.nchannels ):
                ixl = self.chixs[i][0]
                ixu = self.chixs[i][1]
                #flux_raw[j][:,i] = np.sum( ecounts1d[ixsj,ixl:ixu+1], axis=1 )
                flux_raw[j][:,i] = np.sum( ecounts1d[ixsj,ixl:ixu], axis=1 )
                uncs_raw[j][:,i] = np.sqrt( flux_raw[j][:,i] )
            flux_cm[j] = np.zeros( [ ndat, self.nchannels ] )
            uncs_cm[j] = np.zeros( [ ndat, self.nchannels ] )
            for i in range( self.nchannels ):
                flux_cm[j][:,i] = flux_raw[j][:,i]/self.cmode[j]
                uncs_cm[j][:,i] = uncs_raw[j][:,i]#/self.cmode[j]
        self.lc_flux['raw'] = flux_raw
        self.lc_uncs['raw'] = uncs_raw
        self.lc_flux['cm'] = flux_cm
        self.lc_uncs['cm'] = uncs_cm
        return None
    
    
    def MakeShiftStretch( self, wavmicr, ecounts1d, bestfits ):
        self.ss_dspec = {}
        self.ss_wavshift_pix = {}
        self.ss_vstretch = {}
        self.ss_enoise = {}
        if self.ss_dispbound_ixs is 'speclc_range':
            nwav = int( wavmicr.size )
            dwav0 = np.abs( wavmicr-self.wavedgesmicr[0][0] )
            dwav1 = np.abs( wavmicr-self.wavedgesmicr[-1][1] )
            ix0 = np.arange( nwav )[np.argmin( dwav0 )]
            ix1 = np.arange( nwav )[np.argmin( dwav1 )]
            self.ss_dispbound_ixs = [ ix0, ix1 ]
        for j in self.scankeys:
            ixsj = ( self.scandirs==UR.ScanVal( j ) )
            ecounts1dj = ecounts1d[ixsj,:]
            psignalj = bestfits[j]['psignal']
            ixs_full = np.arange( psignalj.size )
            ixs_in = psignalj<1-1e-6
            ixs_out = ixs_full[np.isin(ixs_full,ixs_in,invert=True)]
            refspecj = np.median( ecounts1dj[ixs_out,:], axis=0 )
            self.CalcSpecVars( j, ecounts1dj, refspecj )
            # Normalise the residuals and uncertainties:
            nframes, ndisp = np.shape( ecounts1dj )
            for i in range( nframes ):
                self.ss_dspec[j][i,:] /= refspecj
                self.ss_enoise[j][i,:] /= refspecj
            # Construct the ss lightcurves by adding back in the white psignal:
            flux_ss = np.zeros( [ nframes, self.nchannels ] )
            uncs_ss = np.zeros( np.shape( self.ss_dspec[j] ) )
            for i in range( self.nchannels ):
                a = self.chixs[i][0]
                b = self.chixs[i][1]
                # Bin the differential fluxes over the current channel:
                dspeci = np.mean( self.ss_dspec[j][:,a:b+1], axis=1 )
                # Since the differential fluxes correspond to the raw spectroscopic
                # fluxes corrected for wavelength-common-mode systematics minus the 
                # white transit, we simply add back in the white transit signal to
                # obtain the systematics-corrected spectroscopic lightcurve:
                flux_ss[:,i] = dspeci + psignalj
                # Computed the binned uncertainties for the wavelength channel:
                uncs_ss[:,i] = np.mean( self.ss_enoise[j][:,a:b+1], axis=1 )
                uncs_ss[:,i] /= np.sqrt( float( b-a+1 ) )
            self.lc_flux['ss'][j] = flux_ss
            self.lc_uncs['ss'][j] = uncs_ss
        return None

    
    def CalcSpecVars( self, scan, ecounts1d, refspec ):
        nframes, ndisp = np.shape( ecounts1d )
        if self.ss_smoothing_fwhm is not None:
            smthsig = smoothing_fwhm/2./np.sqrt( 2.*np.log( 2. ) )
            refspec = scipy.ndimage.filters.gaussian_filter1d( refspec, smthsig )
        else:
            smthsig = None
        dwavs, shifted = self.PrepShiftedSpec( refspec )
        nshifts = len( dwavs )
        # Now loop over the individual spectra and determine which
        # of the shifted reference spectra gives the best match:
        print( '\nDetermining shifts and stretches:\nscandir={0}'.format( scan ) )
        self.ss_wavshift_pix[scan] = np.zeros( nframes )
        self.ss_vstretch[scan] = np.zeros( nframes )
        self.ss_dspec[scan] = np.zeros( [ nframes, ndisp ] )
        self.ss_enoise[scan] = np.zeros( [ nframes, ndisp ] )
        ix0 = self.ss_dispbound_ixs[0]
        ix1 = self.ss_dispbound_ixs[1]
        A = np.ones( [ndisp,2] )
        #A = np.column_stack( [ A0, x ] )
        coeffs = []
        for i in range( nframes ):
            print( '... frame {0:.0f} of {1:.0f}'.format( i+1, nframes ) )
            rms_i = np.zeros( nshifts )
            diffs = np.zeros( [ nshifts, ndisp ] )
            vstretches_i = np.zeros( nshifts )
            for j in range( nshifts ):
                A[:,1] = shifted[j,:]
                b = np.reshape( ecounts1d[i,:], [ ndisp, 1 ] )
                res = np.linalg.lstsq( A, b, rcond=None )
                c = res[0].flatten()
                fit = np.dot( A, c )
                vstretches_i[j] = c[1]
                diffs[j,:] = ecounts1d[i,:] - fit
                rms_i[j] = np.sqrt( np.mean( diffs[j,:][ix0:ix1+1]**2. ) )
            ix = np.argmin( rms_i )
            self.ss_dspec[scan][i,:] = diffs[ix,:]#/ref_spectrum
            self.ss_enoise[scan][i,:] = np.sqrt( ecounts1d[i,:] )#/ref_spectrum
            self.ss_wavshift_pix[scan][i] = dwavs[ix]
            self.ss_vstretch[scan][i] = vstretches_i[ix]
        return None

    def PrepShiftedSpec( self, refspec ):
        """
        Interpolates the reference spectrum on to a grid of
        increments equal to the dwav shift increment.
        """
        ndisp = len( refspec )
        xmax = self.ss_maxshift_pix
        dx = self.ss_dshift_pix
        dwavs = np.arange( -xmax, xmax+dx, dx )
        nshifts = len( dwavs )
        npad = xmax+1
        x = np.arange( ndisp )
        xi = np.arange( -npad, ndisp+npad )
        zeropad = np.zeros( npad )
        refspeci = np.concatenate( [ zeropad, refspec, zeropad ] )
        interpf = scipy.interpolate.interp1d( xi, refspeci, kind='cubic' )
        shifted = np.zeros( [ nshifts, ndisp ] )
        for i in range( nshifts ):
            shifted[i,:] = interpf( x+dwavs[i] )
        return dwavs, shifted

    def GetLD( self, spec1d ):
        atlas = AtlasModel()
        atlas.fpath = self.atlas_fpath
        atlas.teff = self.atlas_teff
        atlas.logg = self.atlas_logg
        atlas.newgrid = self.atlas_newgrid
        atlas.ReadGrid()
        ld = LimbDarkening()
        ld.wavmicr = atlas.wavmicr
        ld.intens = atlas.intens
        ld.mus = atlas.mus
        bp = Bandpass()
        bp.config = spec1d.config
        bp.fpath = self.bandpass_fpath
        bp.Read()
        ld.bandpass_wavmicr = bp.bandpass_wavmicr
        ld.bandpass_thput = bp.bandpass_thput
        ld_quad = np.zeros( [ self.nchannels, 2 ] )
        ld_nonlin = np.zeros( [ self.nchannels, 4 ] )
        for i in range( self.nchannels ):
            ld.cutonmicr = self.wavedgesmicr[i][0]
            ld.cutoffmicr = self.wavedgesmicr[i][1]
            ld.Compute()
            ld_quad[i,:] = ld.quad
            ld_nonlin[i,:] = ld.nonlin
        self.ld = {}
        self.ld['quad1d'] = ld_quad
        self.ld['nonlin1d'] = ld_nonlin
        return None
        
    def Save( self ):
        if os.path.isdir( self.lc_dir )==False:
            os.makedirs( self.lc_dir )
        self.GenerateFilePath()
        ofile = open( self.lc_fpath, 'wb' )
        pickle.dump( self, ofile )
        ofile.close()
        print( '\nSaved:\n{0}'.format( self.lc_fpath ) )
        return None

    def Plot( self, spec1d ):
        wavmicr = spec1d.spectra[self.analysis]['wavmicr']
        f = spec1d.spectra[self.analysis]['ecounts1d'][-1,:]
        f /= f.max()
        plt.ioff()
        nchan = len( self.chixs )
        c = 'Blue'
        alpha = [ 0.3, 0.6 ]
        fig = plt.figure()
        ax = fig.add_subplot( 111 )
        ax.plot( wavmicr, f, '-k' )
        ax.set_xlabel( 'Wavelength (micron)' )
        ax.set_ylabel( 'Normalised flux' )
        for i in range( nchan ):
            alphaj = alpha[(i+1)%2]
            ixl = self.chixs[i][0]
            #ixu = self.chixs[i][1]+1
            ixu = self.chixs[i][1]
            ixs = ( wavmicr>=wavmicr[ixl] )*( wavmicr<=wavmicr[ixu] )
            ax.fill_between( wavmicr[ixs], 0, f[ixs], facecolor=c, alpha=alphaj )
        if spec1d.config=='G141':
            ax.set_xlim( [ 0.97, 1.8 ] )
        opath = self.lc_fpath.replace( '.pkl', '.chixs.pdf' )
        titlestr = 'nchan={0:.0f}, cutonmicr={1:.3f}, npixpbin={2:.0f}'\
                   .format( nchan, self.cuton_micron, self.npix_perbin )
        ax.set_title( titlestr )
        fig.savefig( opath )
        plt.ion()
        print( '\nSaved:\n{0}'.format( opath ) )        
        return None
    
    def GenerateFilePath( self ):
        oname = 'speclcs.{0}'.format( os.path.basename( self.spec1d_fpath ) )
        oname = oname.replace( '.pkl', '.nchan{0}.pkl'.format( self.nchannels ) )
        self.lc_fpath = os.path.join( self.lc_dir, oname )
        return None

    def LoadFromFile( self ):
        ifile = open( self.lc_fpath, 'rb' )
        self = pickle.load( ifile )
        ifile.close()
        print( '\nLoaded:{0}\n'.format( self.lc_fpath ) )
        return self

    
    
class WFC3WhiteLightCurve():
    def __init__( self ):
        self.target = ''
        self.dsetname = ''
        self.lc_dir = ''
        self.spec1d_fpath = ''
        self.config = None
        self.dispixs = 'all'
        self.bandpass_fpath = ''
        self.atlas_fpath = ''
        self.atlas_teff = None
        self.atlas_logg = None
        self.atlas_newgrid = True
        self.ld = { 'quad':None, 'nonlin':None }
        
    def Create( self ):
        print( '\nReading:\n{0}'.format( self.spec1d_fpath ) )
        ifile = open( self.spec1d_fpath, 'rb' )
        spec1d = pickle.load( ifile )
        ifile.close()
        print( 'Done.' )
        d1, d2 = spec1d.trim_box[1]
        self.jd = spec1d.jd
        self.scandirs = spec1d.scandirs
        self.config = spec1d.config
        #self.rkeys = ['rlast']
        self.rkeys = spec1d.rkeys
        self.whitelc = {}
        for k in self.rkeys:
            self.whitelc[k] = {}
            self.whitelc[k]['auxvars'] = spec1d.spectra[k]['auxvars']
            wavmicr = spec1d.spectra[k]['wavmicr'][d1:d2+1]
            ecounts1d = spec1d.spectra[k]['ecounts1d'][:,d1:d2+1]
            if self.dispixs=='all':
                self.cutonmicr = wavmicr.min()
                self.cutoffmicr = wavmicr.max()
                flux = np.sum( ecounts1d, axis=1 )
            else:
                ixl = self.dispixs[0]
                ixu = self.dispixs[1]
                self.cutonmicr = wavmicr[ixl]
                self.cutoffmicr = wavmicr[ixu]
                #flux = np.sum( ecounts1d[:,ixl:ixu+1], axis=1 )
                flux = np.sum( ecounts1d[:,ixl:ixu], axis=1 )
            fluxn = flux[-1]
            self.whitelc[k]['flux'] = flux/fluxn
            self.whitelc[k]['uncs'] = np.sqrt( flux )/fluxn
        self.GetLD( spec1d )
        self.Save()
        return None

    def GetLD( self, spec1d ):
        atlas = AtlasModel()
        atlas.fpath = self.atlas_fpath
        atlas.teff = self.atlas_teff
        atlas.logg = self.atlas_logg
        atlas.newgrid = self.atlas_newgrid
        atlas.ReadGrid()
        ld = LimbDarkening()
        ld.wavmicr = atlas.wavmicr
        ld.intens = atlas.intens
        ld.mus = atlas.mus
        bp = Bandpass()
        bp.config = spec1d.config
        bp.fpath = self.bandpass_fpath
        bp.Read()
        ld.bandpass_wavmicr = bp.bandpass_wavmicr
        ld.bandpass_thput = bp.bandpass_thput
        ld.cutonmicr = self.cutonmicr
        ld.cutoffmicr = self.cutoffmicr
        ld.Compute()
        self.ld = {}
        self.ld['quad1d'] = ld.quad
        self.ld['nonlin1d'] = ld.nonlin
        return None
    
    def Plot( self ):
        plt.ioff()
        vbuff = 0.05
        hbuff = 0.05
        nrows = 4
        axw = 1-3.5*hbuff
        axh1 = 0.3
        axh234 = ( 1-2.3*vbuff-axh1 )/3.
        xlow = 3.*hbuff
        ylow1 = 1-0.6*vbuff-axh1
        ylow2 = ylow1-0.3*vbuff-axh234
        ylow3 = ylow2-0.3*vbuff-axh234
        ylow4 = ylow3-0.3*vbuff-axh234
        jd = self.jd
        thrs = 24*( jd-jd[0] )
        scandirs = self.scandirs
        ixsf = ( scandirs==1 )
        ixsb = ( scandirs==-1 )
        ixs = [ ixsf, ixsb ]
        labels = [ 'forward', 'backward' ]
        cs = [ 'm', 'c' ]
        for k in list( self.whitelc.keys() ):
            fig = plt.figure( figsize=[7,12] )
            ax1 = fig.add_axes( [ xlow, ylow1, axw, axh1 ] )
            ax2 = fig.add_axes( [ xlow, ylow2, axw, axh234 ], sharex=ax1 )
            ax3 = fig.add_axes( [ xlow, ylow3, axw, axh234 ], sharex=ax1 )
            ax4 = fig.add_axes( [ xlow, ylow4, axw, axh234 ], sharex=ax1 )
            for ax in [ax1,ax2,ax3]:
                plt.setp( ax.xaxis.get_ticklabels(), visible=False )
            flux = self.whitelc[k]['flux']
            uncs = self.whitelc[k]['uncs']
            cdcs = self.whitelc[k]['auxvars']['cdcs']
            wavshifts = self.whitelc[k]['auxvars']['wavshift_pix']
            bg = self.whitelc[k]['auxvars']['bg_ppix']
            for i in range( 2 ): # scan directions
                if ixs[i].max(): # this could be improved by recording which scan directions are present
                    ax1.plot( thrs[ixs[i]], flux[ixs[i]]/flux[ixs[i]][-1], 'o', \
                              mfc=cs[i], mec=cs[i], label=labels[i] )
                    y2 = cdcs[ixs[i]]-np.mean( cdcs[ixs[i]] )
                    ax2.plot( thrs[ixs[i]], y2, 'o', mfc=cs[i], mec=cs[i] )
                    y3 = wavshifts[ixs[i]]-np.mean( wavshifts[ixs[i]] )
                    ax3.plot( thrs[ixs[i]], y3, 'o', mfc=cs[i], mec=cs[i] )
                    ax4.plot( thrs[ixs[i]], bg[ixs[i]], 'o', mfc=cs[i], mec=cs[i] )
            ax1.legend( loc='lower right' )
            fig.text( 0.7*hbuff, ylow1+0.5*axh1, 'Relative Flux', rotation=90, \
                      horizontalalignment='right', verticalalignment='center' )
            fig.text( 0.7*hbuff, ylow2+0.5*axh234, 'Cross-dispersion drift (pix)', \
                      rotation=90, horizontalalignment='right', \
                      verticalalignment='center' )
            fig.text( 0.7*hbuff, ylow3+0.5*axh234, 'Dispersion drift (pix)', \
                      rotation=90, horizontalalignment='right', \
                      verticalalignment='center' )
            fig.text( 0.7*hbuff, ylow4+0.5*axh234, 'Background (e-/pix)', \
                      rotation=90, horizontalalignment='right', \
                      verticalalignment='center' )
            fig.text( xlow+0.5*axw, 0.1*hbuff, 'Time (h)', rotation=0, \
                      horizontalalignment='center', verticalalignment='bottom' )
            titlestr = '{0} - {1}'\
                       .format( self.target, os.path.basename( self.lc_fpath ) )
            fig.text( xlow+0.5*axw, ylow1+1.03*axh1, titlestr, \
                      horizontalalignment='center', verticalalignment='bottom' )
            opath = self.lc_fpath.replace( '.pkl', '.{0}.pdf'.format( k ) )
            fig.savefig( opath )
            plt.close()
            print( '\nSaved:\n{0}\n'.format( opath ) )
        plt.ion()
        return None

    def GenerateFilePath( self ):
        oname = 'whitelc.{0}'.format( os.path.basename( self.spec1d_fpath ) )
        self.lc_fpath = os.path.join( self.lc_dir, oname )
        return None

    def Save( self ):
        if os.path.isdir( self.lc_dir )==False:
            os.makedirs( self.lc_dir )
        self.GenerateFilePath()
        ofile = open( self.lc_fpath, 'wb' )
        pickle.dump( self, ofile )
        ofile.close()
        print( '\nSaved:\n{0}'.format( self.lc_fpath ) )
        self.Plot()
        return None
        
    def LoadFromFile( self ):
        ifile = open( self.lc_fpath, 'rb' )
        self = pickle.load( ifile )
        ifile.close()
        print( '\nLoaded:{0}\n'.format( self.lc_fpath ) )
        return self



class WFC3Spectra():
    
    def __init__( self ):
        self.config = ''
        self.dsetname = ''
        self.ima_dir = ''
        self.btsettl_fpath = ''
        self.spec1d_dir = ''
        self.spec1d_fpath = ''
        self.ntrim_edge = None
        self.apradius = None
        self.maskradius = None
        self.smoothing_fwhm = None
        self.trim_disp_ixs = []
        self.trim_crossdisp_ixs = []
        self.ss_dispbound_ixs = []
        self.bg_crossdisp_ixs = []
        self.bg_disp_ixs = []
        self.zap2d_nsig_transient = 10
        self.zap2d_nsig_static = 10
        self.zap2d_niter = 1
        self.zap1d_nsig_transient = 5
        self.zap1d_niter = 2
        return None

    
    def Extract1DSpectra( self ):
        if ( self.smoothing_fwhm is None )+( self.smoothing_fwhm==0 ):
            self.smoothing_str = 'unsmoothed'
            self.smoothing_fwhm = 0.0
        else:
            self.smoothing_str = 'smooth{0:.2f}pix'.format( self.smoothing_fwhm )
        if self.config=='G141':
            self.filter_str = 'G141'
        elif self.config=='G102':
            self.filter_str = 'G102'
        else:
            pdb.set_trace()
        ecounts2d = self.ProcessIma()
        # Having problems with ZapBadPix2D, mainly with it seeming
        # to do a bad job of flagging static bad pixels that
        # probably shouldn't be flagged... so I've hacked the routine
        # in the UtilityRoutines module for now to have a crazy high
        # nsig threshold. The ZapBadPix1D below seems to work OK, but
        # ideally something like that should be done before extracting
        # the 1D spectra. It seems my HAT-P-18 pre27may2016/scripts/g141.py
        # cosmic ray routine worked better in practice, so maybe adapt
        # from there to here.....
        ecounts2d = self.ZapBadPix2D( ecounts2d ) 
        self.HSTPhaseTorb()
        self.SumSpatScanSpectra( ecounts2d )
        self.InstallBandpass()
        self.GetWavSol( make_plot=False )
        self.ZapBadPix1D()
        self.ShiftStretch()
        self.SaveEcounts2D( ecounts2d )
        self.SaveSpec1D()
        return None

    
        
    def InstallBandpass( self ):
        bp = Bandpass()
        bp.config = self.config
        bp.fpath = self.bandpass_fpath
        bp.Read()
        self.dispersion_nmppix = bp.dispersion_nmppix
        self.dispersion_micrppix = bp.dispersion_micrppix
        self.bandpass_wavmicr = bp.bandpass_wavmicr
        self.bandpass_thput = bp.bandpass_thput
        return None
        

    def GenerateFileName( self ):
        oname = '{0}.aprad{1:.1f}pix.maskrad{2:.1f}pix.pkl'\
                .format( self.dsetname, self.apradius, self.maskradius )
        #self.spec1d_fpath = os.path.join( self.spec1d_dir, oname )
        return oname

    
    def SaveSpec1D( self ):
        if os.path.isdir( self.spec1d_dir )==False:
            os.makedirs( self.spec1d_dir )
        self.spec1d_fpath = os.path.join( self.spec1d_dir, self.GenerateFileName() )
        ofile = open( self.spec1d_fpath, 'wb' )
        pickle.dump( self, ofile )
        ofile.close()
        print( '\nSaved:\n{0}'.format( self.spec1d_fpath ) )
        return None
    
    def SaveEcounts2D( self, ecounts2d ):
        if os.path.isdir( self.ecounts2d_dir )==False:
            os.makedirs( self.ecounts2d_dir )
        self.ecounts2d_fpath = os.path.join( self.ecounts2d_dir, self.GenerateFileName() )
        ofile = open( self.ecounts2d_fpath, 'wb' )
        pickle.dump( ecounts2d, ofile )
        ofile.close()
        print( '\nSaved:\n{0}'.format( self.ecounts2d_fpath ) )
        return None

    def LoadFromFile( self ):
        ifile = open( self.spec1d_fpath, 'rb' )
        self = pickle.load( ifile )
        ifile.close()
        return self
        
    def ShiftStretch( self ):
        d1, d2 = self.ss_dispbound_ixs
        dpix_max = 1
        dwav_max = dpix_max*self.dispersion_micrppix
        nshifts = int( np.round( 2*dpix_max*(1e3)+1 ) ) # 0.001 pix
        fwhm_e1d = 4. # stdv of smoothing kernel in dispersion pixels
        sig_e1d = fwhm_e1d/2./np.sqrt( 2.*np.log( 2 ) )
        #for k in ['rdiff_zap']:#self.rkeys:
        for k in self.rkeys:
            print( '\n{0}\nComputing shift+stretch for {1}:'.format( 50*'#', k ) )
            wav0 = self.spectra[k]['wavmicr']
            x0 = np.arange( wav0.size )
            e1d0 = self.spectra[k]['ecounts1d'][-1,:]
            e1d0_smth = scipy.ndimage.filters.gaussian_filter1d( e1d0, sig_e1d )
            wshifts_pix = np.zeros( self.nframes )
            vstretches = np.zeros( self.nframes )
            for i in range( self.nframes ):
                print( '{0} ... image {1} of {2}'.format( k, i+1, self.nframes ) )
                e1di = self.spectra[k]['ecounts1d'][i,:]
                if e1di.max()>0:
                    e1di_smth = scipy.ndimage.filters.gaussian_filter1d( e1di, sig_e1d )
                    cc = self.CrossCorrSol( x0, e1di_smth, x0, e1d0_smth, \
                                            dx_max=dpix_max, nshifts=2*dpix_max*1000 )
                    wshifts_pix[i] = cc[0]
                    vstretches[i] = cc[1]
                else:
                    wshifts_pix[i] = -1
                    vstretches[i] = -1
            wshifts_micr = wshifts_pix*self.dispersion_micrppix
            self.spectra[k]['auxvars']['wavshift_pix'] = wshifts_pix
            self.spectra[k]['auxvars']['wavshift_micr'] = wshifts_micr
        return None

    
    def CrossCorrSol( self, x0, ymeas, xtarg, ytarg, dx_max=1, nshifts=1000 ):
        """
        This has now been moved to ClassDefs.py.
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
        ix0, ix1 = self.ss_dispbound_ixs
        for i in range( nshifts ):
            # Assuming the default x-solution is x0, shift the model
            # array by dx. If this provides a good match to the data,
            # it means that the default x-solution x0 is off by dx.
            ytarg_shifted_i = interpf( x0 + shifts[i] )
            A[:,1] = ytarg_shifted_i/ytarg_shifted_i.max()
            res = np.linalg.lstsq( A, b, rcond=None )
            c = res[0].flatten()
            vstretches[i] = c[1]
            fit = np.dot( A, c )
            diffs = b.flatten() - fit.flatten()
            rms[i] = np.mean( diffs[ix0:ix1+1]**2. )
            ss_fits +=[ fit.flatten() ]
        ss_fits = np.row_stack( ss_fits )
        rms -= rms.min()
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
        return shiftsf[ixf], vstretchesf[ixf], ss_fits[ix,:]


    def LoadBTSettl( self ):
        if os.path.isfile( self.btsettl_fpath )==False:
            print( '\nCould not find:\n{0}\n'.format( self.btsettl_fpath ) )
            pdb.set_trace()
        elif self.btsettl_fpath.find( 'binned' )<0:
            fpath_binned = self.btsettl_fpath.replace( '.txt', '.binned.txt' )
            if os.path.isfile( fpath_binned )==False:
                print( '\nFound:\n{0}'.format( self.btsettl_fpath ) )
                print( 'but not:\n{0}'.format( fpath_binned ) )
                print( 'Binning BT Settl model down....' )
                UR.BTSettlBinDown( self.btsettl_fpath, fpath_binned )
            self.btsettl_fpath = fpath_binned
        else:
            print( 'Binned BT Settl model already exists.' )
        print( 'Loading:\n{0}\n'.format( self.btsettl_fpath ) )
        m = np.loadtxt( self.btsettl_fpath )
        wav_micr = m[:,0]*(1e-4) # convert A to micr
        flux_permicr = m[:,1]*(1e4) # convert per A to per micr
        self.btsettl_spectrum = { 'wavmicr':wav_micr, 'flux':flux_permicr }
        return None

    def GetWavSol( self, make_plot=False ):
        if os.path.isdir( self.spec1d_dir )==False:
            os.makedirs( self.spec1d_dir )
        d1, d2 = self.trim_box[1]
        dwav_max = 0.3 # in micron
        nshifts = int( np.round( 2*dwav_max*(1e4)+1 ) ) # 0.0001 micron = 0.1 nm
        for k in self.rkeys:
            print( '\nDetermining the wavelength solution for {0}'.format( k ) )
            e1d = self.spectra[k]['ecounts1d'][0,d1:d2+1]
            A2micron = 1e-4
            ndisp = e1d.size
            wbp = self.bandpass_wavmicr # old tr_wavs
            ybp = self.bandpass_thput # old tr_vals
            dwbp = np.median( np.diff( wbp ) )
            self.LoadBTSettl()
            wstar = self.btsettl_spectrum['wavmicr']
            ystar = self.btsettl_spectrum['flux']
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
            wavsol0 = w0 + self.dispersion_micrppix*delx
            #x0 = np.arange( wavsol0.size )
            # Smooth the stellar flux and model spectrum, because we use
            # the sharp edges of the throughput curve to calibrate the 
            # wavelength solution:
            fwhm_e1d = 4. # stdv of smoothing kernel in dispersion pixels
            sig_e1d = fwhm_e1d/2./np.sqrt( 2.*np.log( 2 ) )
            e1d_smth = scipy.ndimage.filters.gaussian_filter1d( e1d, sig_e1d )
            sig_star = (sig_e1d*self.dispersion_micrppix)/dwstar
            ystar_smth = scipy.ndimage.filters.gaussian_filter1d( ystar, sig_star )
            e1d_smth /= e1d_smth.max()
            ystar_smth /= ystar_smth.max()
            #cc = UR.CrossCorrSol( wavsol0, e1d_smth, wstar, \
            #                                   ystar_smth, dx_max=dwav_max, \
            #                                   nshifts=nshifts )
            cc = self.CrossCorrSol( wavsol0, e1d_smth, wstar, \
                                    ystar_smth, dx_max=dwav_max, \
                                    nshifts=nshifts )
            wshift = cc[0]
            vstretch = cc[1]
            wavmicr0 = wavsol0+wshift
            nl = np.arange( d1 )[::-1]
            nr = np.arange( self.ndisp-d2-1 )
            extl = wavmicr0[0]-(nl+1)*self.dispersion_micrppix
            extr = wavmicr0[-1]+(nr+1)*self.dispersion_micrppix
            self.spectra[k]['wavmicr'] = np.concatenate( [ extl, wavmicr0, extr ] )
            # Plot for checking the spectrum and wavelength solution:
            oname1 = '{0}.aprad{1:.1f}pix.maskrad{2:.1f}pix.'\
                     .format( self.dsetname, self.apradius, self.maskradius )
            oname2 = 'specmodel.{0}.pdf'.format( k )
            opath = os.path.join( self.spec1d_dir, oname1+oname2 )
            plt.ioff()
            plt.figure( figsize=[12,8] )
            specname = os.path.basename( self.btsettl_fpath )
            titlestr = '{0} {1} - {2}'.format( self.dsetname, k, specname )
            plt.title( titlestr, fontsize=20 )
            plt.plot( wbp, ybp/ybp.max(), '-g', \
                      label='{0} bandpass'.format( self.config ) )
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
            plt.savefig( opath )
            print( '\nSaved: {0}\n'.format( opath ) )
            plt.close()
            plt.close()
            plt.ion()
        return None
        
    
    def HSTPhaseTorb( self ):
        jd = self.jd
        delt = jd-jd[-1]
        tv = ( delt-np.mean( delt ) )/np.std( delt )        
        ixs = np.diff( delt )>5*np.median( np.diff( delt ) )
        # Determine first and last exposures of each orbit:
        firstexps = np.concatenate( [ [delt[0]], delt[1:][ixs] ] )
        lastexps = np.concatenate( [ delt[:-1][ixs], [delt[-1]] ] )
        # Take the median orbit start time to be at the same
        # HST phase, so the difference gives the HST period:
        hst_period = np.median( np.diff( firstexps ) )
        norb = int( np.ceil( ( jd.max()-jd.min() )/hst_period ) )
        delt_edges = []
        # Set zero phase a bit before the first exposure:
        deltl0 = firstexps[0]-0.2*hst_period
        # Split the orbits in delt:
        delt_edges = []
        for i in range( norb ):
            deltl = deltl0+i*hst_period
            deltu = deltl+hst_period
            delt_edges += [ [deltl,deltu] ]
        # Convert delt to hstphase, accounting for deltl0
        # as the zero phase:
        hstphase = []
        for i in range( norb ):
            l = delt_edges[i][0]
            u = delt_edges[i][1]
            ixs = ( delt>=l )*( delt<u )
            delti = ( delt[ixs]-deltl0 )-i*hst_period
            hstphase += [ delti/hst_period ]
        hstphase = np.concatenate( hstphase )
        # Split the orbits:
        orbixs = UR.SplitHSTOrbixs( delt*24 )
        torb = np.zeros( jd.size )
        for i in orbixs:
            torb[i] = jd[i]-jd[i][0]
        for k in list( self.spectra.keys() ):
            self.spectra[k]['auxvars']['hstphase'] = hstphase
            self.spectra[k]['auxvars']['torb'] = torb
            self.spectra[k]['auxvars']['tv'] = tv
        return None
    
    
    def ZapBadPix1D( self ):
        # NOTE: this doesn't appear to have been
        # done in previous versions of the code, so it might
        # not be very well tested...
        ntr = self.zap1d_nsig_transient
        d1, d2 = self.trim_box[1]
        #keys = list( self.spectra.keys() )
        keys = self.rkeys
        for k in keys:
            if k.find( 'zap' )>=0:
                print( '\n{0}\nZapping {1} ecounts1d:'.format( 50*'#', k ) )
                ecounts1d = self.spectra[k]['ecounts1d'].copy()
                zk = UR.Zap1D( ecounts1d[:,d1:d2+1], nsig_transient=ntr, \
                               niter=self.zap1d_niter )
                self.spectra[k]['ecounts1d'] = ecounts1d
                self.spectra[k]['ecounts1d'][:,d1:d2+1] = zk[0]
                self.spectra[k]['auxvars'] = self.spectra[k]['auxvars'].copy()
            else:
                continue
        return None

    
    def ZapBadPix2D( self, ecounts2d ):
        ntr = self.zap2d_nsig_transient
        nst = self.zap2d_nsig_static
        c1, c2 = self.trim_box[0]
        d1, d2 = self.trim_box[1]
        keys = list( self.spectra.keys() )
        for k in keys:
            print( '\n{0}\n Zapping {1} data cube:\n'.format( 50*'#', k ) )
            kzap = '{0}_zap'.format( k )
            self.spectra[kzap] = {}
            #ecounts2d = self.spectra[k]['ecounts2d'].copy()
            ecounts2dk = ecounts2d[k].copy() # testing
            zk = UR.Zap2D( ecounts2dk[c1:c2+1,d1:d2+1,:], nsig_transient=ntr, \
                           nsig_static=nst, niter=self.zap2d_niter )
            #self.spectra[kzap]['ecounts2d'] = ecounts2d
            #self.spectra[kzap]['ecounts2d'][c1:c2+1,d1:d2+1,:] = zk[0]
            ecounts2d[kzap] = ecounts2dk # testing
            ecounts2d[kzap][c1:c2+1,d1:d2+1,:] = zk[0] # testing
            self.spectra[kzap]['auxvars'] = self.spectra[k]['auxvars'].copy()
        self.rkeys = list( self.spectra.keys() )
        # TODO: Save a pkl file containing the images along with
        # the bad pixel maps etc; as done previously.
        # e.g. SaveRdiffFrames( self, zrdiff )
        return ecounts2d
        
    
    def ProcessIma( self ):
        # Read in the raw frames:
        search_str = os.path.join( self.ima_dir, '*_ima.fits' )
        self.ima_fpaths = np.array( glob.glob( search_str ), dtype=str )
        self.NframesNscanNdisp()
        self.tstarts = []
        self.exptimes = []
        self.spectra = { 'raw':{}, 'rlast':{}, 'rdiff':{} }
        self.rkeys = list( self.spectra.keys() )        
        self.TrimBox()
        self.BGBox()
        print( '\n{0}\nReading in raw ima files:\n'.format( 50*'#' ) )
        print( 'from directory: {0}\n'.format( self.ima_dir ) )
        ecounts2d = {}
        for k in self.rkeys:
            ecounts2d[k] = []
            self.spectra[k]['scandirs'] = []
            self.spectra[k]['auxvars'] = {}
            self.spectra[k]['auxvars']['bg_ppix'] = []
        self.scandirs = []
        ima_fpaths = []
        for i in range( self.nframes ):
            hdu = pyfits.open( self.ima_fpaths[i] )
            h0 = hdu[0].header
            h1 = hdu[1].header
            cond1 = ( h0['OBSTYPE']=='SPECTROSCOPIC' )
            cond2 = ( h0['FILTER']==self.filter_str )
            if cond1*cond2:
                hdu = pyfits.open( self.ima_fpaths[i] )
                ecounts2di, check = self.Extract2DEcounts( hdu )
                if check==False:
                    print( '... {0} of {1} - skipping {2} (appears corrupt science frame?)'
                           .format( i+1, self.nframes, os.path.basename( self.ima_fpaths[i] ) ) )
                else:
                    print( '... {0} of {1} - keeping {2}+{3}'
                           .format( i+1, self.nframes, h0['OBSTYPE'], h0['FILTER'] ) )
                    self.tstarts += [ h0['EXPSTART'] ]
                    self.exptimes += [ h0['EXPTIME'] ]
                    for k in self.rkeys:
                        ecounts2d[k] += [ ecounts2di[k] ]
                    hdu.close()
                    ima_fpaths += [ self.ima_fpaths[i] ]
            else:
                print( '... {0} of {1} - skipping {2}+{3}'
                       .format( i+1, self.nframes, h0['OBSTYPE'], h0['FILTER'] ) )
        self.tstarts = np.array( self.tstarts )
        self.exptimes = np.array( self.exptimes )
        #self.nframes = len( ima_fpaths )
        self.nframes = len( self.tstarts )
        mjd = self.tstarts + 0.5*self.exptimes/60./60./24.
        ixs = np.argsort( mjd )
        self.scandirs = np.array( self.scandirs )[ixs]
        self.mjd = mjd[ixs]
        self.ima_fpaths = np.array( ima_fpaths, dtype=str )[ixs]
        self.jd = self.mjd + 2400000.5
        self.tstarts = self.tstarts[ixs]
        self.exptimes = self.exptimes[ixs]
        for k in self.rkeys:
            bg_ppix = self.spectra[k]['auxvars']['bg_ppix']
            self.spectra[k]['auxvars'] = { 'bg_ppix':np.array( bg_ppix )[ixs] }
            ecounts2d[k] = np.dstack( ecounts2d[k] )[:,:,ixs]
        return ecounts2d

    
    def SumSpatScanSpectra( self, ecounts2d ):
        """
        Determines the spatial scan centers and extracts the spectra
        by integrating within specified aperture.
        """
        cross_axis = 0
        disp_axis = 1
        frame_axis = 2
        for k in self.rkeys:
            print( '\n{0}\nExtracting 1D spectra for {1}:'.format( 50*'#', k ) )
            #e2d = self.spectra[k]['ecounts2d']
            e2d = ecounts2d[k]
            ninterp = int( 1e4 )
            z = np.shape( e2d )
            ncross = z[cross_axis]
            ndisp = z[disp_axis]
            nframes = z[frame_axis]
            e1d = np.zeros( [ nframes, ndisp ] )
            cdcs = np.zeros( nframes )
            x = np.arange( ncross )
            nf = int( ninterp*len( x ) )
            xf = np.r_[ x.min():x.max():1j*nf ]
            for i in range( nframes ):
                print( '{0} ... image {1} of {2}'.format( k, i+1, nframes ) )
                e2di = e2d[:,:,i]
                cdcs[i] = self.DetermineScanCenter( e2di, delete=True )
                if ( cdcs[i]>=0 )*( cdcs[i]<ncross ):
                    # Determine the cross-dispersion coordinates between
                    # which the integration will be performed:
                    xmin = max( [ 0, cdcs[i]-self.apradius ] )
                    xmax = min( [ cdcs[i]+self.apradius, ncross ] )
                    # Sum rows fully contained within aperture:
                    xmin_full = int( np.ceil( xmin ) )
                    xmax_full = int( np.floor( xmax ) )
                    ixs_full = ( x>=xmin_full )*( x<=xmax_full )
                    e1d[i,:] = np.sum( e2di[ixs_full,:], axis=cross_axis )        
                    # Determine partial rows at edge of the aperture and
                    # add their weighted contributions to the flux:
                    if ixs_full[0]!=True:
                        xlow_partial = xmin_full - xmin
                        e1d[i,:] += xlow_partial*e2di[xmin_full-1,:]
                    if ixs_full[-1]!=True:
                        xupp_partial = xmax - xmax_full
                        e1d[i,:] += xupp_partial*e2di[xmax_full+1,:]
                else:
                    e1d[i,:] = -1
            self.spectra[k]['auxvars']['cdcs'] = cdcs
            self.spectra[k]['ecounts1d'] = e1d
        return None

    
    def DetermineScanCenter( self, ecounts2d, delete=False ):
        # Estimate the center of the scan for purpose of applying mask:
        x = np.arange( self.nscan )
        ninterp = 10000
        nf = int( ninterp*len( x ) )
        xf = np.linspace( self.trim_box[0][0], self.trim_box[0][1], nf )
        # Extract the cross-dispersion profile, i.e. along
        # the axis of the spatial scan:
        cdp = np.sum( ecounts2d, axis=1 )
        # Interpolate cross-dispersion profile to finer grid
        # in order to track sub-pixel shifts:
        cdpf = np.interp( xf, x, cdp )
        # Only consider points above the background level, 
        # otherwise blank sky will bias the result:
        if 0: # testing
            thresh = cdp.min() + 0.05*( cdp.max()-cdp.min() )
            ixs = ( cdpf>thresh )
            cscan = np.mean( xf[ixs] ) # testing
        else: # should be better in theory... but could be biased by cosmic rays...
            thresh = cdpf.min() + 0.05*( cdpf.max()-cdpf.min() )
            ixs = ( cdpf>thresh )
            ws = cdpf[ixs]
            # Determine the center of the scan by taking the
            # point midway between the edges:
            cscan = np.sum( ws*xf[ixs] )/np.sum( ws )
        return cscan

    
    def NframesNscanNdisp( self ):
        self.nframes = len( self.ima_fpaths )         
        hdu = pyfits.open( self.ima_fpaths[0] )
        self.nscan, self.ndisp = np.shape( hdu[1].data )
        return None
            
    def TrimBox( self ):
        """
        Returns edges of the trimmed array, e.g. to avoid edge effects.
        """
        nt = self.ntrim_edge
        c1, c2 = self.trim_crossdisp_ixs
        d1, d2 = self.trim_disp_ixs
        c1t = max( [ 0, nt, c1 ] )
        c2t = min( [ self.nscan-nt, c2 ] )
        d1t = max( [ 0, nt, d1 ] )
        d2t = min( [ self.ndisp-nt, d2 ] )
        self.trim_box = [ [c1t,c2t], [d1t,d2t] ]
        
    def BGBox( self ):
        nt = self.ntrim_edge
        c1 = max( [ nt, self.bg_crossdisp_ixs[0] ] )
        c2 = min( [ self.nscan-nt, self.bg_crossdisp_ixs[1] ] )
        d1 = max( [ nt, self.bg_disp_ixs[0] ] )
        d2 = min( [ self.ndisp-nt, self.bg_disp_ixs[1] ] )
        self.bg_box = [ [c1,c2], [d1,d2] ]
        
    def Extract2DEcounts( self, hdu ):
        nreads = UR.WFC3Nreads( hdu )
        if nreads<0:
            check = False
            return -1, check
        else:
            check = True
            # First, extract flux from final read:
            lastr_ecounts = UR.WFC3JthRead( hdu, nreads, nreads )
            #self.spectra['raw']['ecounts2d'] += [ lastr_ecounts.copy() ]
            #self.spectra['rlast']['ecounts2d'] += [ lastr_ecounts.copy() - lastr_bgppix ]
            ecounts2d = {} # testing
            ecounts2d['raw'] = lastr_ecounts.copy() # testing
            lastr_bgppix = self.BackgroundMed( lastr_ecounts ) # testing
            ecounts2d['rlast'] = lastr_ecounts.copy() - lastr_bgppix # testing
            for k in list( self.spectra.keys() ):
                self.spectra[k]['auxvars']['bg_ppix'] += [ lastr_bgppix ]
            # Second, extract flux by summing read-differences:
            ndiffs = nreads-1
            rdiff_ecounts = np.zeros( [ self.nscan, self.ndisp, ndiffs ] )
            rdiff_cscans = np.zeros( ndiffs )
            for j in range( ndiffs ):
                rix = j+1
                e1 = UR.WFC3JthRead( hdu, nreads, rix )
                e2 = UR.WFC3JthRead( hdu, nreads, rix+1 )
                # Need to perform sky subtraction here to calibrate
                # the flux level between reads, because the sky
                # actually varies quite a lot between successive reads:
                e1 -= self.BackgroundMed( e1 )
                e2 -= self.BackgroundMed( e2 )
                rdiff_ecounts[:,:,j] = e2-e1
                cscan = self.DetermineScanCenter( rdiff_ecounts[:,:,j] )
                # Apply the top-hat mask:
                ixl = int( np.floor( cscan-self.maskradius ) )
                ixu = int( np.ceil( cscan+self.maskradius ) )
                rdiff_ecounts[:ixl+1,:,j] = 0.0
                rdiff_ecounts[ixu:,:,j] = 0.0
                rdiff_cscans[j] = cscan
            dscan = rdiff_cscans[-1]-rdiff_cscans[0]
            if dscan>0:
                self.scandirs += [ +1 ]
            else:
                self.scandirs += [ -1 ]
            firstr_raw = UR.WFC3JthRead( hdu, nreads, 1 )
            firstr_ecounts = firstr_raw-self.BackgroundMed( firstr_raw )
            ecounts_per_read = np.dstack( [ firstr_ecounts, rdiff_ecounts ] )
            #self.spectra['rdiff']['ecounts2d'] += [ np.sum( ecounts_per_read, axis=2 ) ]
            ecounts2d['rdiff'] = np.sum( ecounts_per_read, axis=2 )
            return ecounts2d, check

    def BackgroundMed( self, ecounts2d ):
        c1, c2 = self.bg_box[0]
        d1, d2 = self.bg_box[1]
        bgppix = np.median( ecounts2d[c1:c2+1,d1:d2+1] )
        return bgppix 
    
    def Read1DSpectra( self ):
        self.GenerateFilePath()
        ifile = open( self.spec1d_fpath, 'rb' )
        self = pickle.load( ifile )
        ifile.close()


class Bandpass():

    def __init__( self ):
        self.config = ''
        self.fpath = ''
        self.dispersion_nmppix = None
        self.dispersion_micrppix = None
        self.bandpass_wavmicr = None
        self.bandpass_thput = None

    def Read( self ):
        nm2micr = 1e-3
        if self.config=='G141':
            self.dispersion_nmppix = 0.5*( 4.47+4.78 ) # nm/pixel
            # filename should be WFC3.IR.G141.1st.sens.2.fits        
            z = pyfits.open( self.fpath )
            tr_wavnm = z[1].data['WAVELENGTH']/10.
            tr_thput = z[1].data['SENSITIVITY']
        elif self.config=='G102':
            self.dispersion_nmppix = 0.5*( 2.36+2.51 ) # nm/pixel
            # filename should be WFC3.IR.G102.1st.sens.2.fits        
            z = pyfits.open( self.fpath )
            tr_wavnm = z[1].data['WAVELENGTH']/10.
            tr_thput = z[1].data['SENSITIVITY']
        else:
            pdb.set_trace()
        self.dispersion_micrppix = nm2micr*self.dispersion_nmppix
        #tr_wavs = self.bandpass_wavmicr *nm2micron
        ixs = np.argsort( tr_wavnm )
        self.bandpass_wavmicr = nm2micr*tr_wavnm[ixs]
        self.bandpass_thput = tr_thput[ixs]
        return None

        

class AtlasModel():
    
    def __init__( self ):
        self.fpath = ''
        self.teff = None
        self.logg = None
        self.newgrid = True
        
    def ReadGrid( self ):
        """
        Given the full path to an ATLAS model grid, along with values for
        Teff and logg, this routine extracts the values for the specific
        intensity as a function of mu=cos(theta), where theta is the angle
        between the line of site and the emergent radiation. Calling is:

          mu, wav, intensity = atlas.read_grid( model_filepath='filename.pck', \
                                                teff=6000, logg=4.5, vturb=2. )

        Note that the input grids correspond to a given metallicity and
        vturb parameter. So those parameters are controlled by defining
        the model_filepath input appropriately.

        The units of the output variables are:
          mu - unitless
          wav - nm
          intensity - erg/cm**2/s/nm/ster

        Another point to make is that there are some minor issues with the
        formatting of 'new' ATLAS  grids on the Kurucz website. This
        routine will fail on those if you simply download them and feed
        them as input, unchanged. This is because:
          - They have an extra blank line at the start of the file.
          - More troublesome, the last four wavelengths of each grid
            are printed on a single line, which screws up the expected
            structure that this routine requires to read in the file.
        This is 

        """
        nm2micr = 1e-3
        # Row dimensions of the input file:
        if self.newgrid==False:
            nskip = 0 # number of lines to skip at start of file
            nhead = 3 # number of header lines for each grid point
            nwav = 1221 # number of wavelengths for each grid point
        else:
            nskip = 0 # number of lines to skip at start of file
            nhead = 4 # number of header lines for each grid point
            nwav = 1216 # number of wavelengths for each grid point
        nang = 17 # number of angles for each grid point
        # Note: The 'new' model grids don't quite have the 
        # same format, so they won't work for this code.
        print( '\nLimb darkening:\nreading in the model grid...' )
        ifile = open( self.fpath, 'rU' )
        ifile.seek( 0 )
        rows = ifile.readlines()
        ifile.close()
        rows = rows[nskip:]
        nrows = len( rows )
        print( 'Done.' )
        # The angles, where mu=cos(theta):
        self.mus = np.array( rows[nskip+nhead-1].split(), dtype=float )
        # Read in the teff, logg and vturb values
        # for each of the grid points:
        row_ixs = np.arange( nrows )
        header_ixs = row_ixs[ row_ixs%( nhead + nwav )==0 ]
        if self.newgrid==True:
            header_ixs += 1
            header_ixs = header_ixs[:-1]
        ngrid = len( header_ixs )
        teff_grid = np.zeros( ngrid )
        logg_grid = np.zeros( ngrid )
        for i in range( ngrid ):
            header = rows[header_ixs[i]].split()
            teff_grid[i] = float( header[1] )
            logg_grid[i] = header[3]
        # Identify the grid point of interest:
        logg_ixs = ( logg_grid==self.logg )
        teff_ixs = ( teff_grid==self.teff )
        # Extract the intensities at each of the wavelengths
        # as a function of wavelength:
        grid_ix = ( logg_ixs*teff_ixs )
        row_ix = int( header_ixs[grid_ix] )
        grid_lines = rows[row_ix+nhead:row_ix+nhead+nwav]
        grid = []
        for i in range( nwav ):
            grid += [ grid_lines[i].split() ]
        if self.newgrid==True:
            grid=grid[:-1]
        grid = np.array( np.vstack( grid ), dtype=float )
        wavnm = grid[:,0]
        intens = grid[:,1:]
        nmus = len( self.mus )
        for i in range( 1, nmus ):
            intens[:,i] = intens[:,i]*intens[:,0]/100000.
        # Convert the intensities from per unit frequency to
        # per nm in wavelength:
        for i in range( nmus ):
            intens[:,i] /= ( wavnm**2. )
        self.wavmicr = nm2micr*wavnm
        self.intens = intens
        return None
    

class LimbDarkening():

    def __init__( self ):
        self.wavmicr = None
        self.intens = None
        self.mus = None
        self.bandpass_wavmicr = None
        self.bandpass_thput = None
        self.cutonmicr = None
        self.cutoffmicr = None

        
    def Compute( self ):
        wavnm = (1e3)*self.wavmicr
        cutonnm = (1e3)*self.cutonmicr
        cutoffnm = (1e3)*self.cutoffmicr
        bandpass_wavnm = (1e3)*self.bandpass_wavmicr
        ldcoeffs = ld.fit_law( self.mus, wavnm, self.intens, \
                               bandpass_wavnm, plot_fits=False, \
                               passband_sensitivity=self.bandpass_thput, \
                               cuton_wav_nm=cutonnm, cutoff_wav_nm=cutoffnm )
        # TODO = add 3D STAGGER
        self.lin = ldcoeffs['linear']
        self.quad = ldcoeffs['quadratic']
        self.nonlin = ldcoeffs['fourparam_nonlin']
        return None

        
