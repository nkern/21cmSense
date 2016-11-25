"""
calc_sense.py

Repackaging of original calc_sense.py and calc_tsense_2D.py scripts into a class structure
"""
from __future__ import division
import os
import sys
import aipy as a
import numpy as np
import optparse
from scipy import interpolate

__all__ = ['Calc_Sense']

class PS_Funcs:
    """
    Class containing useful functions for calculating power spectra
    """

    #Convert frequency (GHz) to redshift for 21cm line.
    def f2z(self, fq):
        F21 = 1.42040575177
        return (F21 / fq - 1)

    #Multiply by this to convert an angle on the sky to a transverse distance in Mpc/h at redshift z
    def dL_dth(self, z):
        '''[h^-1 Mpc]/radian, from Furlanetto et al. (2006)'''
        return 1.9 * (1./a.const.arcmin) * ((1+z) / 10.)**.2

    #Multiply by this to convert a bandwidth in GHz to a line of sight distance in Mpc/h at redshift z
    def dL_df(self, z, omega_m=0.266):
        '''[h^-1 Mpc]/GHz, from Furlanetto et al. (2006)'''
        return (1.7 / 0.1) * ((1+z) / 10.)**.5 * (omega_m/0.15)**-0.5 * 1e3

    # Multiply by this to convert a baseline length in wavelengths (at the frequency corresponding to 
    # redshift z) into a tranverse k mode in h/Mpc at redshift z
    def dk_du(self, z):
        '''2pi * [h Mpc^-1] / [wavelengths], valid for u >> 1.'''
        return 2*np.pi / dL_dth(z) # from du = 1/dth, which derives from du = d(sin(th)) using the small-angle approx

    #Multiply by this to convert eta (FT of freq.; in 1/GHz) to line of sight k mode in h/Mpc at redshift z
    def dk_deta(self, z):
        '''2pi * [h Mpc^-1] / [GHz^-1]'''
        return 2*np.pi / dL_df(z)

    #scalar conversion between observing and cosmological coordinates
    def X2Y(self, z):
        '''[h^-3 Mpc^3] / [str * GHz]'''
        return dL_dth(z)**2 * dL_df(z)

    #A function used for binning
    def find_nearest(self, array,value):
        idx = (np.abs(array-value)).argmin()
        return idx
    
    # Gridding Function
    def beamgridder(self,xcen,ycen,size):
        crds = np.mgrid[0:size,0:size]
        cen = size/2 - 0.5 # correction for centering
        xcen += cen
        ycen = -1*ycen + cen
        beam = np.zeros((size,size))
        if round(ycen) > size - 1 or round(xcen) > size - 1 or ycen < 0. or xcen <0.:
            return beam
        else:
            beam[round(ycen),round(xcen)] = 1. #single pixel gridder
            return beam

class Calc_Sense(PS_Funcs):
    """
    Interferometer Sensitivity Calculator

    model : string, choose from ['opt','mod','pess'] (default='mod')
        Foreground model can be optimistic (all modes k modes inside the primary field of view are excluded)
        moderate (all k modes inside horizon + buffer are excluded, but all baselines within a uv pixel are added coherently)
        pessimistic (all k modes inside horizon + buffer are excluded, and all baselines are added incoherently)

    buff : float (default=0.1)
        The size of the additive buffer outside the horizon to exclude in the pessimistic and moderate models.

    freq : float (default=0.135)
        The center frequency of the observation in GHz. If you change from the default, be sure to use
        a sensible power spectrum model from that redshift.  Note that many values in the code are calculated
        relative to .150 GHz and are not affected by changing this value.

    eor : string (default='')
        The model epoch of reionization power spectrum.  The code is built to handle output power spectra from 21cmFAST.

    ndays : float (default=180.0)
        The total number of days observed. The default is 180, which is the maximum a particular R.A. can be observed
        in one year if one only observes at night. The total observing time is ndays*n_per_day.

    n_per_day : float (default=6.0)
        The number of good observing hours per day. This corresponds to the size of a low-foreground region in right ascension
        for a drift scanning instrument.  The total observing time is ndays*n_per_day.  Default is 6.
        If simulating a tracked scan, n_per_day should be a multiple of the length of the track
        (i.e. for two three-hour tracks per day, n_per_day should be 6).

    bwidth : float (default=0.008)
        Cosmological bandwidth in GHz.  Note this is not the total instrument bandwidth, but the redshift range that can be
        considered co-eval.

    nchan : int (default=82)
        Integer number of channels across cosmological bandwidth. Defaults to 82, which is equivalent to 1024 channels over 
        100 MHz of bandwidth.  Sets maximum k_parallel that can be probed, but little to no overall effect on sensitivity.

    no_ps : bool (default=False)
        Remove pure north/south baselines (u=0) from the sensitivity calculation. 
        These baselines can potentially have higher systematics, so excluding them represents a conservative choice.
    """

    def __init__(self, model='mod', buff=0.1, freq=0.135, eor='',
                ndays=180.0, n_per_day=6.0, bwidth=0.008, nchan=82, no_ns=False):
        """ initialize class """
        self.model = model
        self.buff = buff
        self.freq = freq
        self.eor = eor
        self.ndays = ndays
        self.n_per_day = n_per_day
        self.bwidth = bwidth
        self.nchan = nchan
        self.no_ns = no_ns

    def make_arrayfile(self, cal_filename, track=None, bl_min=0.0, bl_max=None, verbose=False):
        """
        Make an array file given a calibration file
        """
        #load cal file and read array parameters
        aa = a.cal.get_aa(cal_filename, np.array([.150]))
        nants = len(aa)
        prms = aa.get_arr_params()
        if track:
            obs_duration=60.*track
            name = prms['name']+'track_%.1fhr' % track
        else:
            obs_duration = prms['obs_duration']
            name = prms['name']+'drift'; print name
        dish_size_in_lambda = prms['dish_size_in_lambda']

        # Fiducial Observational Parameters
        t_int = 60. #how many seconds a single visibility has integrated
        cen_jd = 2454600.90911
        start_jd = cen_jd - (1./24)*((obs_duration/t_int)/2)
        end_jd = cen_jd + (1./24)*(((obs_duration-1)/t_int)/2)
        times = np.arange(start_jd,end_jd,(1./24/t_int))
        if verbose == True: print 'Observation duration:', start_jd, end_jd
        fq = .150

        # Main Code
        cnt = 0
        uvbins = {}

        cat = a.src.get_catalog(cal_filename,'z') #create zenith source object
        aa.set_jultime(cen_jd)
        obs_lst = aa.sidereal_time()
        obs_zen = a.phs.RadioFixedBody(obs_lst,aa.lat)
        obs_zen.compute(aa) #observation is phased to zenith of the center time of the drift 

        #find redundant baselines
        bl_len_min = bl_min / (a.const.c/(fq*1e11)) #converts meters to lambda
        bl_len_max = 0.
        for i in xrange(nants):
            if verbose == True: print 'working on antenna %i of %i' % (i, len(aa))
            for j in xrange(nants):
                if i == j: continue #no autocorrelations
                u,v,w = aa.gen_uvw(i,j,src=obs_zen)
                bl_len = np.sqrt(u**2 + v**2)
                if bl_len > bl_len_max: bl_len_max = bl_len
                if bl_len < bl_len_min: continue
                uvbin = '%.1f,%.1f' % (u,v)
                cnt +=1
                if not uvbins.has_key(uvbin): uvbins[uvbin] = ['%i,%i' % (i,j)]
                else: uvbins[uvbin].append('%i,%i' % (i,j))

        if verbose == True: print 'There are %i baseline types' % len(uvbins.keys())

        if verbose == True: print 'The longest baseline is %.2f meters' % (bl_len_max*(a.const.c/(fq*1e11))) #1e11 converts from GHz to cm
        if bl_max:
            bl_len_max = bl_max / (a.const.c/(fq*1e11)) #units of wavelength
            if verbose == True: print 'The longest baseline being included is %.2f m' % (bl_len_max*(a.const.c/(fq*1e11)))
            
        #grid each baseline type into uv plane
        dim = np.round(bl_len_max/dish_size_in_lambda)*2 + 1 # round to nearest odd
        uvsum,quadsum = np.zeros((dim,dim)), np.zeros((dim,dim)) #quadsum adds all non-instantaneously-redundant baselines incoherently
        for cnt, uvbin in enumerate(uvbins):
            if verbose == True: print 'working on %i of %i uvbins' % (cnt+1, len(uvbins))
            uvplane = np.zeros((dim,dim))
            for t in times:
                aa.set_jultime(t)
                lst = aa.sidereal_time()
                obs_zen.compute(aa)
                bl = uvbins[uvbin][0]
                nbls = len(uvbins[uvbin])
                i, j = bl.split(',')
                i, j = int(i), int(j)
                u,v,w = aa.gen_uvw(i,j,src=obs_zen)
                _beam = self.beamgridder(xcen=u/dish_size_in_lambda,ycen=v/dish_size_in_lambda,size=dim)
                uvplane += nbls*_beam
                uvsum += nbls*_beam
            quadsum += (uvplane)**2

        quadsum = quadsum**.5

        if verbose == True: print "Saving file as %s_blmin%0.f_blmax%0.f_arrayfile.npz" % (name, bl_len_min, bl_len_max)

        np.savez('%s_blmin%0.f_blmax%0.f_arrayfile.npz' % (name, bl_len_min, bl_len_max),
            uv_coverage = uvsum,
            uv_coverage_pess = quadsum,
            name = name,
            obs_duration = obs_duration,
            dish_size_in_lambda = dish_size_in_lambda,
            Trx = prms['Trx'],
            t_int = t_int)

    def calc_sense_1D(self, array_filename):
        """
        Calculates expected sensitivity of a 21cm experiment given a 21cm PS and an array file from make_arrayfile()
        """
        #Load in data from array file; see mk_array_file.py for definitions of the parameters
        array = np.load(array_filename)
        name = array['name']
        obs_duration = array['obs_duration']
        dish_size_in_lambda = array['dish_size_in_lambda']
        Trx = array['Trx']
        t_int = array['t_int']
        if self.model == 'pess':
            uv_coverage = array['uv_coverage_pess']
        else:
            uv_coverage = array['uv_coverage']

        # Observations & Cosmology
        h = 0.7
        B = self.bwidth
        z = f2z(self.freq)

        dish_size_in_lambda = dish_size_in_lambda*(self.freq/.150) # linear frequency evolution, relative to 150 MHz
        first_null = 1.22/dish_size_in_lambda #for an airy disk, even though beam model is Gaussian
        bm = 1.13*(2.35*(0.45/dish_size_in_lambda))**2
        kpls = dk_deta(z) * np.fft.fftfreq(self.nchan,B/self.nchan)

        Tsky = 60e3 * (3e8/(self.freq*1e9))**2.55  # sky temperature in mK
        n_lstbins = self.n_per_day*60./obs_duration

        # EOR Model
        #This is a dimensionless power spectrum, i.e., Delta^2
        modelfile = self.eor
        model = np.loadtxt(modelfile)
        mk, mpk = model[:,0]/h, model[:,1] #k, Delta^2(k)
        #note that we're converting from Mpc to h/Mpc

        #interpolation function for the EoR model
        p21 = interpolate.interp1d(mk, mpk, kind='linear')

        # Main Code
        #set up blank arrays/dictionaries
        kprs = []
        #sense will include sample variance, Tsense will be Thermal only
        sense, Tsense = {}, {}

        uv_coverage *= t_int
        SIZE = uv_coverage.shape[0]

        # Cut unnecessary data out of uv coverage: auto-correlations & half of uv plane (which is not statistically independent for real sky)
        uv_coverage[SIZE/2,SIZE/2] = 0.
        uv_coverage[:,:SIZE/2] = 0.
        uv_coverage[SIZE/2:,SIZE/2] = 0.
        if self.no_ns: uv_coverage[:,SIZE/2] = 0.

        #loop over uv_coverage to calculate k_pr
        nonzero = np.where(uv_coverage > 0)
        for iu,iv in zip(nonzero[1], nonzero[0]):
            u, v = (iu - SIZE/2) * dish_size_in_lambda, (iv - SIZE/2) * dish_size_in_lambda
            umag = np.sqrt(u**2 + v**2)
            kpr = umag * dk_du(z)
            kprs.append(kpr)
            #calculate horizon limit for baseline of length umag
            if self.model in ['mod','pess']: hor = dk_deta(z) * umag/self.freq + self.buff
            elif self.model in ['opt']: hor = dk_deta(z) * (umag/self.freq)*np.sin(first_null/2)
            else: print '%s is not a valid foreground model; Aborting...' % self.model; sys.exit()
            if not sense.has_key(kpr):
                sense[kpr] = np.zeros_like(kpls)
                Tsense[kpr] = np.zeros_like(kpls)
            for i, kpl in enumerate(kpls):
                #exclude k_parallel modes contaminated by foregrounds
                if np.abs(kpl) < hor: continue
                k = np.sqrt(kpl**2 + kpr**2)
                if k < min(mk): continue
                #don't include values beyond the interpolation range (no sensitivity anyway)
                if k > np.max(mk): continue
                tot_integration = uv_coverage[iv,iu] * self.ndays
                delta21 = p21(k)
                Tsys = Tsky + Trx
                bm2 = bm/2. #beam^2 term calculated for Gaussian; see Parsons et al. 2014
                bm_eff = bm**2 / bm2 # this can obviously be reduced; it isn't for clarity
                scalar = X2Y(z) * bm_eff * B * k**3 / (2*np.pi**2)
                Trms = Tsys / np.sqrt(2*(B*1e9)*tot_integration)
                #add errors in inverse quadrature
                sense[kpr][i] += 1./(scalar*Trms**2 + delta21)**2
                Tsense[kpr][i] += 1./(scalar*Trms**2)**2

        #bin the result in 1D
        delta = dk_deta(z)*(1./B) #default bin size is given by bandwidth
        kmag = np.arange(delta,np.max(mk),delta)

        kprs = np.array(kprs)
        sense1d = np.zeros_like(kmag)
        Tsense1d = np.zeros_like(kmag)
        for ind, kpr in enumerate(sense.keys()):
            #errors were added in inverse quadrature, now need to invert and take square root to have error bars; also divide errors by number of indep. fields
            sense[kpr] = sense[kpr]**-.5 / np.sqrt(n_lstbins)
            Tsense[kpr] = Tsense[kpr]**-.5 / np.sqrt(n_lstbins)
            for i, kpl in enumerate(kpls):
                k = np.sqrt(kpl**2 + kpr**2)
                if k > np.max(mk): continue
                #add errors in inverse quadrature for further binning
                sense1d[find_nearest(kmag,k)] += 1./sense[kpr][i]**2
                Tsense1d[find_nearest(kmag,k)] += 1./Tsense[kpr][i]**2

        #invert errors and take square root again for final answer
        for ind,kbin in enumerate(sense1d):
            sense1d[ind] = kbin**-.5
            Tsense1d[ind] = Tsense1d[ind]**-.5

        #save results to output npz
        n.savez('%s_%s_%.3f.npz' % (name,opts.model,opts.freq),ks=kmag,errs=sense1d,T_errs=Tsense1d)

        #calculate significance with least-squares fit of amplitude
        A = p21(kmag)
        M = p21(kmag)
        wA, wM = A * (1./sense1d), M * (1./sense1d)
        wA, wM = n.matrix(wA).T, n.matrix(wM).T
        amp = (wA.T*wA).I * (wA.T * wM)
        #errorbars
        Y = n.float(amp) * wA
        dY = wM - Y
        s2 = (len(wM)-1)**-1 * (dY.T * dY)
        X = n.matrix(wA).T * n.matrix(wA)
        err = n.sqrt((1./n.float(X)))
        print 'total snr = ', amp/err


    def calc_sense_2D(self):
        """
        """
        pass





