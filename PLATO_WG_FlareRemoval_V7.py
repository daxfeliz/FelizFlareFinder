# # Code Version 7 (V7.1): July 22nd, 2023
__version__ = "7"

#modules required:
import numpy as np
import pandas as pd
import lightkurve as lk

import os,sys

import warnings
# warnings.filterwarnings(action='once') #useful to see a warning once but that's it
warnings.simplefilter("ignore", category=PendingDeprecationWarning)
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
    
import matplotlib
# matplotlib.use('Agg') #if using on cluster with no display, make sure to turn off pltshow() !
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats


# get lightcurve from public data based on ID, mission and mode of observation (cadence = long or short)
def TPF_to_LC(tpf):
    target_mask = tpf.create_threshold_mask(threshold=2.5, reference_pixel='center')
    n_target_pixels = target_mask.sum()
#     target_lc = tpf.to_lightcurve(aperture_mask=target_mask)
    target_lc = tpf.to_lightcurve(aperture_mask=tpf.pipeline_mask)
    n_target_pixels = tpf.pipeline_mask.sum()
    background_mask = ~tpf.create_threshold_mask(threshold=0.001, reference_pixel=None)
    n_background_pixels = background_mask.sum()

    
    background_lc_per_pixel = tpf.to_lightcurve(aperture_mask=background_mask) / n_background_pixels
    background_estimate_lc = background_lc_per_pixel * n_target_pixels
    
    cnew=target_lc - background_estimate_lc.flux
    #remove bad data points using quality flags
    q = cnew.quality!=0
    cnew=cnew[~q]
    cnew=cnew.remove_nans()    #.remove_outliers() #<---- this truncates transits!!!
    corrected_lc = cnew.normalize()
    
    
    return corrected_lc
def get_mission_ID(starname,mission):
    from astroquery.mast import Catalogs
    from astropy.coordinates import SkyCoord, Angle
    import astropy.units as u
    import numpy as np
    cone = 5*u.arcsec    
    #
    if mission=='TESS':
        catalog='TIC'
    if starname is not int or starname is not float:
        result = Catalogs.query_object(starname, catalog = catalog, radius=Angle(cone, "arcsec"))[0]
        try:
            if int(result['duplicate_id'])!=np.nan:
                ID = int(result['duplicate_id'])
        except MaskError:
            ID = int(result['ID'])
    return ID

def get_lc(ID,mission,cadence,downloadpath):
    target=mission+' '+str(int(ID))
    lc=lk.search.search_lightcurvefile(target, cadence=cadence,\
                                       mission=mission).download_all(download_dir=downloadpath)
    ###
    if lc==None:
        print('nope. using custom.')
        if cadence=='short':
            try:
                tpfs = lk.search_targetpixelfile(target,cadence='short',\
                                                 mission="Kepler").download_all(download_dir=downloadpath)
                for x in range(len(tpfs)):
                    tpf=tpfs[x]
                    lc=TPF_to_LC(tpf)
                    if x>0:
                        lc=lc.append(TPF_to_LC(tpf))
                LC=pd.DataFrame({'time':lc.time.value,'flux':lc.flux.value,\
                                 'flux_err':lc.flux_err.value})  
                nanmask = np.where(np.isfinite(np.array(LC['flux']))==True)[0]
                t=np.array(LC['time'])[nanmask]
                f=np.array(LC['flux'])[nanmask]
                fe=np.array(LC['flux_err'])[nanmask]
                #excluding NaNs
                finalLC = pd.DataFrame({'time':t,'flux':f,\
                                        'flux_err':fe})
                return finalLC
            except TypeError as TE:
                print(target+' has no data at '+str(cadence)+' cadence!')
                pass    
        if cadence=='long':
            try:
                lc=lk.search.search_lightcurvefile(target, cadence=cadence,\
                                       mission=mission).download_all(download_dir=downloadpath)
                for x in range(len(tpfs)):
                    tpf=tpfs[x]
                    lc=TPF_to_LC(tpf)
                    if x>0:
                        lc=lc.append(TPF_to_LC(tpf))
                LC=pd.DataFrame({'time':lc.time.value,'flux':lc.flux.value,\
                                 'flux_err':lc.flux_err.value})  
                nanmask = np.where(np.isfinite(np.array(LC['flux']))==True)[0]
                t=np.array(LC['time'])[nanmask]
                f=np.array(LC['flux'])[nanmask]
                fe=np.array(LC['flux_err'])[nanmask]
                #excluding NaNs
                finalLC = pd.DataFrame({'time':t,'flux':f,\
                                        'flux_err':fe})
                return finalLC
            except TypeError as TE:
                print(target+' has no data at '+str(cadence)+' cadence!')
                pass            
    else:
        try:
            LC=lc.PDCSAP_FLUX.stitch()
            LCdf=pd.DataFrame({'time':LC.time.value,'flux':LC.flux.value,'flux_err':LC.flux_err.value})
#             LCdf.to_csv(downloadpath+str(target)+'_LC.csv',index=False)
            nanmask = np.where(np.isfinite(LC.flux)==True)[0]
            t=LC.time.value[nanmask]
            f=LC.flux.value[nanmask]
            fe=LC.flux_err.value[nanmask]       
            finalLC = pd.DataFrame({'time':t,'flux':f,\
                                        'flux_err':fe})
            return finalLC
        except ValueError:
            print(target+' is crowded!')      
            LC=None
            pass
    if LC is None:
        print('bad target!')
    print('')  

#for smoothing data using a user-defined window
def BWMC_auto_OLD(window_size_in_hrs,lc): #bt = break tolerance, pipeline uses window_size/2.0
    import lightkurve as lk
    from wotan import flatten
    import time as clock
    start=clock.time()
    window_size = window_size_in_hrs/24.0 # The length of the filter window in units of time (days)
    time=np.array(lc['time'])#.to_list())
    flux=np.array(lc['flux'])#.to_list())
    flux_error=np.array(lc['flux_err'])#.to_list())
    print('len check: ',len(time),len(flux),len(flux_error))
    flatten_lc, trend_lc = flatten(time, flux, window_length=window_size, return_trend=True, break_tolerance=window_size/2.0,method='biweight',robust=True)
    T=time
    F=flatten_lc
    FE=flux_error

    #excluding NaNs
    nanmask = np.where(np.isfinite(F)==True)[0]
    T = T[nanmask]
    F = F[nanmask]
    FE =FE[nanmask]
    trend_lc=trend_lc[nanmask]
    
    #forcing same size outputs
    m=np.where(np.in1d(T,time)==True)[0]
    t2=time[m]
    f2=flux[m]
    fe2=flux_error[m]
    
    lc=pd.DataFrame({'time':t2,'flux':f2,'flux_err':fe2})
    newlc=pd.DataFrame({'time':T,'flux':F,'flux_err':FE})
    print('len check after detrending: New ',len(np.array(newlc['time'])),len(np.array(newlc['flux'])),' input ',len(np.array(lc['time'])),len(np.array(lc['flux'])))
    
    end=clock.time()
    runtime=end-start
    if runtime>60:
        print('smoothing runtime: ',np.round(runtime/60,2),' minutes')
    if runtime<60:
        print('smoothing runtime: ',np.round(runtime,2),' seconds')
        
    return lc, newlc, trend_lc #time, raw ,detrended, error, trend


def BWMC_auto(window_size_in_hrs,lc,filter_type='medfilt'): #bt = break tolerance, pipeline uses window_size/2.0
    import lightkurve as lk
    from wotan import flatten
    import time as clock
    start=clock.time()
    window_size = window_size_in_hrs/24.0 # The length of the filter window in units of time (days)
    time=np.array(lc['time'].to_list())
    flux=np.array(lc['flux'].to_list())
    flux_error=np.array(lc['flux_err'].to_list())
    print('len check: ',len(time),len(flux),len(flux_error))
    
    if filter_type=='biweight':
        flatten_lc, trend_lc = flatten(time, flux, window_length=window_size, return_trend=True, \
                                       break_tolerance=window_size/2.0,edge_cutoff=window_size/2.0,\
                                       method=filter_type,robust=True)
    if filter_type=='medfilt':
        cadence = np.nanmedian(np.diff(time))
        def round_up_to_odd(f):
            return int(np.ceil(f) // 2 * 2 + 1)
        Npts = round_up_to_odd(int((window_size_in_hrs/24)/cadence))

        flatten_lc, trend_lc = flatten(time, flux, window_length=Npts, method=filter_type, \
                                       return_trend=True,robust=True)
    print(filter_type)
    T=time
    F=flatten_lc
    FE=flux_error

    #excluding NaNs
    nanmask = np.where(np.isfinite(F)==True)[0]
    T = T[nanmask]
    F = F[nanmask]
    FE =FE[nanmask]
    trend_lc=trend_lc[nanmask]
    
    #forcing same size outputs
    m=np.where(np.in1d(T,time)==True)[0]
    t2=time[m]
    f2=flux[m]
    fe2=flux_error[m]
    
    lc=pd.DataFrame({'time':t2,'flux':f2,'flux_err':fe2})
    newlc=pd.DataFrame({'time':T,'flux':F,'flux_err':FE})
    print('len check after detrending: New ',len(np.array(newlc['time'])),len(np.array(newlc['flux'])),' input ',len(np.array(lc['time'])),len(np.array(lc['flux'])))
    
    end=clock.time()
    runtime=end-start
    if runtime>60:
        print('smoothing runtime: ',np.round(runtime/60,2),' minutes')
    if runtime<60:
        print('smoothing runtime: ',np.round(runtime,2),' seconds')
        
    return lc, newlc, trend_lc #time, raw ,detrended, error, trend

def characterize_timescale(t, tpeak, fwhm):
    return (t-tpeak)/fwhm

def find_nearest(array,value):
    import math
    idx = np.searchsorted(array, value, side="left")
    #changed idx from > 0 to >= 0 
    if idx >= 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])): 
        return [idx-1]
    else:
        return [idx]
    
#for counting consecutive outliers (give indices as inputs)
def consecutive(data, stepsize=1):
    consec=np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
    c=list(filter(lambda x : len(x) > 1, consec))
    newc=[]
    for x in range(len(c)):
        newc=np.append(newc,c[x])
    consec = np.sort(np.array([np.int64(x) for x in newc]))
    return consec    


def FINDFLARE(flux, error, N1=3, N2=2, N3=3):
    '''
    The algorithm for local changes due to flares defined by
    S. W. Chang et al. (2015), Eqn. 3a-d
    http://arxiv.org/abs/1510.01005
    Note: these equations were originally in magnitude units, i.e. smaller
    values are increases in brightness. The signs have been changed, but
    coefficients have not been adjusted to change from log(flux) to flux.
    Parameters:
    ----------
    flux : numpy array
        data to search over
    error : numpy array
        errors corresponding to data.
    N1 : int, optional
        Coefficient from original paper (Default is 3 in paper, 3 here)
        How many times above the stddev is required.
    N2 : int, optional
        Coefficient from original paper (Default is 1 in paper, 2 here)
        How many times above the stddev and uncertainty is required
    N3 : int, optional
        Coefficient from original paper (Default is 3)
        The number of consecutive points required to flag as a flare
    Return:
    ------------
    isflare : numpy array of booleans
        datapoints are flagged with 1 if they belong to a flare candidate
    '''
    from scipy import stats
    isflare = np.zeros_like(flux, dtype='bool')
    median=np.nanmedian(flux)# stats.median_abs_deviation(flux) 
    mean=np.nanmean(flux)
    sigma = stats.median_abs_deviation(flux) # np.nanstd(flux) #error
    T0 = flux - median # excursion should be positive #"N0"
    T1 = np.abs(flux - median) / sigma #N1
    T2 = np.abs(flux - median + error) / sigma #N2
       
    # apply thresholds N0-N2:
    pass_thresholds = np.where((T0 > 0) & (T1 > N1) & (T2 > N2))
    
    #array of indices where thresholds are exceeded:
    is_pass_thresholds = np.zeros_like(flux)
    is_pass_thresholds[pass_thresholds] = 1

    # Need to find cumulative number of points that pass_thresholds
    # Counted in reverse!
    # Examples reverse_counts = [0 0 0 3 2 1 0 0 1 0 4 3 2 1 0 0 0 1 0 2 1 0]
    #                 isflare = [0 0 0 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0]

    reverse_counts = np.zeros_like(flux, dtype='int')
    for k in range(2, len(flux)):
        reverse_counts[-k] = (is_pass_thresholds[-k]
                                * (reverse_counts[-(k-1)]
                                + is_pass_thresholds[-k]))
                                
    # find flare start where values in reverse_counts switch from 0 to >=N3
    istart_i = np.where((reverse_counts[1:] >= N3) &
                        (reverse_counts[:-1] - reverse_counts[1:] < 0))[0] + 1
                        
    # use the value of reverse_counts to determine how many points away stop is
    istop_i = istart_i + (reverse_counts[istart_i])
    isflare = np.zeros_like(flux, dtype='bool')
    
    for (l,r) in list(zip(istart_i,istop_i)):
        isflare[l:r+1] = True
    return isflare

def aflare1(t, tpeak, fwhm, ampl, upsample=True, uptime=10,version='Davenport'): 
    from scipy.stats import binned_statistic
    from scipy import special
    #
    if version=='Davenport':
        _fr = [1.00000, 1.94053, -0.175084, -2.24588, -1.12498] #rise  coeffs
        _fd = [0.689008, -1.60053, 0.302963, -0.278318]         #decay coeffs


        if upsample:
            
            dt = np.nanmedian(np.diff(t))
            timeup = np.linspace(min(t)-dt, max(t)+dt, t.size * uptime)

            flareup = np.piecewise(timeup, [(timeup<= tpeak) * (timeup-tpeak)/fwhm > -1.,
                                            (timeup > tpeak)],
                                        [lambda x: (_fr[0]+                       # 0th order
                                                    _fr[1]*((x-tpeak)/fwhm)+      # 1st order
                                                    _fr[2]*((x-tpeak)/fwhm)**2.+  # 2nd order
                                                    _fr[3]*((x-tpeak)/fwhm)**3.+  # 3rd order
                                                    _fr[4]*((x-tpeak)/fwhm)**4. ),# 4th order
                                         lambda x: (_fd[0]*np.exp( ((x-tpeak)/fwhm)*_fd[1] ) +
                                                    _fd[2]*np.exp( ((x-tpeak)/fwhm)*_fd[3] ))]
                                        ) * np.abs(ampl) # amplitude

            # and now downsample back to the original time...
            ## this way might be better, but makes assumption of uniform time bins
            # flare = np.nanmean(flareup.reshape(-1, uptime), axis=1)

            ## This way does linear interp. back to any input time grid
    #         flare = np.interp(t, timeup, flareup)

            ## this way uses "binned statistic"
            downbins = np.concatenate((t-dt/2.,[max(t)+dt/2.]))
            flare,_,_ = binned_statistic(timeup, flareup, statistic='mean', bins=downbins) #original method
    #         flare,_,_ = binned_statistic(timeup, flareup, statistic='median', bins=downbins)        
            #
        else:
            #
            flare = np.piecewise(t, [(t<= tpeak) * (t-tpeak)/fwhm > -1., (t > tpeak)],
                                    [lambda x: (_fr[0]+                       # 0th order
                                                _fr[1]*((x-tpeak)/fwhm)+      # 1st order
                                                _fr[2]*((x-tpeak)/fwhm)**2.+  # 2nd order
                                                _fr[3]*((x-tpeak)/fwhm)**3.+  # 3rd order
                                                _fr[4]*((x-tpeak)/fwhm)**4. ),# 4th order
                                     lambda x: (_fd[0]*np.exp( ((x-tpeak)/fwhm)*_fd[1] ) +
                                                _fd[2]*np.exp( ((x-tpeak)/fwhm)*_fd[3] ))]
                                    ) * np.abs(ampl) # amplitude
    if version=='Tovar':#needs testing, currently produces flat models
        # from https://github.com/lupitatovar/Llamaradas-Estelares/blob/main/Flare_model.py
        def flare_eqn(t,tpeak,fwhm,ampl):
            '''
            The equation that defines the shape for the Continuous Flare Model
            '''
            
            #Values were fit & calculated using MCMC 256 walkers and 30000 steps

            A,B,C,D1,D2,f1 = [0.9687734504375167,-0.251299705922117,0.22675974948468916,
                              0.15551880775110513,1.2150539528490194,0.12695865022878844]

            # We include the corresponding errors for each parameter from the MCMC analysis

            A_err,B_err,C_err,D1_err,D2_err,f1_err = [0.007941622683556804,0.0004073709715788909,0.0006863488251125649, 0.0013498012884345656,0.00453458098656645,0.001053149344530907 ]
            #
            if (np.round(np.nanmedian(f1),0))==1: #is THIS why models are flat???
                f2 = 1-f1
            else:
                f2 = f1
            #
            eqn = ((1 / 2) * np.sqrt(np.pi) * A * C * f1 * np.exp(-D1 * t + ((B / C) + (D1 * C / 2)) ** 2) * special.erfc(((B - t) / C) + (C * D1 / 2))) + ((1 / 2) * np.sqrt(np.pi) * A * C * f2 * np.exp(-D2 * t+ ((B / C) + (D2 * C / 2)) ** 2) * special.erfc(((B - t) / C) + (C * D2 / 2)))
            #
            return eqn * ampl
        ###
#         t_new = (t-tpeak)/fwhm
        if upsample:
            dt = np.nanmedian(np.diff(t))
            timeup = np.linspace(min(t)-dt, max(t)+dt, t.size * uptime)

            flareup = flare_eqn(timeup,tpeak,fwhm,ampl)

            # and now downsample back to the original time...

            ## this way uses "binned statistic"
            downbins = np.concatenate((t-dt/2.,[max(t)+dt/2.]))
            flare,_,_ = binned_statistic(timeup, flareup, statistic='mean', bins=downbins)
        else:
            flare = flare_eqn(t_new,tpeak,fwhm,ampl)
        
    
    return flare

def aflare1_ALT(t, tpeak, fwhm, ampl, upsample=True, uptime=10,version='Davenport'): 
    from scipy.stats import binned_statistic
    from scipy import special
    #
    if version=='Davenport':
        _fr = [1.00000, 1.94053, -0.175084, -2.24588, -1.12498] #rise  coeffs
        _fd = [0.689008, -1.60053, 0.302963, -0.278318]         #decay coeffs


        if upsample:
            
            dt = np.nanmedian(np.diff(t))
            timeup = np.linspace(min(t)-dt, max(t)+dt, t.size * uptime)

            flareup = np.piecewise(timeup, [(timeup<= tpeak) * (timeup-tpeak)/fwhm > -1.,
                                            (timeup > tpeak)],
                                        [lambda x: (_fr[0]+                       # 0th order
                                                    _fr[1]*((x-tpeak)/fwhm)+      # 1st order
                                                    _fr[2]*((x-tpeak)/fwhm)**2.+  # 2nd order
                                                    _fr[3]*((x-tpeak)/fwhm)**3.+  # 3rd order
                                                    _fr[4]*((x-tpeak)/fwhm)**4. ),# 4th order
                                         lambda x: (_fd[0]*np.exp( ((x-tpeak)/fwhm)*_fd[1] ) +
                                                    _fd[2]*np.exp( ((x-tpeak)/fwhm)*_fd[3] ))]
                                        ) * np.abs(ampl) # amplitude

            # and now downsample back to the original time...
            ## this way might be better, but makes assumption of uniform time bins
            # flare = np.nanmean(flareup.reshape(-1, uptime), axis=1)

            ## This way does linear interp. back to any input time grid
    #         flare = np.interp(t, timeup, flareup)

            ## this way uses "binned statistic"
            downbins = np.concatenate((t-dt/2.,[max(t)+dt/2.]))
            flare,_,_ = binned_statistic(timeup, flareup, statistic='mean', bins=downbins) #original method
    #         flare,_,_ = binned_statistic(timeup, flareup, statistic='median', bins=downbins)        
            #
        else:
            #
#             flare = np.piecewise(t, [(t<= tpeak) * (t-tpeak)/fwhm > -1., (t > tpeak)],
#                                     [lambda x: (_fr[0]+                       # 0th order
#                                                 _fr[1]*((x-tpeak)/fwhm)+      # 1st order
#                                                 _fr[2]*((x-tpeak)/fwhm)**2.+  # 2nd order
#                                                 _fr[3]*((x-tpeak)/fwhm)**3.+  # 3rd order
#                                                 _fr[4]*((x-tpeak)/fwhm)**4. ),# 4th order
#                                      lambda x: (_fd[0]*np.exp( ((x-tpeak)/fwhm)*_fd[1] ) +
#                                                 _fd[2]*np.exp( ((x-tpeak)/fwhm)*_fd[3] ))]
#                                     ) * np.abs(ampl) # amplitude
            flare = np.piecewise(t, [(t<= tpeak) * (t-tpeak)/fwhm > -1.,
                                            (t > tpeak)],
                                        [lambda x: (_fr[0]+                       # 0th order
                                                    _fr[1]*((x-tpeak)/fwhm)+      # 1st order
                                                    _fr[2]*((x-tpeak)/fwhm)**2.+  # 2nd order
                                                    _fr[3]*((x-tpeak)/fwhm)**3.+  # 3rd order
                                                    _fr[4]*((x-tpeak)/fwhm)**4. ),# 4th order
                                         lambda x: (_fd[0]*np.exp( ((x-tpeak)/fwhm)*_fd[1] ) +
                                                    _fd[2]*np.exp( ((x-tpeak)/fwhm)*_fd[3] ))]
                                        ) * np.abs(ampl) # amplitude
    if version=='Tovar':#needs testing, currently produces flat models
        # from https://github.com/lupitatovar/Llamaradas-Estelares/blob/main/Flare_model.py
        def flare_eqn(t,tpeak,fwhm,ampl):
            '''
            The equation that defines the shape for the Continuous Flare Model
            '''
            
            #Values were fit & calculated using MCMC 256 walkers and 30000 steps

            A,B,C,D1,D2,f1 = [0.9687734504375167,-0.251299705922117,0.22675974948468916,
                              0.15551880775110513,1.2150539528490194,0.12695865022878844]

            # We include the corresponding errors for each parameter from the MCMC analysis

            A_err,B_err,C_err,D1_err,D2_err,f1_err = [0.007941622683556804,0.0004073709715788909,0.0006863488251125649, 0.0013498012884345656,0.00453458098656645,0.001053149344530907 ]
            #
            if (np.round(np.nanmedian(f1),0))==1:
                f2 = 1-f1
            else:
                f2 = f1
            #
            eqn = ((1 / 2) * np.sqrt(np.pi) * A * C * f1 * np.exp(-D1 * t + ((B / C) + (D1 * C / 2)) ** 2) * special.erfc(((B - t) / C) + (C * D1 / 2))) + ((1 / 2) * np.sqrt(np.pi) * A * C * f2 * np.exp(-D2 * t+ ((B / C) + (D2 * C / 2)) ** 2) * special.erfc(((B - t) / C) + (C * D2 / 2)))
            #
            return eqn * ampl
        ###
        #t_new = (t-tpeak)/fwhm
        if upsample:
            #dt = np.nanmedian(np.diff(np.abs(t_new)))
            #timeup = np.linspace(min(t_new) - dt, max(t_new) + dt, t_new.size * uptime)
            dt = np.nanmedian(np.diff(t))
            timeup = np.linspace(min(t)-dt, max(t)+dt, t.size * uptime)            

            flareup = flare_eqn(timeup,tpeak,fwhm,ampl)

            # and now downsample back to the original time...

            #downbins = np.concatenate((t_new - dt / 2.,[max(t_new) + dt / 2.]))
            downbins = np.concatenate((t-dt/2.,[max(t)+dt/2.]))
            flare,_,_ = binned_statistic(timeup, flareup, statistic='mean',bins=downbins)
        else:
            flare = flare_eqn(t_new,tpeak,fwhm,ampl)
        
    
    return flare



def model_flares_ALT_OLD(time_in_window,flux_in_window,error_in_window,N3,localT,verbose,t_char_min,t_char_max,version):
    if len(time_in_window)<N3:
        return np.full_like(time_in_window,np.nan),\
        np.full_like(flux_in_window,np.nan),\
        np.nan,np.nan,np.nan
    else:    
        from scipy.optimize import curve_fit, minimize
        # get initial guesses of properties of flare in window
        ampl = np.max(flux_in_window)
        #f_half = np.where((flux_in_window >= ampl*0.5))
        f_half = np.where((flux_in_window/np.nanmedian(flux_in_window) >= np.nanmax(ampl/np.nanmedian(flux_in_window))*0.5))
#         try:
        fwhm = np.max(time_in_window[f_half]) - np.min(time_in_window[f_half])   
        cadence = np.nanmedian(np.diff(time_in_window))
        if fwhm<cadence:
            fwhm=cadence  
        #initial flare parameter guesses
        p0 = [time_in_window[np.argmax(flux_in_window)], fwhm, ampl]
        #
        #optimize initial fit with Aflare
        try:
            mf=10000
            popt, pcov = curve_fit(aflare1, time_in_window, flux_in_window, p0=p0, maxfev=mf)
        except RuntimeError as RE:
            print(RE)
            print('trying with 10x higher maxfev')
            mf=mf*10
            popt, pcov = curve_fit(aflare1, time_in_window, flux_in_window, p0=p0, maxfev=mf,method='dogbox')
        #
        # the decay region:
        # do from PEAK to 5X the FWHM timescale (i.e. don't fit the out-of-flare region)
        # the flare fit region (before and after peak flare) 
        flare_region = np.where((time_in_window >= popt[0] - t_char_min*popt[1]) & (time_in_window <= popt[0] + t_char_max*popt[1]))        
        ###
        ###
        flare_times = time_in_window[flare_region] #<-- let's not use this for now
        ###
        ###
        try:
            model = aflare1(flare_times, *popt,version=version)
        except ValueError as e:
            print(e)
            return np.full_like(time_in_window,np.nan),\
        np.full_like(flux_in_window,np.nan),\
        np.nan,np.nan,np.nan
        #
        #
        out_tpeak, out_fwhm, out_ampl = popt[0],popt[1],popt[2]
        #
        # EXPERIMENT (WHAT IF WE ALLOW THIS IN...)
        #
#         if (out_ampl < localT) or (out_fwhm > 0.1):
#             print('something went wrong on this event:')
#             if out_ampl < localT:
#                 print('out_ampl ',str(out_ampl),' < localT = ',localT)
#             if out_fwhm > 0.1:
#                 print('out_fwhm',str(out_fwhm),' > 0.1')
#                 #print('out_fpeak',out_ampl,'out_fwhm',out_fwhm)
#             return np.full_like(time_in_window,np.nan), np.full_like(flux_in_window,np.nan),np.nan,np.nan,np.nan
#         else:
        #
        # EXPERIMENT (WHAT IF WE ALLOW THIS IN...)
        #
        if verbose==True:
            print('model flare len check',len(time_in_window),len(model))               
        return flare_times, model, out_tpeak, out_fwhm, out_ampl
#         except ValueError as VE:            
#                 print(VE)
#                 return np.full_like(time_in_window,np.nan),\
#             np.full_like(flux_in_window,np.nan),\
#             np.nan,np.nan,np.nan

def model_flares_ALT3(time_in_window,flux_in_window,error_in_window,N3,localT,verbose,t_char_min,t_char_max,version):
    if len(time_in_window)<N3:
        return np.full_like(time_in_window,np.nan),\
        np.full_like(flux_in_window,np.nan),\
        np.nan,np.nan,np.nan
    else:    
        from scipy.optimize import curve_fit, minimize
        # get initial guesses of properties of flare in window
        ampl = np.max(flux_in_window)
        #f_half = np.where((flux_in_window >= ampl*0.5))
        f_half = np.where((flux_in_window/np.nanmedian(flux_in_window) >= np.nanmax(ampl/np.nanmedian(flux_in_window))*0.5))
#         try:
        fwhm = np.max(time_in_window[f_half]) - np.min(time_in_window[f_half])   
        cadence = np.nanmedian(np.diff(time_in_window))
        if fwhm<cadence:
            fwhm=cadence  
        #initial flare parameter guesses
        p0 = [time_in_window[np.argmax(flux_in_window)], fwhm, flux_in_window[np.argmax(flux_in_window)]]
        t0_bounds=(time_in_window[np.argmax(flux_in_window)]-cadence,time_in_window[np.argmax(flux_in_window)]+cadence)
        fwhm_bounds=(fwhm-fwhm*(1/100),fwhm+fwhm*(1/100))
        fpeak_bounds=(ampl-ampl*(1/100),ampl+ampl*(1/100))
# #         bounds = ((np.min(t0_bounds),np.min(fwhm_bounds),np.min(fpeak_bounds)) ,\
# #                   (np.max(t0_bounds),np.max(fwhm_bounds),np.max(fpeak_bounds)))        
        bounds = ((-np.inf,-np.inf,-np.inf),(np.inf,np.inf,np.inf))
        #
        #optimize initial fit with Aflare
        try:
            mf=10000
            popt, pcov = curve_fit(aflare1_ALT, time_in_window, flux_in_window, \
                                   sigma=np.nanstd(flux_in_window)*np.ones_like(flux_in_window),\
                                   p0=p0, maxfev=mf,method='dogbox',\
                                   bounds=bounds)
        except RuntimeError as RE:
            print(RE)
            print('trying with 10x higher maxfev')
            mf=mf*10
            popt, pcov = curve_fit(aflare1_ALT, time_in_window, flux_in_window, \
                                   sigma=np.nanstd(flux_in_window)*np.ones_like(flux_in_window), \
                                   p0=p0, maxfev=mf,\
                                   method='dogbox',\
                                   bounds=bounds)
        #
        # the decay region:
        # do from PEAK to 5X the FWHM timescale (i.e. don't fit the out-of-flare region)
        # the flare fit region (before and after peak flare) 
        flare_region = np.where((time_in_window >= popt[0] - t_char_min*popt[1]) & (time_in_window <= popt[0] + t_char_max*popt[1]))        
        ###
        flare_times = time_in_window#[flare_region]
#         flare_times = np.linspace(np.min(flare_times),np.max(flare_times),num=len(time_in_window[flare_region]))
        ###
        try:
            model = aflare1_ALT(flare_times, *popt,version=version)
        except ValueError as e:
            print(e)
            return np.full_like(time_in_window,np.nan),\
        np.full_like(flux_in_window,np.nan),\
        np.nan,np.nan,np.nan
        #
        #
        out_tpeak, out_fwhm, out_ampl = popt[0],popt[1],popt[2]
        #
#         if (out_ampl < localT) or (out_fwhm > 0.1):
#             print('something went wrong on this event:')
#             if out_ampl < localT:
#                 print('out_ampl ',str(out_ampl),' < localT = ',localT)
#             if out_fwhm > 0.1:
#                 print('out_fwhm',str(out_fwhm),' > 0.1')
#                 #print('out_fpeak',out_ampl,'out_fwhm',out_fwhm)
#             return np.full_like(time_in_window,np.nan), np.full_like(flux_in_window,np.nan),np.nan,np.nan,np.nan
#         else:
        if verbose==True:
            print('model flare len check',len(time_in_window),len(model))               
        return flare_times, model, out_tpeak, out_fwhm, out_ampl
#         except ValueError as VE:            
#                 print(VE)
#                 return np.full_like(time_in_window,np.nan),\
#             np.full_like(flux_in_window,np.nan),\
#             np.nan,np.nan,np.nan

def model_flares_ALT(time_in_window,flux_in_window,error_in_window,N3,localT,verbose,t_char_min,t_char_max,version):
    if len(time_in_window)<N3:
        return np.full_like(time_in_window,np.nan),\
        np.full_like(flux_in_window,np.nan),\
        np.nan,np.nan,np.nan
    else:    
        from scipy.optimize import curve_fit, minimize
        # get initial guesses of properties of flare in window
        ampl = np.max(flux_in_window)
        #f_half = np.where((flux_in_window >= ampl*0.5))
        f_half = np.where((flux_in_window/np.nanmedian(flux_in_window) >= np.nanmax(ampl/np.nanmedian(flux_in_window))*0.5))
#         try:
        fwhm = np.max(time_in_window[f_half]) - np.min(time_in_window[f_half])   
        cadence = np.nanmedian(np.diff(time_in_window))
        if fwhm<cadence:
            fwhm=cadence  
        #initial flare parameter guesses
        p0 = [time_in_window[np.argmax(flux_in_window)], fwhm, flux_in_window[np.argmax(flux_in_window)]]
        t0_bounds=(time_in_window[np.argmax(flux_in_window)]-cadence,time_in_window[np.argmax(flux_in_window)]+cadence)
        fwhm_bounds=(fwhm-fwhm*(1/100),fwhm+fwhm*(1/100))
        fpeak_bounds=(ampl-ampl*(1/100),ampl+ampl*(1/100))
# #         bounds = ((np.min(t0_bounds),np.min(fwhm_bounds),np.min(fpeak_bounds)) ,\
# #                   (np.max(t0_bounds),np.max(fwhm_bounds),np.max(fpeak_bounds)))        
        bounds = ((-np.inf,-np.inf,-np.inf),(np.inf,np.inf,np.inf))
        #
        #optimize initial fit with Aflare
        try:
            mf=10000
            popt, pcov = curve_fit(aflare1, time_in_window, flux_in_window, \
                                   sigma=np.nanstd(flux_in_window)*np.ones_like(flux_in_window),\
                                   p0=p0, maxfev=mf,method='dogbox',\
                                   bounds=bounds)
        except RuntimeError as RE:
            print(RE)
            print('trying with 10x higher maxfev')
            mf=mf*10
            popt, pcov = curve_fit(aflare1, time_in_window, flux_in_window, \
                                   sigma=np.nanstd(flux_in_window)*np.ones_like(flux_in_window), \
                                   p0=p0, maxfev=mf,\
                                   method='dogbox',\
                                   bounds=bounds)
        #
        # the decay region:
        # do from PEAK to 5X the FWHM timescale (i.e. don't fit the out-of-flare region)
        # the flare fit region (before and after peak flare) 
        flare_region = np.where((time_in_window >= popt[0] - t_char_min*popt[1]) & (time_in_window <= popt[0] + t_char_max*popt[1]))        
        ###
        flare_times = time_in_window #[flare_region]
#         flare_times = np.linspace(np.min(flare_times),np.max(flare_times),num=len(time_in_window[flare_region]))
        ###
        try:
            model = aflare1(flare_times, *popt,version=version)
        except ValueError as e:
            print(e)
            return np.full_like(time_in_window,np.nan),\
        np.full_like(flux_in_window,np.nan),\
        np.nan,np.nan,np.nan
        #
        #
        out_tpeak, out_fwhm, out_ampl = popt[0],popt[1],popt[2]
        #
        if (out_ampl < localT) or (out_fwhm > 0.1):
            print('something went wrong on this event:')
            if out_ampl < localT:
                print('out_ampl ',str(out_ampl),' < localT = ',localT)
            if out_fwhm > 0.1:
                print('out_fwhm',str(out_fwhm),' > 0.1')
                #print('out_fpeak',out_ampl,'out_fwhm',out_fwhm)
            return np.full_like(time_in_window,np.nan), np.full_like(flux_in_window,np.nan),np.nan,np.nan,np.nan
        else:
            if verbose==True:
                print('model flare len check',len(time_in_window),len(model))               
            return flare_times, model, out_tpeak, out_fwhm, out_ampl
#         except ValueError as VE:            
#                 print(VE)
#                 return np.full_like(time_in_window,np.nan),\
#             np.full_like(flux_in_window,np.nan),\
#             np.nan,np.nan,np.nan


def smoother(window_size_in_hrs,time,flux,flux_error=None,filter_type='medfilt'): 
    '''
    The algorithm for smoothing light curves.
    Parameters:
    ----------
    window_size_in_hrs :float
        The window size in hours used to smooth the data with the selected filter.
    time : numpy array
        time stamps of light curve.
    flux : numpy array
        flux values of light curve.
    flux_error : numpy array, optional
        flux errors of light curve. If array is not provided, one will be approximated.
    filter_type : str, optional
        Choice of filter used in smoothing. For a list of options, see https://github.com/hippke/wotan/
    Return:
    ------------
    lc : pandas dataframe
        contains columns of "time", "flux" and "flux_err" of input time, flux, flux_error arrays
    newlc: pandas datafrmae
        contains columns of "time", "flux" and "flux_err" of smoothed time, flux, flux_error arrays
    trend_lc: numpy array
        contains trend line of smoothed flux values.
    '''
    from wotan import flatten #<--- DEPENDENCY
    import time as clock # used for timing smoothing function
    start=clock.time() 
    window_size = window_size_in_hrs/24.0 # Convert the length of the filter window to units of time (days)
    if flux_error==None:
        flux_error = np.full_like(flux,np.nanstd(flux))
    #print('len check: ',len(time),len(flux),len(flux_error))
    if filter_type=='biweight':
        flatten_lc, trend_lc = flatten(time, flux, window_length=window_size, return_trend=True, \
                                       method=filter_type,robust=True)
    if filter_type=='medfilt':
        # this filter type needs a window size in number of data points
        # instead of a window size in unit of time
        # Below converts input window size to number of data points:
        cadence = np.nanmedian(np.diff(time))
        def round_up_to_odd(f):
            return int(np.ceil(f) // 2 * 2 + 1)
        Npts = round_up_to_odd(int((window_size_in_hrs/24)/cadence))
        # now smooth with medfilt filter:
        flatten_lc, trend_lc = flatten(time, flux, window_length=Npts, method=filter_type, \
                                       return_trend=True,robust=True)
    print('filter type: ',filter_type)
    # rename time, smoothed flux and flux error arrays
    T=time
    F=flatten_lc
    FE=flux_error
    #forcing same size outputs
    m=np.where(np.in1d(T,time)==True)[0]
    t2=time[m]
    f2=flux[m]
    fe2=flux_error[m]
    # packaging results in pandas dataframes for output 
    lc=pd.DataFrame({'time':t2,'flux':f2,'flux_err':fe2})
    newlc=pd.DataFrame({'time':T,'flux':F,'flux_err':FE})
    print('len check after detrending: New ',len(np.array(newlc['time'])),len(np.array(newlc['flux'])),' input ',len(np.array(lc['time'])),len(np.array(lc['flux'])))
    end=clock.time()
    runtime=end-start
    if runtime>60:
        print('smoothing runtime: ',np.round(runtime/60,2),' minutes')
    if runtime<60:
        print('smoothing runtime: ',np.round(runtime,2),' seconds')

    return lc, newlc, trend_lc #time, raw ,detrended, error, trend



def separate_flares_OLD(s,separation_ind_fl, ttemp, ftemp, etemp):
    #
    if s<1:
        window = np.where(ttemp < ttemp[separation_ind_fl][s])[0]
    else:
        window = np.where( (ttemp > ttemp[separation_ind_fl][s-1]) & (ttemp < ttemp[separation_ind_fl][s] ) )[0]
    time_in_window = ttemp[window] 
    flux_in_window = ftemp[window] 
    error_in_window = etemp[window]
    if len(window)==0:
        print('empty window in separate flares ftn')
    #sort indices so that time stamps are ALWAYS increasing...
    inorder_mask = np.argsort(time_in_window)
    time_in_window = time_in_window[inorder_mask]
    flux_in_window = flux_in_window[inorder_mask]
    error_in_window = error_in_window[inorder_mask]
    #drop any duplicates created???
    dupe_mask = np.unique(time_in_window,return_index=True)[1]
    if len(dupe_mask)==0:
        print('empty "dupe_mask" in separate flares ftn')    
    time_in_window = time_in_window[dupe_mask]
    flux_in_window = flux_in_window[dupe_mask]
    error_in_window = error_in_window[dupe_mask] 
    # what if the window is empty?
    
    return time_in_window, flux_in_window, error_in_window

def separate_flares(s,separation_ind_fl, ind_fl, time,flux,error,N_before_pts):
    ttemp = time[ind_fl]
    ftemp = flux[ind_fl]
    etemp = error[ind_fl]    
    #
    # CHANGE THIS PART
    if s<1:
        window = np.where(ttemp < ttemp[separation_ind_fl][s])[0]
    else:
        window = np.where( (ttemp > ttemp[separation_ind_fl][s-1]) & (ttemp < ttemp[separation_ind_fl][s] ) )[0]
    cadence = np.nanmedian(np.diff(ttemp))
#     window2 = np.where((time >= np.min(ttemp[window])-1*N_before_pts*cadence) &\
#                        (time <= np.max(ttemp[window])+3*N_before_pts*cadence ) )[0] #OLD
    if len(window)!=0:
        
#         window2 = np.where((time >= np.min(ttemp[window])-1*N_before_pts*cadence) &\
#                            (time <= np.max(ttemp[window])+2.5*N_before_pts*cadence )&\
#                           (np.abs(np.nanmax(ftemp[window]/(ftemp[window[-1]])) > 2 )))[0]    
        window2 = np.where((time >= np.min(ttemp[window])-1*N_before_pts*cadence) &\
                   (time <= np.max(ttemp[window])+2.5*N_before_pts*cadence )&\
                  (np.abs(np.nanmax(ftemp[window]/(ftemp[np.argmax(ttemp)])) > 2 )))[0]
    else:
        window2=window
    
    #consider defining last data point where np.nanmax(ftemp[window]) / ftemp[window][-1] > 2
    # maybe need to use a while loop?
    #
    # CHANGE THIS PART        
#     time_in_window = ttemp[window] 
#     flux_in_window = ftemp[window] 
#     error_in_window = etemp[window]
    time_in_window = time[window2]
    flux_in_window = flux[window2]
    error_in_window = error[window2]
    #
    if len(window)==0:
        print('empty window in separate flares ftn')
    #sort indices so that time stamps are ALWAYS increasing...
    inorder_mask = np.argsort(time_in_window)
    time_in_window = time_in_window[inorder_mask]
    flux_in_window = flux_in_window[inorder_mask]
    error_in_window = error_in_window[inorder_mask]
    #drop any duplicates created???
    dupe_mask = np.unique(time_in_window,return_index=True)[1]
    if len(dupe_mask)==0:
        print('empty "dupe_mask" in separate flares ftn')    
    time_in_window = time_in_window[dupe_mask]
    flux_in_window = flux_in_window[dupe_mask]
    error_in_window = error_in_window[dupe_mask] 
    # what if the window is empty?
    
    return time_in_window, flux_in_window, error_in_window

def make_initial_flare_guess(time_in_window,flux_in_window):
    #make initial flare parameter guesses
    cadence=np.nanmedian(np.diff(time_in_window))
    f_half0 = np.where((flux_in_window/np.nanmedian(flux_in_window) >= np.nanmax(flux_in_window/np.nanmedian(flux_in_window))*0.5))
    try:    
        fwhm0 = np.max(time_in_window[f_half0]) - np.min(time_in_window[f_half0])   
    except ValueError:
        print('Error definining FWHM. min twindow:',np.min(time_in_window),'; max twindow:',np.max(time_in_window))
        print('f_half0:',f_half0)
        print('fmax:',np.nanmax(flux_in_window))
    tpeak0 = time_in_window[np.argmax(flux_in_window)]
    fpeak0 = np.nanmax(flux_in_window)
    
    return cadence,f_half0, fwhm0,tpeak0,fpeak0

def get_noise_stats(input_flux,flux_in_window,threshold_type,N1):
    from scipy import stats
    if threshold_type=='local':  
        mad=stats.median_abs_deviation(flux_in_window) 
        median=np.nanmedian(flux_in_window)
        sigma=np.nanstd(flux_in_window)
        localT = N1*mad+median
    if threshold_type=='global':  
        mad=stats.median_abs_deviation(input_flux) 
        median=np.nanmedian(input_flux)
        sigma=np.nanstd(input_flux)                
        localT = N1*mad+median                
    ###
    return mad, median, sigma, localT

# def calc_ED(flux,times):
#         '''
#         Compute the Equivalent Duration of a fake flare.
#         This is the area under the flare, in relative flux units.
#         Parameters:
#         -------------
#         time : numpy array
#             units of DAYS
#         flux : numpy array
#             relative flux units
#         Return:
#         ------------
#         p : float
#             equivalent duration of a single event in units of seconds
#         '''
#         x = times  #in units of days
#         #ED = np.sum(flux[:-1] * np.diff(x) )
#         ED = np.abs(np.trapz(flux, x)) #* np.abs(np.nanmax(flux)) #<--- not sure why we're including this...
#         return ED
    
def calc_ED(flux, time, median):#, flux_error):
    '''
    Compute the Equivalent Duration of an event. This is simply the area
    under the flare, in relative flux units.
    NOTE: sums up all the data given, no start/stop times input. Only
        pass the flare!
    Flux must be array in units of zero-centered RELATIVE FLUX
    Time must be array in units of DAYS
    Fluxerror must be array in units RELATIVE FLUX
    Output has units of DAYS
    
    note: error comes from the Spenser's code
    '''
    dtime = np.diff(time)
    y = flux+median
    ED = np.abs(np.trapz(y, x=(time))) #x=(time * 60.0 * 60.0 * 24.0))
    #ED_err = np.sqrt(np.sum((dtime*flux_error[:-1])**2))

    return ED #, ED_err    

def validation_check_1(time_in_window,flux_in_window,flare, flare_times, localT, N3, input_flux_median, verbose):
    #
    ## Validation check one: consecutive flare points above local noise threshold        
    #
    ## counting number of modeled flare data points
    flarepts = len(flare)
    #calc residual
    try:
        if len(flux_in_window)>len(flare):        
            resid_mask = np.where((flare_times >= np.nanmin(time_in_window))&(flare_times <= np.nanmax(time_in_window)))
            residuals = ((flux_in_window[resid_mask] - flare) - np.nanmean((flux_in_window[resid_mask] - flare)))/np.nanstd((flux_in_window[resid_mask] - flare))
            #calculate equivalent duration (integrate under flare curve)
            inorder_mask = np.argsort(flare_times)
            #ED = np.abs(np.trapz(residuals[inorder_mask], flare_times[inorder_mask])) # in days
            ED_old = np.abs(np.sum(np.diff(flare_times[inorder_mask]) * residuals[:-1]))
            #
#             ED = np.trapz(residuals, flare_times)*24*3600 * np.abs(np.nanmax(residuals))
            #ED = calc_ED(residuals,flare_times)
            ED = calc_ED(flux_in_window[resid_mask], flare_times, input_flux_median)
            if ED<0:
                ED = np.abs(ED)
            # calculate duration
            dur = np.max(flare_times) - np.min(flare_times) # in days
            #
        else:
            resid_mask = np.where((time_in_window >= np.nanmin(flare_times))&(time_in_window <= np.nanmax(flare_times)))
            residuals = ((flux_in_window - flare[resid_mask]) - np.nanmean((flux_in_window - flare[resid_mask])))/np.nanstd((flux_in_window - flare[resid_mask]))    
            #calculate equivalent duration (integrate under flare curve)
            inorder_mask = np.argsort(time_in_window)
            #ED = np.abs(np.trapz(residuals[inorder_mask], time_in_window[inorder_mask])) # in days
            #ED = np.abs(np.sum(np.diff(time_in_window[inorder_mask]) * residuals[:-1]))
            #ED = calc_ED(residuals[inorder_mask][:-1],time_in_window[inorder_mask])
            ED = calc_ED(flux_in_window[inorder_mask][:-1], time_in_window[inorder_mask], input_flux_median)
            # calculate duration
            dur = np.nanmax(time_in_window) - np.nanmin(time_in_window) # in days            
            #
            #
    except ValueError as e:
        residuals = ((flux_in_window - flare) - np.nanmean((flux_in_window - flare)))/np.nanstd((flux_in_window - flare))
        #ED = np.abs(np.trapz(residuals, time_in_window)) # in days
        #ED = np.abs(np.sum(np.diff(time_in_window) * residuals[:-1]))
        
        #ED = calc_ED(residuals,time_in_window)
        try:
            ED = calc_ED(flux_in_window,time_in_window, input_flux_median)
        except ValueError as e2:
            print(e2,'on ED calc',len(residuals),len(time_in_window))
            ED = calc_ED(flux_in_window[:-1],time_in_window, input_flux_median)
        
        # calculate duration
        dur = np.nanmax(time_in_window) - np.nanmin(time_in_window) # in days            
    #
    #
    # 
    consec = consecutive(np.where(flare>localT)[0])
    n_consec = len(consec)
    consec_real = consecutive(np.where(flux_in_window>localT)[0])
    n_consec_real = len(consec_real)  
    #
    ##
    #
    if (n_consec >= N3) & (n_consec_real >= N3) :      
        result1=True
    else:
        if verbose==True:
            print(' ')
            print('failed check 1')
            if (n_consec <= N3):
                print('n_consec <= N3')
            if (n_consec_real <= N3):      
                print('(n_consec_real <= N3)')
        result1=False
    return result1, flarepts, ED, dur, consec, n_consec, consec_real, n_consec_real


def validation_check_2(time_in_window, flux_in_window, out_tpeak, out_fwhm, out_fpeak, flare_times,verbose,t_char_min,t_char_max):
    # Validation check two: consecutive flare points above local noise threshold: 
    # flux in decay is higher than flux in rise at +/- 1 FWHM
    # and
    # there is data between tpeak +/- FWHM + offset
    #    
    mint = np.min(time_in_window)   
    #
    #is 2 hours too much?! maybe this should be a multiple of flare cadences
    # dt=2/24 #add some extra time before/after to data in window 
    # dt = flare_sep_in_cadences*3 #WHAT IS THIS USED FOR????!?!
    #is 2 hours too much?! maybe this should be a multiple of flare cadences
    #
    # since this initially guessed flare model passes criteria one,
    # refit it using the initial guess parameters for the data in the window
    #
    t = np.linspace(-t_char_min*out_fwhm+out_tpeak,t_char_max*out_fwhm+out_tpeak,len(time_in_window)) 
    t = np.linspace(np.min(time_in_window),np.max(time_in_window),len(time_in_window)) 
    #
    out_popt = [out_tpeak,out_fwhm,out_fpeak]
    #
    if np.isnan(out_fpeak)==False:
        f =  aflare1(t, *out_popt)
        #
        #compare rise/decay in units of FWHM with the synthetic flare model
        # this is done using the closest data point to what should be tpeak +/- 1 FWHM
        trise_at_minus_1_FHWM = t[find_nearest(characterize_timescale(t, out_tpeak, out_fwhm),-1)]
        frise_at_minus_1_FHWM = f[find_nearest(characterize_timescale(t, out_tpeak, out_fwhm),-1)]
        tdecay_at_plus_1_FHWM = t[find_nearest(characterize_timescale(t, out_tpeak, out_fwhm),1)]
        fdecay_at_plus_1_FHWM = f[find_nearest(characterize_timescale(t, out_tpeak, out_fwhm),1)]
    else:
        f = np.zeros_like(t)*1
        #
        trise_at_minus_1_FHWM = np.nan
        frise_at_minus_1_FHWM = np.nan
        tdecay_at_plus_1_FHWM = np.nan
        fdecay_at_plus_1_FHWM = np.nan 
    #
    ###
    ###
    #
    fwhm_offset = 0.6 # add some wiggle room
    mask_between_0_and_1_FWHM = np.where((characterize_timescale(flare_times,\
                                                                 out_tpeak,out_fwhm)>=0) &\
                                         (characterize_timescale(flare_times,\
                                                                 out_tpeak,out_fwhm)<=0+1+fwhm_offset) )[0]
    mask_between_minus_1_and_0_FWHM = np.where((characterize_timescale(flare_times,\
                                                                 out_tpeak,out_fwhm)>=0-1-fwhm_offset) &\
                                         (characterize_timescale(flare_times,\
                                                                 out_tpeak,out_fwhm)<=0) )[0]        
    if (fdecay_at_plus_1_FHWM > frise_at_minus_1_FHWM) & (len(mask_between_0_and_1_FWHM)>0) or (len(mask_between_minus_1_and_0_FWHM)>0):    
        #
        # THIS EXCLUDES SHALLOW FLARES, TRY Fpeak > Flast INSTEAD?
        #
        #
        # Validation check 2a: check that Fpeak > Flast flare point:
        if (out_fpeak > flux_in_window[np.argmax(time_in_window)]):             
            result2 = True
        else:
            if verbose==True:
                print('failed check 2a') 
                if (out_fpeak <= flux_in_window[np.argmax(time_in_window)]): 
                    print('(out_fpeak <= flux_in_window[np.argmax(time_in_window)])')            
            result2 = False
        #        
        # Validation check 2a: check that Fpeak / Flast flare point >=2:
#         if (out_fpeak / flux_in_window[np.argmax(time_in_window)]) >=2:             
#             result2 = True
#         else:
#             print('failed check 2a') 
#             if (out_fpeak / flux_in_window[np.argmax(time_in_window)]) <2: 
#                 print('(out_fpeak / flux_in_window[np.argmax(time_in_window)]) <2')            
#             result2 = False
        #
    else:
        if verbose==True:
            print('failed check 2b')        
            if (fdecay_at_plus_1_FHWM < frise_at_minus_1_FHWM):
                print('fdecay_at_plus_1_FHWM < frise_at_minus_1_FHWM')
            if (len(mask_between_0_and_1_FWHM)<1):
                print('len(mask_between_0_and_1_FWHM)<1')            
            if (len(mask_between_minus_1_and_0_FWHM)<1):
                print('len(mask_between_minus_1_and_0_FWHM)<1')
        result2 = False
    return result2, mint, t, f, trise_at_minus_1_FHWM, frise_at_minus_1_FHWM, tdecay_at_plus_1_FHWM, fdecay_at_plus_1_FHWM, mask_between_0_and_1_FWHM, mask_between_minus_1_and_0_FWHM

def validation_check_2_ALT(time_in_window, flux_in_window, out_tpeak, out_fwhm, out_fpeak, flare_times,verbose,t_char_min,t_char_max):
    # Validation check two: consecutive flare points above local noise threshold: 
    # flux in decay is higher than flux in rise at +/- 1 FWHM
    # and
    # there is data between tpeak +/- FWHM + offset
    #    
    mint = np.min(time_in_window)   
    #
    #is 2 hours too much?! maybe this should be a multiple of flare cadences
    # dt=2/24 #add some extra time before/after to data in window 
    # dt = flare_sep_in_cadences*3 #WHAT IS THIS USED FOR????!?!
    #is 2 hours too much?! maybe this should be a multiple of flare cadences
    #
    # since this initially guessed flare model passes criteria one,
    # refit it using the initial guess parameters for the data in the window
    #
    t = np.linspace(-t_char_min*out_fwhm+out_tpeak,t_char_max*out_fwhm+out_tpeak,len(time_in_window)) 
    t = np.linspace(np.min(time_in_window),np.max(time_in_window),len(time_in_window)) 
    #
    out_popt = [out_tpeak,out_fwhm,out_fpeak]
    #
    if np.isnan(out_fpeak)==False:
        f =  aflare1(t, *out_popt)
        #
        #compare rise/decay in units of FWHM with the synthetic flare model
        # this is done using the closest data point to what should be tpeak +/- 1 FWHM
        # TRY -2/+1
        N_FWHM=[2,1]
        trise_at_minus_1_FHWM = t[find_nearest(characterize_timescale(t, out_tpeak, out_fwhm),-N_FWHM[0])]
        frise_at_minus_1_FHWM = f[find_nearest(characterize_timescale(t, out_tpeak, out_fwhm),-N_FWHM[0])]
        tdecay_at_plus_1_FHWM = t[find_nearest(characterize_timescale(t, out_tpeak, out_fwhm),N_FWHM[1])]
        fdecay_at_plus_1_FHWM = f[find_nearest(characterize_timescale(t, out_tpeak, out_fwhm),N_FWHM[1])]
    else:
        f = np.zeros_like(t)*1
        #
        trise_at_minus_1_FHWM = np.nan
        frise_at_minus_1_FHWM = np.nan
        tdecay_at_plus_1_FHWM = np.nan
        fdecay_at_plus_1_FHWM = np.nan 
    #
    ###
    ###
    #
    fwhm_offset = 0.6 # add some wiggle room
    mask_between_0_and_1_FWHM = np.where((characterize_timescale(flare_times,\
                                                                 out_tpeak,out_fwhm)>=0) &\
                                         (characterize_timescale(flare_times,\
                                                                 out_tpeak,out_fwhm)<=0+N_FWHM[1]+fwhm_offset) )[0]
    mask_between_minus_1_and_0_FWHM = np.where((characterize_timescale(flare_times,\
                                                                 out_tpeak,out_fwhm)>=0-N_FWHM[0]-fwhm_offset) &\
                                         (characterize_timescale(flare_times,\
                                                                 out_tpeak,out_fwhm)<=0) )[0]        
    if (fdecay_at_plus_1_FHWM > frise_at_minus_1_FHWM) & (len(mask_between_0_and_1_FWHM)>0) or (len(mask_between_minus_1_and_0_FWHM)>0):    
        #
        # THIS EXCLUDES SHALLOW FLARES, TRY Fpeak > Flast INSTEAD?
        #
        #
        # Validation check 2a: check that Fpeak > Flast flare point:
        if (out_fpeak > flux_in_window[np.argmax(time_in_window)]):             
            result2 = True
        else:
            if verbose==True:
                print('failed check 2a') 
                if (out_fpeak <= flux_in_window[np.argmax(time_in_window)]): 
                    print('(out_fpeak <= flux_in_window[np.argmax(time_in_window)])')            
            result2 = False
        #        
        # Validation check 2a: check that Fpeak / Flast flare point >=2:
#         if (out_fpeak / flux_in_window[np.argmax(time_in_window)]) >=2:             
#             result2 = True
#         else:
#             print('failed check 2a') 
#             if (out_fpeak / flux_in_window[np.argmax(time_in_window)]) <2: 
#                 print('(out_fpeak / flux_in_window[np.argmax(time_in_window)]) <2')            
#             result2 = False
        #
    else:
        if verbose==True:
            print('failed check 2b')        
            if (fdecay_at_plus_1_FHWM < frise_at_minus_1_FHWM):
                print('fdecay_at_plus_1_FHWM < frise_at_minus_1_FHWM')
            if (len(mask_between_0_and_1_FWHM)<1):
                print('len(mask_between_0_and_1_FWHM)<1')            
            if (len(mask_between_minus_1_and_0_FWHM)<1):
                print('len(mask_between_minus_1_and_0_FWHM)<1')
        result2 = False
    return result2, mint, t, f, trise_at_minus_1_FHWM, frise_at_minus_1_FHWM, tdecay_at_plus_1_FHWM, fdecay_at_plus_1_FHWM, mask_between_0_and_1_FWHM, mask_between_minus_1_and_0_FWHM, N_FWHM


def validation_check_3(trise_at_minus_1_FHWM, tdecay_at_plus_1_FHWM, out_tpeak, out_fwhm,verbose):
    # Validation check three: make sure tpeak +/- FWHM is ACTUALLy near +/-1
    # our method grabs the CLOSEST datapoint to +/- 1 but this allows some
    # strange things into consideration when validating. Check to see that
    # there are at least more than 1 point within +/- fwhm_offset of +/- 1 
    # (EITHER OR is fine). Also check that trise_at_minus_1_FHWM =/= tdecay_at_plus_1_FHWM.
    # signs should be opposite!
    #
    fwhm_round  = 1
    fwhm_offset = 0.6
    #   
    t_m1_fwhm = np.round(characterize_timescale(trise_at_minus_1_FHWM,out_tpeak, out_fwhm),fwhm_round)
    t_p1_fwhm = np.round(characterize_timescale(tdecay_at_plus_1_FHWM,out_tpeak, out_fwhm),fwhm_round)
#     print('grey FWHM pt check:')
#     print('rise',characterize_timescale(trise_at_minus_1_FHWM,out_tpeak, out_fwhm),\
#           '-1, rounded:',t_m1_fwhm)
#     print('decay',characterize_timescale(tdecay_at_plus_1_FHWM,out_tpeak, out_fwhm),\
#           '+1, rounded:',t_p1_fwhm)
#     print(' ')
    if (t_m1_fwhm >= 0-1-fwhm_offset) or (t_p1_fwhm <= 0+1+fwhm_offset) :
        if (np.sign(t_m1_fwhm) == np.sign(t_p1_fwhm)):
            if verbose==True:
                print('(np.sign(t_m1_fwhm) == np.sign(t_p1_fwhm)):',np.sign(t_m1_fwhm),np.sign(t_p1_fwhm))
            result3=False
        else:
            result3=True
    else:
        if verbose==True:
            print('failed check 3')  
            if (np.sign(t_m1_fwhm) == np.sign(t_p1_fwhm)):
                print('(np.sign(t_m1_fwhm) >= np.sign(t_p1_fwhm))')
            if (t_m1_fwhm < 0-1-fwhm_offset):
                print('(t_m1_fwhm < 0-1-fwhm_offset)')
            if (t_p1_fwhm > 0+1+fwhm_offset):
                print('(t_p1_fwhm > 0+1+fwhm_offset)')
        result3=False
    #
    return result3

def legend_without_duplicate_labels(ax,loc='upper left',ncol=1,framealpha=0.5):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique),loc='upper left',ncol=1,framealpha=0.5)


def plot_validation_checks(pdf,string,time,flux,time_in_window,flux_in_window,\
                           t,f,flare_times,flare,ttemp,ftemp,\
                           tstart,out_tpeak,out_fpeak, out_fwhm,\
                           localT,sigma,\
                           trise_at_minus_1_FHWM,frise_at_minus_1_FHWM,\
                           tdecay_at_plus_1_FHWM,fdecay_at_plus_1_FHWM,\
                           N1, N2, N3, mad, median, \
                           show_figs, inject_lc,t_char_min,t_char_max,ED,dur):
    fmin,fmax = np.nanmin(flux_in_window),np.nanmax(flux_in_window)
    ymin,ymax = fmin-3*sigma, fmax+3*sigma
    #
    flarepts = len(flare)
    #
    #
    fig=plt.figure(figsize=(10, 5))
    fig.suptitle(string,x=0.5,y=1.02)
    ax=fig.add_subplot(111)

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    ax2 = divider.append_axes("bottom", size='30%', pad=0.5)
    ax.figure.add_axes(ax2)
    
    
    # data - model
    try:
        if len(flux_in_window)>len(flare):        
            resid_mask = np.where((flare_times >= np.nanmin(time_in_window))&(flare_times <= np.nanmax(time_in_window)))
            residuals = ((flux_in_window[resid_mask] - flare) - np.nanmean((flux_in_window[resid_mask] - flare)))/np.nanstd((flux_in_window[resid_mask] - flare))
        else:
            resid_mask = np.where((time_in_window >= np.nanmin(flare_times))&(time_in_window <= np.nanmax(flare_times)))
            residuals = ((flux_in_window - flare[resid_mask]) - np.nanmean((flux_in_window - flare[resid_mask])))/np.nanstd((flux_in_window - flare[resid_mask]))    
    except ValueError as e:
        print(e,'on calculating residuals')
        print('mask',resid_mask)
        print('len check (window flux, flare, resid mask):',len(flux_in_window),len(flare), len(resid_mask))
        residuals = ((flux_in_window - flare) - np.nanmean((flux_in_window - flare)))/np.nanstd((flux_in_window - flare))
        
    if np.isnan(out_fwhm)==False:    
        #
        try:
            ax2.plot(characterize_timescale(flare_times, out_tpeak, out_fwhm), residuals, 'k.')
        except ValueError as e:
            print(e,' on plotting residuals')
            print('mask',resid_mask)
            print('len check (window flux, flare, resid mask):',len(flux_in_window),len(flare), len(resid_mask))
            residuals = ((flux_in_window - flare) - np.nanmean((flux_in_window - flare)))/np.nanstd((flux_in_window - flare))
            ax2.plot(characterize_timescale(flare_times, out_tpeak, out_fwhm), residuals, 'k.')
        ax2.axhline(y=0,color='red')
        try:
            ax.plot(characterize_timescale(ttemp, out_tpeak, out_fwhm),\
                    ftemp,'go',label='flagged by FINDFLARE')
        except ValueError as e:
            print(e,'on plotting FINDFLARE points')
            print('fitted flare params [tpeak,fwhm,ampl]:',out_tpeak, out_fwhm,out_fpeak)
            print('len check:',len(ttemp),len(ftemp))
            ax.plot(characterize_timescale(ttemp, out_tpeak, out_fwhm),\
                    ftemp,'go',label='flagged by FINDFLARE')
            #
        ax.plot(characterize_timescale(time,out_tpeak,out_fwhm),\
                flux,marker='.',color='grey',linestyle='none',label='Non-Flagged Data',ms=3,zorder=-10)
        ax.plot(characterize_timescale(time_in_window,out_tpeak,out_fwhm),\
                flux_in_window,'k.',label='Data',ms=3)
        ax.plot(characterize_timescale(flare_times, out_tpeak, out_fwhm),\
                flare,'kx-',ms=10,lw=3,label='Initial Flare Model')
        ax.plot(characterize_timescale(t, out_tpeak, out_fwhm), f,'r-',lw=2,label='Interpolated Flare Model',ms=3)
        ax.axhline(y=localT, color='cyan',\
                   linestyle='--',zorder=-10,\
                   label=str(N1)+'x MAD + median')
        ax.axhline(y=median, color='green',\
                   linestyle='--',zorder=-10,label='median')
    #     ax.axhline(y=float(N1*mad), color='grey',\
    #                linestyle='--',zorder=-10,label=str(N1)+'x MAD')
    #     ax.axhline(y=float(0.5*np.max(flux_in_window)), color='red',\
    #                linestyle='--',zorder=-10,label='Half Max')                    
        #
        ax.plot(characterize_timescale(trise_at_minus_1_FHWM, \
                                       out_tpeak, out_fwhm),\
                frise_at_minus_1_FHWM,\
                color='grey',marker='o',\
                label='flux at FWHM',markersize=10,zorder=100)
        #
        ax.plot(characterize_timescale(tdecay_at_plus_1_FHWM, \
                                       out_tpeak, out_fwhm),\
                fdecay_at_plus_1_FHWM,\
                color='grey',marker='o',\
                markersize=10,zorder=100)
        
    if np.isnan(out_fwhm)==True:           
        print('out_fwhm is NaN!')
        print('fitted flare params [tpeak,fwhm,ampl]:',out_tpeak, out_fwhm,out_fpeak)
        ax.plot(ttemp,ftemp,'go',label='flagged by FINDFLARE')   
        ax.plot(time, flux,marker='.',color='grey',linestyle='none',label='Non-Flagged Data',ms=3,zorder=-10)        
        ax.plot(time_in_window,flux_in_window,'k.',label='Data',ms=3)
        ax.plot(flare_times, flare,'kx-',ms=15,lw=3,label='Initial Flare Model')
        ax.plot(t, f,'r-',lw=2,label='Interpolated Flare Model',ms=3)
        ax.axhline(y=localT, color='cyan',\
                   linestyle='--',zorder=-10,\
                   label=str(N1)+'x MAD + median')
        ax.axhline(y=median, color='green',\
                   linestyle='--',zorder=-10,label='median')        
        if inject_lc is not None:
            for ii in range(len(inject_lc)):
                # should we just plot them everywhere, regardless of window?
                if (inject_lc['tpeak'][ii]>np.min(time_in_window)) & (inject_lc['tpeak'][ii]<np.max(time_in_window)):
                    ax.axvline(x=inject_lc['tstart'][ii], \
                               color='black',linestyle='--',\
                               lw=2,label='inject start time')
                    ax.axvline(x=inject_lc['tpeak'][ii],\
                               color='orange',linestyle='--',\
                               lw=2,label='inject peak time')
                else:
                    ax.axvline(x=inject_lc['tstart'][ii],
                               color='black',linestyle='--',\
                               lw=2)
                    ax.axvline(x=inject_lc['tpeak'][ii],\
                               color='orange',linestyle='--',\
                               lw=2)        
        
    #
    #number of digits for reporting values
    ndigi =4
    xpos0=0.025*16
    left = xpos0-xpos0/3
    right = xpos0+xpos0/2
    ypos0=0.975
    string = 'fpeak = '+str(np.round(out_fpeak,ndigi))
    ax.text(left, ypos0, string, weight='bold',horizontalalignment='left',\
            verticalalignment='center', transform=ax.transAxes, wrap=True)    
    string = 'tstart = '+str(np.round(tstart,ndigi))
    ax.text(left, ypos0-0.05*1, string, weight='bold',horizontalalignment='left',\
            verticalalignment='center', transform=ax.transAxes, wrap=True)
    string = 'tpeak = '+str(np.round(out_tpeak,ndigi))
    ax.text(left, ypos0-0.05*2, string, weight='bold',horizontalalignment='left',\
            verticalalignment='center', transform=ax.transAxes, wrap=True)
    string = 'tend = '+str(np.round(tstart+dur,ndigi))
    ax.text(left, ypos0-0.05*3, string, weight='bold',horizontalalignment='left',\
            verticalalignment='center', transform=ax.transAxes, wrap=True)        
    string = 'Nflarepts = '+str(np.round(flarepts,ndigi))
    ax.text(left, ypos0-0.05*4, string, weight='bold',horizontalalignment='left',\
            verticalalignment='center', transform=ax.transAxes, wrap=True)
    string = 'Equiv Duration = '+str(np.round(ED*60*24,ndigi))+' min'
    ax.text(right, ypos0, string, weight='bold',horizontalalignment='left',\
            verticalalignment='center', transform=ax.transAxes, wrap=True)
    string = 'Duration = '+str(np.round(dur*60*24,ndigi))+' min'
    ax.text(right, ypos0-0.05*1, string, weight='bold',horizontalalignment='left',\
            verticalalignment='center', transform=ax.transAxes, wrap=True)    
    string = 'Flux at -1 FWHM = '+str(np.round(frise_at_minus_1_FHWM,ndigi))
    ax.text(right, ypos0-0.05*2, string, weight='bold',horizontalalignment='left',\
            verticalalignment='center', transform=ax.transAxes, wrap=True)                
    string = 'Flux at +1 FWHM = '+str(np.round(fdecay_at_plus_1_FHWM,ndigi))
    ax.text(right, ypos0-0.05*3, string, weight='bold',horizontalalignment='left',\
            verticalalignment='center', transform=ax.transAxes, wrap=True)
    ax.axvline(x=-1,linestyle='--',color='grey')
    ax.axvline(x=1,linestyle='--',color='grey')    
    ###
    ###
    if inject_lc is not None:
        for ii in range(len(inject_lc)):
            # should we just plot them everywhere, regardless of window?
            if (inject_lc['tpeak'][ii]>np.min(time_in_window)) & (inject_lc['tpeak'][ii]<np.max(time_in_window)):
                ax.axvline(x=characterize_timescale(inject_lc['tstart'][ii], \
                                                    out_tpeak, out_fwhm), \
                           color='black',linestyle='--',\
                           lw=2,label='inject start time')
                ax.axvline(x=characterize_timescale(inject_lc['tpeak'][ii],\
                                                    out_tpeak, out_fwhm),\
                           color='orange',linestyle='--',\
                           lw=2,label='inject peak time')
                ax.axvline(x=characterize_timescale(inject_lc['tstart'][ii]+inject_lc['duration'][ii],\
                                                    out_tpeak, out_fwhm),\
                           color='black',linestyle='--',\
                           lw=2,label='inject end time')                
                string = 'Injected tstart = '+str(np.round(inject_lc['tstart'][ii],ndigi))
                ax.text(xpos0+xpos0/2, ypos0-0.05*4, string, weight='bold',horizontalalignment='left',\
                        verticalalignment='center', transform=ax.transAxes, wrap=True) 
                string = 'Injected tpeak = '+str(np.round(inject_lc['tpeak'][ii],ndigi))
                ax.text(xpos0+xpos0/2, ypos0-0.05*5, string, weight='bold',horizontalalignment='left',\
                        verticalalignment='center', transform=ax.transAxes, wrap=True)                 
                string = 'Injected tend = '+str(np.round(inject_lc['tstart'][ii]+inject_lc['duration'][ii],ndigi))
                ax.text(xpos0+xpos0/2, ypos0-0.05*6, string, weight='bold',horizontalalignment='left',\
                        verticalalignment='center', transform=ax.transAxes, wrap=True)                                 
                string = 'Injected Fpeak = '+str(np.round(inject_lc['fpeak'][ii],ndigi))
                ax.text(xpos0+xpos0/2, ypos0-0.05*7, string, weight='bold',horizontalalignment='left',\
                        verticalalignment='center', transform=ax.transAxes, wrap=True)
                string = 'Injected Duration = '+str(np.round(inject_lc['duration'][ii]*24*60,ndigi))+' min'
                ax.text(xpos0+xpos0/2, ypos0-0.05*8, string, weight='bold',horizontalalignment='left',\
                        verticalalignment='center', transform=ax.transAxes, wrap=True)                
                string = 'Injected Nflarepts = '+str(np.round(inject_lc['Npts'][ii],ndigi))
                ax.text(xpos0+xpos0/2, ypos0-0.05*9, string, weight='bold',horizontalalignment='left',\
                        verticalalignment='center', transform=ax.transAxes, wrap=True)                  
                        
            else:
                ax.axvline(x=characterize_timescale(inject_lc['tstart'][ii], \
                                                    out_tpeak, out_fwhm), \
                           color='black',linestyle='--',\
                           lw=2)
                ax.axvline(x=characterize_timescale(inject_lc['tpeak'][ii],\
                                                    out_tpeak, out_fwhm),\
                           color='orange',linestyle='--',\
                           lw=2)
    ###

    ymin,ymax = fmin-3*sigma, fmax+3*sigma
    ymax = fmax+3*sigma + (fmax+3*sigma)/10    
    if ymax < np.nanmax(f):
        ymax = out_fpeak+3*sigma + (out_fpeak+3*sigma)/10    
    ax.set_ylim(ymin,ymax)   
    # below not used
    ###try:
    ###    ymin2,ymax2 = np.min(residuals)-3*np.nanstd(residuals), np.max(residuals)+3*np.nanstd(residuals)
    ###except ValueError:
    ###    residuals = ((flux_in_window - flare) - np.nanmean((flux_in_window - flare)))/np.nanstd((flux_in_window - flare))
    ###    ymin2,ymax2 = np.min(residuals)-3*np.nanstd(residuals), np.max(residuals)+3*np.nanstd(residuals)
    # above not used
    #
    legend_without_duplicate_labels(ax,loc='upper left',ncol=1,framealpha=0.5) #handles all the inject start/peak time labels
    #
    ax2.set_xlabel('Time [in units of FWHM of flare]')
    ax.set_ylabel('Relative Flux')
    ax2.set_ylabel('Standardized \nResiduals')
    ax2.set_ylim(-3.5,3.5)
#     ax.set_xticks([])


    #instead of fixed window lengths (-5,7), what if we make this flexible based on duration?
#     ax2.set_xticklabels(np.arange(-6,6,3))
#     ax.set_xlim(-5,7)
    
#     ax.set_xlim(-t_char_min-1,t_char_max+1)
#     ax2.set_xlim(-t_char_min-1,t_char_max+1)
    if np.isnan(out_fwhm)==True:           
        ax.set_xlim(np.nanmin(time_in_window)-5/60/24,np.nanmax(time_in_window)+5/60/24)
        ax2.set_xlim(np.nanmin(time_in_window)-5/60/24,np.nanmax(time_in_window)+5/60/24)
    else:
        tmin = np.nanmin(characterize_timescale(t,out_tpeak,out_fwhm))-1
        tmax = np.round(np.nanmax(characterize_timescale(t,out_tpeak,out_fwhm)),-1)+10
        ax.set_xlim(-tmax,tmax) #using tmin, makes it asymmetric (flare on leftside instead of center)
        ax2.set_xlim(-tmax,tmax)        


    fig.tight_layout(pad=1)
    pdf.savefig(bbox_inches='tight')
    if show_figs==True:
        plt.show()
    if show_figs==False:
        plt.close()        
        
def pack_up_result_per_iteration_and_plot(pdf,starname,s,result1,result2,result3, \
                                          Npts, flarepts, tstarts,tstart,\
                                          tpeaks,out_tpeak, fpeaks,out_fpeak,\
                                          equiv_durs,ED, durs,dur, times,flare_times,\
                                          models,flare, validations, \
                                          Nflares,Npotflares,\
                                          time,flux,\
                                          time_in_window,flux_in_window,\
                                          t,f, ttemp,ftemp, out_fwhm,\
                                          localT,sigma,\
                                          trise_at_minus_1_FHWM,frise_at_minus_1_FHWM,\
                                          tdecay_at_plus_1_FHWM,fdecay_at_plus_1_FHWM,\
                                          show_figs,N1, N2, N3, mad, median, inject_lc,\
                                          t_char_min,t_char_max):
    ### all scenarios
    ### T T T
    ### T T F
    ### T F T
    ### T F F
    ### F F F
    ### F F T
    ### F T F
    ### F T T
    if (result1==True) & (result2==True) & (result3==True): # TTT
        string=starname+' Flare #'+str(s+1)+': passed validation checks # 1, 2 & 3' 
        plot_validation_checks(pdf,string,time,flux,time_in_window,flux_in_window,\
                           t,f,flare_times,flare,ttemp,ftemp,\
                           tstart,out_tpeak,out_fpeak, out_fwhm,\
                           localT,sigma,\
                           trise_at_minus_1_FHWM,frise_at_minus_1_FHWM,\
                           tdecay_at_plus_1_FHWM,fdecay_at_plus_1_FHWM,\
                           N1, N2, N3, mad, median, \
                           show_figs, inject_lc,\
                              t_char_min,t_char_max,ED,dur)
    if (result1==True) & (result2==True) & (result3==False): # TTF
        string=starname+' Flare #'+str(s+1)+': passed validation checks # 1 & 2, failed # 3' 
        plot_validation_checks(pdf,string,time,flux,time_in_window,flux_in_window,\
                           t,f,flare_times,flare,ttemp,ftemp,\
                           tstart,out_tpeak,out_fpeak, out_fwhm,\
                           localT,sigma,\
                           trise_at_minus_1_FHWM,frise_at_minus_1_FHWM,\
                           tdecay_at_plus_1_FHWM,fdecay_at_plus_1_FHWM,\
                           N1, N2, N3, mad, median, \
                           show_figs, inject_lc,\
                              t_char_min,t_char_max,ED,dur)
    if (result1==True) & (result2==False) & (result3==True): # TFT
        string=starname+' Flare #'+str(s+1)+': passed validation checks # 1 & 3, failed # 2' 
        plot_validation_checks(pdf,string,time,flux,time_in_window,flux_in_window,\
                           t,f,flare_times,flare,ttemp,ftemp,\
                           tstart,out_tpeak,out_fpeak, out_fwhm,\
                           localT,sigma,\
                           trise_at_minus_1_FHWM,frise_at_minus_1_FHWM,\
                           tdecay_at_plus_1_FHWM,fdecay_at_plus_1_FHWM,\
                           N1, N2, N3, mad, median, \
                           show_figs, inject_lc,\
                              t_char_min,t_char_max,ED,dur)
    if (result1==True) & (result2==False) & (result3==False): # TFF
        string=starname+' Flare #'+str(s+1)+': passed validation check # 1, failed # 2 & 3' 
        plot_validation_checks(pdf,string,time,flux,time_in_window,flux_in_window,\
                           t,f,flare_times,flare,ttemp,ftemp,\
                           tstart,out_tpeak,out_fpeak, out_fwhm,\
                           localT,sigma,\
                           trise_at_minus_1_FHWM,frise_at_minus_1_FHWM,\
                           tdecay_at_plus_1_FHWM,fdecay_at_plus_1_FHWM,\
                           N1, N2, N3, mad, median, \
                           show_figs, inject_lc,\
                              t_char_min,t_char_max,ED,dur)  
    if (result1==False) & (result2==True) & (result3==False): # FFF
        string=starname+' Flare #'+str(s+1)+': passed validation check # 2, failed # 1 & 3' 
        plot_validation_checks(pdf,string,time,flux,time_in_window,flux_in_window,\
                           t,f,flare_times,flare,ttemp,ftemp,\
                           tstart,out_tpeak,out_fpeak, out_fwhm,\
                           localT,sigma,\
                           trise_at_minus_1_FHWM,frise_at_minus_1_FHWM,\
                           tdecay_at_plus_1_FHWM,fdecay_at_plus_1_FHWM,\
                           N1, N2, N3, mad, median, \
                           show_figs, inject_lc,\
                              t_char_min,t_char_max,ED,dur)
    if (result1==False) & (result2==False) & (result3==True): # FFT
        string=starname+' Flare #'+str(s+1)+': failed validation checks # 1 & 2, passed 3' 
        plot_validation_checks(pdf,string,time,flux,time_in_window,flux_in_window,\
                           t,f,flare_times,flare,ttemp,ftemp,\
                           tstart,out_tpeak,out_fpeak, out_fwhm,\
                           localT,sigma,\
                           trise_at_minus_1_FHWM,frise_at_minus_1_FHWM,\
                           tdecay_at_plus_1_FHWM,fdecay_at_plus_1_FHWM,\
                           N1, N2, N3, mad, median, \
                           show_figs, inject_lc,\
                              t_char_min,t_char_max,ED,dur)  
    if (result1==False) & (result2==False) & (result3==False): # FFF
        string=starname+' Flare #'+str(s+1)+': failed validation checks # 1, 2 & 3' 
        plot_validation_checks(pdf,string,time,flux,time_in_window,flux_in_window,\
                           t,f,flare_times,flare,ttemp,ftemp,\
                           tstart,out_tpeak,out_fpeak, out_fwhm,\
                           localT,sigma,\
                           trise_at_minus_1_FHWM,frise_at_minus_1_FHWM,\
                           tdecay_at_plus_1_FHWM,fdecay_at_plus_1_FHWM,\
                           N1, N2, N3, mad, median, \
                           show_figs, inject_lc,\
                              t_char_min,t_char_max,ED,dur)
    if (result1==False) & (result2==True) & (result3==True): # FTT
        string=starname+' Flare #'+str(s+1)+': failed validation check # 1, passed # 2 & 3' 
        plot_validation_checks(pdf,string,time,flux,time_in_window,flux_in_window,\
                           t,f,flare_times,flare,ttemp,ftemp,\
                           tstart,out_tpeak,out_fpeak, out_fwhm,\
                           localT,sigma,\
                           trise_at_minus_1_FHWM,frise_at_minus_1_FHWM,\
                           tdecay_at_plus_1_FHWM,fdecay_at_plus_1_FHWM,\
                           N1, N2, N3, mad, median, \
                           show_figs, inject_lc,\
                              t_char_min,t_char_max,ED,dur)        

        
    return result1,result2,result3, Npts, flarepts, tstarts,tstart,tpeaks,out_tpeak, fpeaks,out_fpeak, equiv_durs,ED, durs,dur, times, flare_times, models,flare, validations, Nflares, Npotflares 




def flare_validation_summary_plot(savepath, Nflares, LC_summary, inputLC, LCDF, cleanedDF, starname, inject_lc, N1,N2,N3, smooth_window_in_hours, show_fig):
    Ns = LC_summary.loc[LC_summary['Validation']=='N'].reset_index(drop=True) #failed flare validation
    Ys = LC_summary.loc[LC_summary['Validation']=='Y'].reset_index(drop=True) #passed flare validation
    print('Ns:',len(Ns),'Ys:',len(Ys))
    #
    fig = plt.figure(figsize=(10,5))
    ax1=fig.add_subplot(311)
    ax2=fig.add_subplot(312)
    ax3=fig.add_subplot(313)
    #
    ax1.plot(np.array(inputLC['time']),np.array(inputLC['flux']),\
             color='red',marker='.',linestyle='none',label='Raw')
    ax1.plot(LCDF['Detrended Time'],LCDF['Detrended Flux'],'k.',label='Smoothed with '+str(smooth_window_in_hours)+' hr window')
    ax1.set_xlabel('Time (BKJD)')
    ax1.set_ylabel('Relative Flux')
    ax1.set_ylim(np.min(np.array(inputLC['flux']))-10*np.std(np.array(inputLC['flux'])),\
                 np.max(np.array(inputLC['flux']))+10*np.std(np.array(inputLC['flux'])))
    ax1.legend(loc='best',ncol=2,framealpha=0.5,fancybox=True)
    #
    ax2.plot(LCDF['Detrended Time'],LCDF['Detrended Flux'],'k.',label='Smoothed data')
    #     ax2.plot(FlareDF['flare times'], FlareDF['flare models']+np.nanmedian(detLC.flux),color='grey',marker='.',linestyle='none',label='Flagged flares')
    if inject_lc is not None:
        for n in range(len(inject_lc)):
            ax1.axvline(inject_lc['tpeak'][n].item(),color='orange',linestyle='-',zorder=-1)
            ax2.axvline(inject_lc['tpeak'][n].item(),color='orange',linestyle='-',zorder=-1)
            ax3.axvline(inject_lc['tpeak'][n].item(),color='orange',linestyle='-',zorder=-1)
        ax1.axvline(inject_lc['tpeak'][n].item(),color='orange',linestyle='-',label='Injected',zorder=-1)
        ax2.axvline(inject_lc['tpeak'][n].item(),color='orange',linestyle='-',label='Injected',zorder=-1)
        ax3.axvline(inject_lc['tpeak'][n].item(),color='orange',linestyle='-',label='Injected',zorder=-1)            
    #
    if len(Ns)==0:
        pass
    else:
        if len(Ns)>2:
            for n in range(len(Ns)-1):
                ax2.axvline(x=Ns['tpeak'][n].item(),color='red',linestyle='--',linewidth=3,zorder=-10)
            ax2.axvline(x=Ns['tpeak'][n+1].item(),color='red',linestyle='--',linewidth=3,zorder=-10,label='Failed flare validation')
        else:
            ax2.axvline(x=Ns['tpeak'][0].item(),color='red',linestyle='--',linewidth=3,zorder=-10,label='Failed flare validation')
    if len(Ys)==0:
        pass
    else:
        if len(Ys)>2:
            for y in range(len(Ys)-1):
                ax2.axvline(x=Ys['tpeak'][y].item(),color='cyan',linestyle='-',linewidth=3,zorder=-10)
            ax2.axvline(x=Ys['tpeak'][y+1].item(),color='cyan',linestyle='-',linewidth=3,zorder=-10,label='Passed flare validation')
        else:
            ax2.axvline(x=Ys['tpeak'][0].item(),color='cyan',linestyle='-',linewidth=3,zorder=-10,label='Passed flare validation')
    ax2.set_xlabel('Time (BKJD)')
    ax2.set_ylabel('Relative Flux')
    ax2.set_ylim(np.min(np.array(inputLC['flux']))-10*np.std(np.array(inputLC['flux'])),\
                 np.max(np.array(inputLC['flux']))+20*np.std(np.array(inputLC['flux'])))
    ax2.legend(loc='best',ncol=4,framealpha=1,fancybox=True)
    #
    ax3.plot(np.array(inputLC['time']),np.array(inputLC['flux']),\
             color='grey',marker='.',linestyle='none')
    ax3.plot(cleanedDF['Time'],cleanedDF['Flux'],'k.',label='Flare Removed data')
    ax3.set_xlabel('Time (BKJD)')
    ax3.set_ylabel('Relative Flux')
    ax3.set_ylim(np.min(np.array(inputLC['flux']))-10*np.std(np.array(inputLC['flux'])),\
                 np.max(np.array(inputLC['flux']))+10*np.std(np.array(inputLC['flux'])))
    ax3.legend(loc='best',ncol=2,framealpha=0.5,fancybox=True)
    fig.suptitle(starname+' Number of flares: '+str(Nflares),x=0.55,y=1.02)
    fig.tight_layout(pad=1)
    cadence=np.nanmedian(np.diff(inputLC['time']))
    fig.savefig(savepath+starname+'_flare_analysis_'+str(int(np.round(cadence*60*60*24,0)))+'s_N1_'+str(N1)+'_N2_'+str(N2)+'_N3_'+str(N3)+'_.png',bbox_inches='tight')
    if show_fig==True:
        plt.show()
    else:
        plt.close()
        
def scatter_hist(LC_summary, inject_lc,ax, ax_histx, ax_histy, dur_type, Nbins=200):
    #Nbins can also be set to 'auto'
    LC_summary = LC_summary.dropna().reset_index(drop=True) # remove NaN rows for plotting
    validated = LC_summary.loc[LC_summary['Validation']=='Y'].reset_index(drop=True)
    if inject_lc is not None:
        try:
            try:
                injx,injy = inject_lc[dur_type]*24*60, inject_lc['Fpeak']
            except KeyError:
                if dur_type=='dur':
                    injx,injy = inject_lc['duration']*24*60, inject_lc['Fpeak']
        except KeyError:
            try:
                injx,injy = inject_lc[dur_type]*24*60, inject_lc['fpeak']
            except KeyError:
                if dur_type=='dur':
                    injx,injy = inject_lc['duration']*24*60, inject_lc['fpeak']            
    allx, ally = LC_summary[dur_type]*24*60, LC_summary['Fpeak']
#     allx, ally = np.array(LC_summary['equiv_dur'])[np.isnan(np.array(LC_summary['equiv_dur']))==False]*24*60, np.array(LC_summary['Fpeak'])[np.isnan(np.array(LC_summary['equiv_dur']))==False]
    x, y = validated[dur_type]*24*60, validated['Fpeak']


    x = x.loc[x>0].dropna().reset_index(drop=True)
    allx = allx.loc[allx>0].dropna().reset_index(drop=True)
    y = y.loc[y>0].dropna().reset_index(drop=True)
    ally = ally.loc[ally>0].dropna().reset_index(drop=True)     

    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    try:
        ax.scatter(allx, ally,color='grey',label='Potential flares ('+str(len(allx)-len(validated))+')',zorder=-10)
        ax.scatter(x, y,color='red',label='Validated flares ('+str(len(validated))+')',zorder=0)
        if inject_lc is not None:
            ax.scatter(injx, injy,color='green',label='Injected flares ('+str(len(inject_lc))+')',zorder=1)
    except ValueError as e:
        print(e,' on mismatched lengths')
        print('all x, all y',len(allx),len(ally),' from LC_summary')
        print('x , y',len(x),len(y),' from Validated')
        if inject_lc is not None:
            print('injx , injy',len(injx),len(injy),' from Injected')

    hist,bins=np.histogram((allx),bins=Nbins)#'auto')
    histb,binsb=np.histogram((ally),bins=Nbins)#'auto')    
    hist2,bins2=np.histogram((x),bins=Nbins)#'auto')    
    hist2b,bins2b=np.histogram((y),bins=Nbins)#'auto')
    if inject_lc is not None:
        hist0,bins0=np.histogram((injx),bins=Nbins)#'auto')    
        hist0b,bins0b=np.histogram((injy),bins=Nbins)#'auto')        

    ax_histx.hist(allx, bins=bins,color='grey',zorder=-10)
    ax_histy.hist(ally, bins=binsb,color='grey', orientation='horizontal',zorder=-10)

    ax_histx.hist(x, bins=bins2,color='red',zorder=0)
    ax_histy.hist(y, bins=bins2b,color='red', orientation='horizontal',zorder=0)
    if inject_lc is not None:    
        ax_histx.hist(injx, bins=bins0,color='green',zorder=1)
        ax_histy.hist(injy, bins=bins0b,color='green', orientation='horizontal',zorder=1)

    ax_histx.set_yticks([0,25,50,75,100])
    ax_histy.set_xticks([0,25,50,75,100])
    [label.set_visible(False) for label in ax_histx.get_xticklabels()]
    [label.set_visible(False) for label in ax_histy.get_yticklabels()]    

def plot_histogram(savepath,starname,LC_summary,inject_lc,show_fig,dur_type):

    # start with a square Figure
    fig = plt.figure(figsize=(8, 8))

    gs = fig.add_gridspec(2, 2,  width_ratios=(7, 2), height_ratios=(2, 7),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.05)

    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax_histx.set_xscale('log')
    ax_histx.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    #
    ax_histy.set_yscale('log')
    ax_histy.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    # use the previously defined function
    scatter_hist(LC_summary, inject_lc,ax, ax_histx, ax_histy,dur_type)
    
    ax.set_xlabel('Flare Duration [Minutes]')
    ax.set_ylabel('Relative Flare Amplitude')
    ax_histx.set_ylabel('Frequency')
    ax_histy.set_xlabel('Frequency')    

    ax.set_xticks([1,2,3,4,5,10,20,30,40,50,100,200])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    ax.set_yticks([0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    ax.legend(loc='upper right')
    plt.setp(ax_histx, xlim=ax.get_xlim())
    plt.setp(ax_histy, ylim=ax.get_ylim())    
    #
    gs.tight_layout(fig,pad=1)
    fig.savefig(savepath+starname+'_histogram.png')
    if show_fig==True:
        plt.show()
    else:
        plt.close()
        
# NEW FUNCTION
# NEW FUNCTION
# NEW FUNCTION
def find_gaps_ALT(time, maxgap=0.09, minspan=10):
        '''
        Find gaps in light curve and stores them in the gaps attribute.

        Parameters
        ------------
        time : numpy array with floats
            sorted array, in units of days
        maxgap : 0.09 or float
            maximum time gap between two datapoints in days,
            default equals approximately 2h
        minspan : 10 or int
            minimum number of datapoints in continuous observation,
            i.e., w/o gaps as defined by maxgap

        Returns
        --------
        FlareLightCurve

        '''
        dt = np.diff(time)
        gap = np.where(np.append(0, dt) >= maxgap)[0]
        # add start/end of LC to loop over easily
        gap_out = np.append(0, np.append(gap, len(time)))

        # left start, right end of data
        left, right = gap_out[:-1], gap_out[1:]

        #drop too short observation periods
        too_short = np.where(np.diff(gap_out) < 10)
        left, right = np.delete(left,too_short), np.delete(right,(too_short))
        gaps = list(zip(left, right))
        return gaps        
def FINDFLARE_ALT(flux, error, N1=3, N2=2, N3=3):
    # from PLATO V6
    from scipy import stats
    #
    isflare = np.zeros_like(flux, dtype='bool')
    median=np.nanmedian(flux)# stats.median_abs_deviation(flux) 
    mean=np.nanmean(flux)
    for i in range(10):
        sigma = np.nanstd(flux[~isflare])
        T0 = flux - median # excursion should be positive #"N0"
        T1 = np.abs(flux - median) / sigma #N1
        T2 = np.abs(flux - median - error) / sigma #N2

        # apply thresholds N0-N2:
        pass_thresholds = np.where((T0 > 0) & (T1 > N1) & (T2 > N2))

        #array of indices where thresholds are exceeded:
        is_pass_thresholds = np.zeros_like(flux)
        is_pass_thresholds[pass_thresholds] = 1

        # Need to find cumulative number of points that pass_thresholds
        # Counted in reverse!
        # Examples reverse_counts = [0 0 0 3 2 1 0 0 1 0 4 3 2 1 0 0 0 1 0 2 1 0]
        #                 isflare = [0 0 0 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0]

        reverse_counts = np.zeros_like(flux, dtype='int')
        for k in range(2, len(flux)):
            reverse_counts[-k] = (is_pass_thresholds[-k]
                                    * (reverse_counts[-(k-1)]
                                    + is_pass_thresholds[-k]))

        # find flare start where values in reverse_counts switch from 0 to >=N3
        istart_i = np.where((reverse_counts[1:] >= N3) &
                            (reverse_counts[:-1] - reverse_counts[1:] < 0))[0] + 1

        # use the value of reverse_counts to determine how many points away stop is
        istop_i = istart_i + (reverse_counts[istart_i])
        isflare = np.zeros_like(flux, dtype='bool')

        for (l,r) in list(zip(istart_i,istop_i)):
            isflare[l:r+1] = True
    return isflare
# NEW FUNCTION
# NEW FUNCTION
# NEW FUNCTION


def full_pipeline(ID, mission,cadence, inject_lc,\
                   savepath, downloadpath, threshold_type='global', \
                   N1=2, N2=2, N3=3, maxgap=2/24,minspan=10, smooth_window_in_hours=12.0, \
                   flare_sep_in_cadences=6,input_lc=None,filter_type='medfilt',\
                  show_figs=True,verbose=False,t_char_min=5,t_char_max=7,version='Davenport'):
    import os,sys
    starname=mission+' '+str(ID)                  
    starname2=mission+'_'+str(ID)
    # input params:
    # N1 = Noise threshold
    # N2 = A weighted noise threshold
    # N3 = number of consecutive points to consider as a flare that satisfy N1, N2 criteria    
    import time as clock
    start=clock.time()                  

    #STEP 0: check if directories exist
    #check if save/download paths exist
    if os.path.exists(savepath)==True:
        pass
    else:
        os.makedirs(savepath)

    if os.path.exists(downloadpath)==True:
        pass
    else:
        os.makedirs(downloadpath)
    #
    #Step 0.5: check input light curve 
    if input_lc is None:    
        # get public data light curve
        inputLC = get_lc(ID,mission,cadence,downloadpath)
    else:
        inputLC=input_lc # 'time' , 'flux', 'flux_err' columns                  
    
    # Step 1: Smooth input LC
    from scipy import stats
    #Step 2a: Smooth light curve to remove stellar variation
    print('smoothing window (hrs): ',np.round(smooth_window_in_hours,5))
    window_size_in_hours = smooth_window_in_hours
    inputLC = input_lc
    filter_type = 'medfilt'# quicker than biweight
    
    inputLC2, detLC, trend_lc = smoother(window_size_in_hours,inputLC.time,inputLC.flux,filter_type=filter_type)
#     what if we just don't smooth at all?    
#     detLC = inputLC


    # make LCDF dataframe as a output at the end to make summary figure
    try:
        LCDF = pd.DataFrame({"Raw Time":np.array(inputLC['time']),\
                             "Raw Flux":np.array(inputLC['flux']),\
                             "Raw Error":np.array(inputLC['flux_err']),\
                             "Detrended Time":np.array(detLC['time']),\
                             "Detrended Flux":np.array(detLC['flux']),\
                             "Detrended Error":np.array(detLC['flux_err']),\
                             "Fitted trend":trend_lc})
    except ValueError as VE:
        LCDF = pd.DataFrame({"Raw Time":np.array(inputLC['time'])[:-1],\
                             "Raw Flux":np.array(inputLC['flux'])[:-1],\
                             "Raw Error":np.array(inputLC['flux_err'])[:-1],\
                             "Detrended Time":np.array(detLC['time']),\
                             "Detrended Flux":np.array(detLC['flux']),\
                             "Detrended Error":np.array(detLC['flux_err']),\
                             "Fitted trend":trend_lc})
        
    def find_cts_flare(time,flux,error,N1,N2,N3,maxgap,minspan):
        all_flares_ALT=[]
        gaps = find_gaps_ALT(time,maxgap=maxgap,minspan=minspan)
        for (le,ri) in gaps:
            ind_fl = FINDFLARE_ALT(flux[le:ri],error[le:ri],N1=N1, N2=N2, N3=N3)
    #         ind_fl = find_flares_in_cont_obs_period_ALT(flux[le:ri], error[le:ri],
    #                                                 N1=N1,N2=N2,N3=N3)
            all_flares_ALT=np.append(all_flares_ALT,ind_fl).astype(bool)
    #     all_flares_ALT=all_flares_ALT.astype(bool)    
        return all_flares_ALT
    
    time = LCDF['Detrended Time']
    flux = LCDF['Detrended Flux']
    error = LCDF['Detrended Error']
    ind_fl = find_cts_flare(time,flux,error, 
                            N1=N1, N2=N2, N3=N3,
                            maxgap=maxgap,minspan=minspan)
    
    # Step 3: Separate flares
    fwhm_round = 1
    fail_count=0
    fail_ind = []


    #flare models
    times=[]
    models=[]        

    #flare stats
    tstarts=[]
    tpeaks=[]
    fpeaks=[]
    Npts=[]
    validations=[]
    equiv_durs=[]
    durs = []
    fwhms = []
    Nflares=0
    Npotflares=0
    #
    flare_inds = []
    
    print('flare_sep_in_cadences:',flare_sep_in_cadences)
    cadence = np.nanmedian(np.diff(time)) #in days
    print('cadence (in seconds)',cadence*3600*24)
    flare_sep = flare_sep_in_cadences*cadence

    print('flare_sep (in days):',flare_sep,';  (in seconds):',flare_sep*3600*24)
    print('')
    
    time =  np.array(LCDF['Raw Time'])
    input_flux =  np.array(LCDF['Raw Flux'])
    flux =  np.array(LCDF['Raw Flux']) - np.nanmedian(np.array(LCDF['Raw Flux'])) #1 #centered on zero
    error = np.array(LCDF['Raw Error'])    
    
    separation_ind_fl = np.where(np.diff(time[ind_fl]) > flare_sep)[0]  
    print('')
    if inject_lc is not None:
        print(len(inject_lc),'Injected flares')
    print('Identified ',len(separation_ind_fl),' seperate flare like events')
    print('Beginning Validation Now')
    
            
    from matplotlib.backends.backend_pdf import PdfPages
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    with PdfPages(savepath+str(starname2)+'_flares_'+str(int(np.round(cadence*60*60*24,0)))+'s_N1_'+str(N1)+'_N2_'+str(N2)+'_N3_'+str(N3)+'_'+version+'.pdf') as pdf:
        for s in range(len(separation_ind_fl)):                
            if verbose==True:
                print(' ')
                print('Flare #',s+1)
            ttemp = time[ind_fl]
            ftemp = flux[ind_fl]
            etemp = error[ind_fl]
            ##
            #
            # time_in_window, flux_in_window, error_in_window = separate_flares(s,separation_ind_fl, ttemp, ftemp, etemp) #OLD    
            time_in_window, flux_in_window, error_in_window = separate_flares(s,separation_ind_fl, ind_fl, time,flux,error,\
                                                                              N_before_pts=N3*3)
            #
            ##
            try:
                tstart=np.nanmin(time_in_window) #NEED START OF MODELED FLARE NOT START OF WINDOW
                print('Flagged flare starts at:',tstart) 
            except ValueError: #if window is empty
                print('ValueError: empty window?')
                print('len check:',len(time_in_window),len(flux_in_window),len(error_in_window))
                print('')
                continue
            ##
            # make initial flare parameter guesses
            ##
            cadence,f_half0, fwhm0,tpeak0,fpeak0 = make_initial_flare_guess(time_in_window,flux_in_window)
            ##
            ##
            # #Doing some basic vetting
            # defining noise thresholds
            ##
            ###
            mad, median, sigma, localT = get_noise_stats(flux,flux_in_window,threshold_type,N1)            
            # 
            # model flares from initial guess
            ##
            try:
#                 flare_times, flare, out_tpeak, out_fwhm, out_fpeak = model_flares_ALT(time_in_window,flux_in_window,error_in_window,N3,localT,verbose,t_char_min,t_char_max,version)#OLD
                flare_times, flare, out_tpeak, out_fwhm, out_fpeak = model_flares_ALT(time_in_window,flux_in_window,error_in_window,N3,localT,verbose,t_char_min,t_char_max,version)
                #EXPERIMENTAL LINES
                #EXPERIMENTAL LINES
                #EXPERIMENTAL LINES            
                if fwhm0 > 0.1:
                    flare_times, flare, out_tpeak, out_fwhm, out_fpeak = model_flares_ALT3(time_in_window,flux_in_window,error_in_window,N3,localT,verbose,t_char_min,t_char_max,version)
                #EXPERIMENTAL LINES
                #EXPERIMENTAL LINES
                #EXPERIMENTAL LINES                    
            except TypeError as TE:
                print('TypeError')
                print('')        
                continue
            ###
            ###
            ##
            # Validation check 1:
            input_flux_median = np.nanmedian(input_flux)
            result1, flarepts, ED, dur, consec, n_consec, consec_real, n_consec_real = validation_check_1(time_in_window,flux_in_window,flare, flare_times, localT, N3, input_flux_median, verbose)
            ###
            ##
            #
            result2, mint, t, f, trise_at_minus_1_FHWM, frise_at_minus_1_FHWM, tdecay_at_plus_1_FHWM, fdecay_at_plus_1_FHWM, mask_between_0_and_1_FWHM, mask_between_minus_1_and_0_FWHM = validation_check_2(time_in_window, flux_in_window, out_tpeak, out_fwhm, out_fpeak, flare_times,verbose,t_char_min,t_char_max)
            #
            ##
            ###
        #         if result2==True: #continue onto check 3
            ###
            ###
            #
            # Validation check 3:
            result3 = validation_check_3(trise_at_minus_1_FHWM, \
                                         tdecay_at_plus_1_FHWM, out_tpeak, out_fwhm,verbose)
            #
            #
            #
            result1,result2,result3, Npts, flarepts, tstarts,tstart,tpeaks,out_tpeak, fpeaks,out_fpeak, equiv_durs,ED, durs,dur, times,flare_times, models,flare, validations, Nflares, Npotflares  = pack_up_result_per_iteration_and_plot(pdf,starname,s,result1,result2,result3, \
                                          Npts, flarepts, tstarts,tstart,\
                                          tpeaks,out_tpeak, fpeaks,out_fpeak,\
                                          equiv_durs,ED,durs,dur, times,flare_times,\
                                          models,flare, validations, \
                                          Nflares,Npotflares,\
                                          time,flux,\
                                          time_in_window,flux_in_window,\
                                          t,f, ttemp,ftemp, out_fwhm,\
                                          localT,sigma,\
                                          trise_at_minus_1_FHWM,frise_at_minus_1_FHWM,\
                                          tdecay_at_plus_1_FHWM,fdecay_at_plus_1_FHWM,\
                                          show_figs,N1, N2, N3, mad, median, inject_lc,\
                              t_char_min,t_char_max)
            ####
            if (result1==True) & (result2==True) & (result3==True):
                if verbose==True:
                    print('passed all conditions')
                #pass conditions
                Npts=np.append(Npts,flarepts)
                tstarts=np.append(tstarts,tstart)            
                tpeaks=np.append(tpeaks,out_tpeak)
                fpeaks=np.append(fpeaks,out_fpeak)                    
                equiv_durs=np.append(equiv_durs,ED)
                durs = np.append(durs,dur)
                fwhms = np.append(fwhms,out_fwhm)
                #
                times=np.append(times,flare_times)
                models=np.append(models,flare)
                validations=np.append(validations,'Y')
                Nflares+=1  
            else: 
                if verbose==True:                
                    print('failed all conditions')
                #fail conditions        
                Npts=np.append(Npts,flarepts)
                tstarts=np.append(tstarts,tstart)            
                tpeaks=np.append(tpeaks,out_tpeak)
                fpeaks=np.append(fpeaks,out_fpeak)                    
                equiv_durs=np.append(equiv_durs,ED) 
                durs = np.append(durs,dur)
                fwhms = np.append(fwhms,out_fwhm)
                #
                times=np.append(times,np.nan)
                models=np.append(models,np.nan)
                validations=np.append(validations,'N')
                Npotflares+=1
    #
    # Step 4: identify data points with validated flares and pack up results as dataframes.
    if len(times)>0:            
        ndigits=13
        around_t = set(np.around(times,ndigits))
        index_bigt = np.array([i for i,b in enumerate(np.around(time,ndigits)) if b in around_t]).astype(int)
        print('Nflare times:',len(times),'N index_bigt:',len(index_bigt))
        #removing flagged flares
        newt=np.delete(np.array(inputLC['time']),index_bigt)
        newf=np.delete(np.array(inputLC['flux']),index_bigt)
        newe=np.delete(np.array(inputLC['flux_err']),index_bigt)
        #
    else:
        newt=np.array(inputLC['time'])
        newf=np.array(inputLC['flux'])
        newe=np.array(inputLC['flux_err'])
    FlareDF = pd.DataFrame({"flare times":times,"flare models":models})
    cleanedDF = pd.DataFrame({"Time":newt,"Flux":newf, "Error":newe})
    #print('len check:',len(tstarts),len(tpeaks),len(fpeaks),len(Npts), len(equiv_durs),len(validations))
    # print('quick lc sum len check',len(equiv_durs),len(durs))
    LC_summary = pd.DataFrame({"tstart":tstarts,"tpeak":tpeaks,"Fpeak":fpeaks,\
                               "Npts":Npts,"equiv_dur":equiv_durs,"dur":durs,"FWHM":fwhms,\
                               "Validation":validations }) 
    
    FlareDF.to_csv(savepath+str(starname2)+'_ModeledFlares_N1_'+str(N1)+'_N2_'+str(N2)+'_N3_'+str(N3)+'_'+version+'.csv',index=False)
    cleanedDF.to_csv(savepath+str(starname2)+'_cleanedLC_N1_'+str(N1)+'_N2_'+str(N2)+'_N3_'+str(N3)+'_'+version+'.csv',index=False)
    LC_summary.to_csv(savepath+str(starname2)+'_LC_summary_N1_'+str(N1)+'_N2_'+str(N2)+'_N3_'+str(N3)+'_'+version+'.csv',index=False)        
    LCDF.to_csv(savepath+str(starname2)+'_LC_N1_'+str(N1)+'_N2_'+str(N2)+'_N3_'+str(N3)+'_'+version+'.csv')        
    ###
    # Step 5: Create Summary plots
    print('')
    print('')
    print('')
    print('Creating Summary Plots')
    #
    flare_validation_summary_plot(savepath, Nflares, LC_summary, inputLC, LCDF, cleanedDF, starname2,\
                                  inject_lc,\
                                  N1=N1,N2=N2,N3=N3,\
                                  smooth_window_in_hours=smooth_window_in_hours,\
                                  show_fig=show_figs)
    #
    plot_histogram(savepath,starname2,LC_summary,inject_lc,show_fig=show_figs,dur_type='dur')
    #plot_histogram(savepath,starname2,LC_summary,show_fig=show_figs,dur_type='equiv_dur')
    print(' ')
    end=clock.time()
    runtime=end-start
    if runtime>60:
        print('runtime: ',np.round(runtime/60,1),' minutes')
    else:
        print('runtime: ',runtime,' seconds')                                    
              
#
#
#
#########################################################
#########################################################
################### Change Log ##########################
#########################################################
#########################################################
#
# ## Change log for Flare Flagger V7
# ## 
# ## August 11th, 2023
# ## - Added get_ID function
# ## - Added jupyter notebook tutorial and uploaded to GitHub. Will now start
# ##   regularly uploading there. Eventually, will make it pip installable once
# ##   I settle on a program name.
# ## 
# ## July 22nd, 2023
# ## - Incorporated several techniques from AltaiPony
# ##       - Including the "find_gaps" and "find_flares_in_cont_obs_period" functions
# ##             - From testing on Wolf 359's QPP flares in K2 data, this seems to work 
# ##               better on separating multi-peak flare events.
# ## - Also fixed implementation of "model_flares_ALT3" for events with FWHM > 0.1
# ## 
# ## 
# ## April 26th, 2023
# ## - Tried altering ??? from V5.1
# ## 
# ## April 3rd,2023
# ## 
# ## - Decided to make a change log
# ## - I Noticed that saved flare models look flat in the rise phases when compared to the input fluxes.
# ##   I think instead of using the smoothed LCs for validation, the smoothed LCs should be used just to
# ##   DETECT events and the input fluxes should be used for VALIDATION. Let's try it out...