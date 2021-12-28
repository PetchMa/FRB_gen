import numpy as np
import matplotlib.pyplot as plt
import psrsigsim as pss
from .utils import gaussian
from numba import jit
from scipy.stats import skewnorm

class Observation(object):
    '''
    Facilitate the creation of entirely synthetic FRB pulse
    '''
    def __init__(self,
                 f0=1500,
                 bw=800,
                 Nf=2048,
                 Nt = 76,
                 df=None,
                 dt=None,
                 false_period = 0.00457,
                 func_freq = None,
                 func_time = None,
                 mu_time =-0.5,
                 sigma_time = 0.1,
                 loc_freq =-1,
                 scale_freq = 0.5,
                 a_norm = 4,
                 Smean = 10,
                 ref_freq = 1500,
                 d_tau = 5e-4,
                 dm = 100,
                 obs_data=None,
                 **kwargs):
        """Generate random points
        Parameters
        ----------
        f0 :center frequency and in MHz 
        bw : bandwidth in MHz 
        Nf : number of frequency bins
        Nt : number of time bins 
        df : frequency resolution MHz 
        dt : time resolution in seconds 
        false_period : fake pulsar period in seconds
        func_freq : function govering the profile as func of freq
        func_time : function govering the profile as a func of time     
        mu_time: mean for guassian time profile 
        sigma_time: deviation for guassian time profile 
        loc_time : location center for skewnorm frequency profile
        scale_time : spread of skewnorm frequency
        a_val : skewnorm parameter 
        Smean : The mean flux of the pulsar  
        d_tau: seconds; note this is an unphysical number and controls scattering
        ref_freq:  MHz, reference frequency of the scatter timescale input   
        dm : dispersion measure
        Returns
        -------
        coordinates : numpy array [num, 2]
        coordinates of where the beams will be in the sky"""
        self.f0 =f0
        self.bw= bw
        self.Nf = Nf
        self.Nt = Nt 
        self.func_freq = func_freq
        self.func_time = func_time

        self.mu_time=mu_time
        self.sigma_time = sigma_time
        self.loc_freq=loc_freq
        self.scale_freq = scale_freq
        self.a_norm = a_norm
        self.Smean = Smean
        self.d_tau = d_tau # seconds; note this is an unphysical number
        self.ref_freq = ref_freq # MHz, reference frequency of the scatter timescale input
        self.dm = dm

        self.sublen = 100
        if df ==None:
            self.df = (1.0/false_period)*76*10**-6 
        
        self.df = df 
        self.false_period = false_period

    def create_FRB_profile(self):
        # setup grid
        time, freq = np.meshgrid(np.linspace(-1, 1, self.Nt), np.linspace(-1, 1, self.Nf))

        if self.func_freq and self.func_time:
            profile = self.func_freq(freq)*self.func_time(time)
        else:
            profile = gaussian(time,  mu=self.mu_time, sig=self.sigma_time )*skewnorm.pdf(freq, self.a_norm ,  
                                                                                        loc=self.loc_freq, 
                                                                                        scale=self.scale_freq)
        # then we normalize it 
        profile_pulse = pss.pulsar.DataProfile(profile/profile.max())

        return profile_pulse

    def create_pulse(self):
        # we set up the signal:
        self.signal = pss.signal.FilterBankSignal(fcent = self.f0, bandwidth = self.bw, Nsubband=self.Nf, sample_rate = self.df,
                                        sublen = self.sublen, fold = True) # fold is set to `True`

        self.profile_pulse = self.create_FRB_profile()

        psr_name = "FRB_test" 

        self.pulse_template = pss.pulsar.Pulsar(self.false_period, self.Smean, profiles=self.profile_pulse, name = psr_name)

        # define the observation length
        self.obslen = 60.0*20

        ism_sim = pss.ism.ISM()
        ism_sim.disperse(self.signal, self.dm)

        # Now add the FD parameter delay, this takes two arguements, the signal and the list of FD parameters
        ism_sim.scatter_broaden(self.signal , d_tau, ref_freq, convolve = True, pulsar = self.pulse_template)

        # Re-make the pulses
        self.pulse_template .make_pulses(self.signal, tobs = self.obslen)

