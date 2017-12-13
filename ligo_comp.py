import time
import cProfile
import os
import glob
import hybridmodules as hy
from functools import partial
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy as sci
import lal
import lalsimulation as lalsim
from pycbc.filter import match,matched_filter,sigmasq,make_frequency_series,matched_filter_core
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.types import TimeSeries,FrequencySeries
import multiprocessing as mp
from multiprocessing import Pool
from pycbc.waveform import get_td_waveform
h1_paths = glob.glob('HybridAnnex/BNS/B00/TEOBResum_ROMlambda_2684.34/B0_TEOBResum_ROMfp2_*.h5')
h2_paths = glob.glob('HybridAnnex/BNS/B00/TaylorT2lambda_2684.34/B0_TaylorT2fp2_*.h5')
h1_paths = h1_paths[1:len(h1_paths)]
h2_paths = h2_paths[:len(h2_paths)-1]
#for name in h2_paths:
#    print name, 2
#for name in h1_paths:
#    print name, 1
h1_data = h5py.File(h1_paths[0], 'r')
h2_data = h5py.File(h2_paths[0], 'r')
sim_name = h1_data.attrs['sim_name']
approx1 = h1_data.attrs['approx']
approx2 = h2_data.attrs['approx']
ip2_h1 = 0 
ip2_h2 = 0
sim_type = 'BNS'
psd_type = 'aLIGOZeroDetHighPower'
h1_data.close()
h2_data.close()
sample_rate = 4096.0*10
delta_t = 1.0/sample_rate
distance = 1.0
inclination = 0.0

def _poolinit():
    global prof
    prof = cProfile.Profile()
    def finish():
        prof.dump_stats('./Profiles/profile-%s.out' % mp.current_process().pid)
    mp.util.Finalize(None,finish,exitpriority = 1)

def compare(h1_path,h2_path): 
    global distance 
    global inclination
    global _snr
    hh1 = h5py.File(h1_path, 'r')
    hh2 = h5py.File(h2_path, 'r')
    distance = 1.0
    inclination = 0.0
    m1 = hh1.attrs['grav_mass1']
    m2 = hh1.attrs['grav_mass2']
    s1x = hh1.attrs['spin1x']
    s1y = hh1.attrs['spin1y']
    s1z = hh1.attrs['spin1z']
    s2x = hh1.attrs['spin2x']
    s2y = hh1.attrs['spin2y']
    s2z = hh1.attrs['spin2z']
    LNhatx = hh1.attrs['LNhatx']
    LNhaty = hh1.attrs['LNhaty']
    LNhatz = hh1.attrs['LNhatz']
    n_hatx = hh1.attrs['nhatx']
    n_haty = hh1.attrs['nhaty']
    n_hatz = hh1.attrs['nhatx']
    hh_freq = hh1.attrs['hybridize_freq']
    hh1tidal = hh1.attrs['lambda1']
    hh2tidal = hh2.attrs['lambda2']
    f_lower_hh1= hh1.attrs['f_lower_at_1MSUN']/(2.0 * lal.TWOPI * lal.MTSUN_SI*(m1+m2)) ##convert back to Hz
    f_lower_hh2 = hh2.attrs['f_lower_at_1MSUN']/(2.0 * lal.TWOPI * lal.MTSUN_SI*(m1+m2))
    hh1_freq = hh1.attrs['hybridize_freq']
    hh2_freq = hh2.attrs['hybridize_freq']
    hh1_shift_time = hh1.attrs['shift_time']
    hh2_shift_time = hh2.attrs['shift_time']
    hh1.close()
    hh2.close()
    h1p,h1c = get_td_waveform(approximant='NR_hdf5',numrel_data=h1_path,mass1=m1,
                                                              mass2=m2,
                                                              spin1z=s1z,spin1x=s1x,spin1y=s1y,
                                                              spin2z=s2z,spin2x=s2x,spin2y=s2y,
                                                              delta_t=delta_t,distance=distance,
                                                              f_lower=f_lower_hh1,
                                                              inclination=inclination)
    h2p,h2c = get_td_waveform(approximant='NR_hdf5',numrel_data=h2_path,mass1=m1,
                                                              mass2=m2,
                                                              spin1z=s1z,spin1x=s1x,spin1y=s1y,
                                                              spin2z=s2z,spin2x=s2x,spin2y=s2y,
                                                              delta_t=delta_t,distance=distance,
                                                              f_lower=f_lower_hh2,
                                                              inclination=inclination)
    t_len = max(len(h1p), len(h2p))
    h1p.resize(t_len)
    h2p.resize(t_len)
    #Generate the aLIGO ZDHP PSD
    delta_f = 1.0 /h2p.duration
    f_len = t_len/2 + 1
    f_cutoff = 30.0
    #psd_path = '/home/jsread/O2PSDS/lalinferencemcmc-1-V1H1L1-1187008882.45-20.hdf5L1-PSD.txt'
    #real_psd = pycbc.psd.read.from_txt(psd_path, f_len, delta_f, f_cutoff, is_asd_file=True)
    psd = aLIGOZeroDetHighPower(f_len, delta_f, f_cutoff)
    h1p = np.array(h1p)
    h2p = np.array(h2p)
    l2_distance = hy.delta_h(h1p,h2p)
    #print h1_path,'path1'
    #print h2_path,'path2'
    h1p_tilde = np.fft.fft(h1p,norm=None)/sample_rate
    h2p_tilde = np.fft.fft(h2p,norm=None)/sample_rate 
    h1p_tilde_new = FrequencySeries(h1p_tilde[:len(h1p)/2], delta_f=delta_f)
    h2p_tilde_new = FrequencySeries(h2p_tilde[:len(h2p)/2], delta_f=delta_f)
    h1p_tilde_new.resize(f_len)
    h2p_tilde_new.resize(f_len)
    match_plus, i = match(h1p_tilde_new,h2p_tilde_new,psd=psd,low_frequency_cutoff=f_cutoff)
    return(l2_distance,match_plus,hh1_shift_time,hh2_shift_time,hh1_freq,hh2_freq)
    ##       0          1            2             3               4       5
l2_norm = []
match_filter = []
hh1_shift_time = []
hh2_shift_time = []
hh1_hybridize_freq = []
hh2_hybridize_freq = []
if __name__ == '__main__':
    p = mp.Pool(mp.cpu_count())
    for i,h1_path in enumerate(h1_paths[:51]):
        h2_path_iter = h2_paths[i:51]
        func_compare = partial(compare,h1_path)
        comparisons = p.map(func_compare,h2_path_iter)
        for tuple_data in comparisons:
            l2_norm.append(tuple_data[0])
            match_filter.append(tuple_data[1])
            hh1_shift_time.append(tuple_data[2])
            hh2_shift_time.append(tuple_data[3])
            hh1_hybridize_freq.append(tuple_data[4])
            hh2_hybridize_freq.append(tuple_data[5])
    h5_dir = 'HybridAnnex/'+sim_type+'/'+sim_name+str(ip2_h1)+'/Comparisons/'
    if not os.path.exists(os.path.dirname(h5_dir)):
        try:
            os.makedirs(os.path.dirname(h5_dir))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    with h5py.File(h5_dir+approx1+'-'+approx2+'_psd_'+psd_type+'.h5','w') as fd:
        fd.attrs.create('Read_Group', 'Flynn')
        fd.attrs.create('approx1', approx1)
        fd.attrs.create('approx2', approx2)
        fd.attrs.create('sim_name', sim_name)
        fd.attrs.create('PSD_name',psd_type)
        fd.create_dataset(approx1+'_shift_time',data=hh1_shift_time)
        fd.create_dataset(approx2+'_shift_time',data=hh2_shift_time)
        fd.create_dataset(approx1+'_hybridize_freq',data=hh1_hybridize_freq)
        fd.create_dataset(approx2+'_hybridize_freq',data=hh2_hybridize_freq)
        fd.create_dataset('L2_norm',data=l2_norm)
        fd.create_dataset('match_filter',data=match_filter)
        fd.close()
'''
tlen = max(len(hplus), len(hplus1))
hplus.resize(tlen)
hplus1.resize(tlen)
delta_f = 1.0 / hplus1.duration
flen = tlen/2 + 1
psd = aLIGOZeroDetHighPower(flen, delta_f, f_low)
x,y,norm11 = matched_filter_core(hplus,hplus,psd,low_frequency_cutoff=f_low)
m11, i = match(hplus, hplus, psd=psd, low_frequency_cutoff=f_low)
sig11 = sigmasq(hplus,psd,low_frequency_cutoff=f_low)
x,y,norm22 = matched_filter_core(hplus1,hplus1,psd,low_frequency_cutoff=f_low)
m22, i = match(hplus1, hplus1, psd=psd, low_frequency_cutoff=f_low)
sig22 = sigmasq(hplus1,psd,low_frequency_cutoff=f_low)
x,y,norm12 = matched_filter_core(hplus,hplus1,psd,low_frequency_cutoff=f_low)
m12, i = match(hplus, hplus1, psd=psd, low_frequency_cutoff=f_low)
sig12 =sigmasq(hplus1,psd,low_frequency_cutoff=f_low)
disting = (m11*sig11**(.5)/norm11+m22*sig22**(.5)/norm22-2*m12*sig12**(.5)/norm12)**.5
print disting
'''
