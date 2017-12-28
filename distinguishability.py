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
h1_paths = glob.glob('HybridAnnex/BNS/Haas500/TaylorT2/Haas_TaylorT2fp2_*.h5')
h2_paths = glob.glob('HybridAnnex/BNS/Haas500/TEOBResum_ROM/Haas_TEOBResum_ROMfp2_*.h5')
h1_paths = h1_paths[:len(h1_paths)-1]
h2_paths = h2_paths[:len(h2_paths)-1]
h1_data = h5py.File(h1_paths[0], 'r')
h2_data = h5py.File(h2_paths[0], 'r')
sim_name = h1_data.attrs['sim_name']
approx1 = h1_data.attrs['approx']
approx2 = h2_data.attrs['approx']
ip2_h1 = 500
ip2_h2 = 500
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

def distinguish(h1_path,h2_path):
    global distance
    global inclination
    hh1 = h5py.File(h1_path, 'r')
    hh2 = h5py.File(h2_path, 'r')
    distance = 1.0
    inclination = 0.0
    m1 = hh1.attrs['ADM_mass1']
    m2 = hh1.attrs['ADM_mass2']
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
    h1p_tilde = np.fft.fft(h1p,norm=None)/sample_rate
    h2p_tilde = np.fft.fft(h2p,norm=None)/sample_rate
    h1p_tilde_new = FrequencySeries(h1p_tilde[:len(h1p)/2], delta_f=delta_f)
    h2p_tilde_new = FrequencySeries(h2p_tilde[:len(h2p)/2], delta_f=delta_f)
    h1p_tilde_new.resize(f_len)
    h2p_tilde_new.resize(f_len)
    x,y,norm11 = matched_filter_core(h1p_tilde_new,h1p_tilde_new,psd,low_frequency_cutoff=f_cutoff)
    x,y,norm22 = matched_filter_core(h2p_tilde_new,h1p_tilde_new,psd,low_frequency_cutoff=f_cutoff)
    x,y,norm12 = matched_filter_core(h1p_tilde_new,h2p_tilde_new,psd,low_frequency_cutoff=f_cutoff)
    sig11 = sigmasq(h1p_tilde_new,psd,low_frequency_cutoff=f_cutoff)
    sig22 = sigmasq(h2p_tilde_new,psd,low_frequency_cutoff=f_cutoff)
    sig12 =sigmasq(h2p_tilde_new,psd,low_frequency_cutoff=f_cutoff)
    m11,i = match(h1p_tilde_new,h1p_tilde_new,psd=psd,low_frequency_cutoff=f_cutoff) 
    m22,i = match(h2p_tilde_new,h2p_tilde_new,psd=psd,low_frequency_cutoff=f_cutoff)  
    m12,i = match(h1p_tilde_new,h2p_tilde_new,psd=psd,low_frequency_cutoff=f_cutoff)
    disting  = m11 + m22 - 2*m12
    disting2 = (m11*sig11**(.5))/norm11 + (m22*sig22**(.5))/norm22 - 2*(m12*sig12**(.5))/norm12  
    return(disting,disting2,hh1_shift_time,hh2_shift_time,hh1_freq,hh2_freq)
    ##       0          1                2             3         4      

distinguish1 = []
distinguish2 = []
hh1_shift_time = []
hh2_shift_time = []
hh1_hybridize_freq = []
hh2_hybridize_freq = []
if __name__ == '__main__':
    p = mp.Pool(mp.cpu_count())
    for i,h1_path in enumerate(h1_paths[:100]):
        h2_path_iter = h2_paths[i:100]
        func_distinguish = partial(distinguish,h1_path)
        comparisons = p.map(func_distinguish,h2_path_iter)
        for tuple_data in comparisons:
            distinguish1.append(tuple_data[0])
            distinguish2.append(tuple_data[1])
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
    with h5py.File(h5_dir+approx1+'-'+approx2+'_psd_'+psd_type+'distinguish.h5','w') as fd:
        fd.attrs.create('group', 'Read_Group')
        fd.attrs.create('approx1', approx1)
        fd.attrs.create('approx2', approx2)
        fd.attrs.create('sim_name', sim_name)
        fd.attrs.create('PSD_name',psd_type)
        fd.create_dataset(approx1+'_shift_time',data=hh1_shift_time)
        fd.create_dataset(approx2+'_shift_time',data=hh2_shift_time)
        fd.create_dataset(approx1+'_hybridize_freq',data=hh1_hybridize_freq)
        fd.create_dataset(approx2+'_hybridize_freq',data=hh2_hybridize_freq)
        fd.create_dataset('distinguish_normed',data=distinguish1)
        fd.create_dataset('distinguish_unnormed',data=distinguish2)
        fd.close()
