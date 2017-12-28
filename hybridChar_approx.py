import sys
import time
import h5py
import numpy as np
import os 
import re
import scipy.signal as sig
import hashlib
from scipy import interpolate as inter
import scipy as sci
import pycbc
from pycbc.waveform import td_approximants, get_td_waveform
from pycbc.types.timeseries import TimeSeries
from pycbc.waveform import utils as wfutils
from pycbc import pnutils
import glob
import multiprocessing as mp
from multiprocessing import Pool
import hybridmodules as hy 
from functools import partial
import random
#import matplotlib
import matplotlib.pyplot as plt
import romspline
#### Things to do: find a way to write the "true" starting sim freq. Place the new format
TWOPI = 2.0*np.pi
MRSUN_SI = 1476.6250614
PC_SI = 3.08567758149e+16
MTSUN_SI = 4.92549102554e-06
solar_mass_mpc = MRSUN_SI / (1e6*PC_SI)
Sph_Harm = 0.6307831305 ## 2-2 spherical harmoniic at theta = 0 phi = 90 
sample_rate = 4096.0*8.0
delta_t = 1.0/sample_rate

#### First select PN you want to compare to
sim_name = 'TaylorT4'
#### Next select other models to compare to 
PN_names = ['TEOBv4','TEOBResum_ROM','TaylorT1','TaylorT2'] 
### If you want to compare models using sim parameters give the path below
sim_paths = ['../Simulations/BNS/Haas/Haas_Reformat_rh_CceR2090_l2_m2.h5']
wave_read = h5py.File(sim_paths[0], 'r')
lambda_initial = wave_read.attrs['lambda1']
lambda_final = wave_read.attrs['lambda2']
wave_read.close()
delta_tidal = 1.0
tidal_range = np.arange(lambda_initial,lambda_final+1,delta_tidal) 
ip1 = 0
ip2 = 500
match_i = 0
type = 'Approximates'
f_low1 = 120 ### lower frequency bound for long wave 
f_low2 = 160
f_low_long = 30 #### lower frequency for best hybrid
distance = 1.0 ### in Mpcs
inclination = 0
#### Hybrid Params
match_lower = 0.95
num_hybrids = 100
#####
def writeHybrid_h5(path_name_part,metadata,approx,sim_name,h1,h1_ts,h2,h2_ts,delta_t,tidal_1,tidal_2,ip2,fp2):
    global solar_mass_mpc
    global f_low1
    m1 = metadata[0][0]
    m2 = metadata[0][1]
    s1x = metadata[1][0]
    s1y = metadata[1][1]
    s1z = metadata[1][2]
    s2x = metadata[1][3]
    s2y = metadata[1][4]
    s2z = metadata[1][5]
    LNhatx = metadata[2][0]
    LNhaty = metadata[2][1]
    LNhatz = metadata[2][2]
    n_hatx = metadata[3][0]
    n_haty = metadata[3][1]
    n_hatz = metadata[3][2]
    M = 300
    total_mass = m1 + m2
    hybridPN_Num = hy.hybridize(h1,h1_ts,h2,h2_ts,match_i=0,match_f=fp2,delta_t=delta_t,M=M,info=1)
    shift_time = (hybridPN_Num[6],fp2)
    hh_freq = (hybridPN_Num[7],fp2)
    path_name = path_name_part+'fp2_'+str(fp2)+'.h5'
    f_low_M = f_low * (TWOPI * total_mass * MTSUN_SI)
    with h5py.File(path_name,'w') as fd:
        mchirp, eta = pnutils.mass1_mass2_to_mchirp_eta(m1, m2)
        hashtag = hashlib.md5()
        fd.attrs.create('type', 'Hybrid:%s'%type)
        hashtag.update(fd.attrs['type'])
        fd.attrs.create('hashtag', hashtag.digest())
        fd.attrs.create('Read_Group', 'Flynn')
        fd.attrs.create('Format',1)
        fd.attrs.create('Lmax',2)
        fd.attrs.create('approx', approx)
        fd.attrs.create('sim_name', sim_name)
        fd.attrs.create('f_lower_at_1MSUN', f_low_M)
        fd.attrs.create('eta', eta)
        fd.attrs.create('spin1x', s1x)
        fd.attrs.create('spin1y', s1y)
        fd.attrs.create('spin1z', s1z)
        fd.attrs.create('spin2x', s2x)
        fd.attrs.create('spin2y', s2y)
        fd.attrs.create('spin2z', s2z)
        fd.attrs.create('LNhatx', LNhatx)
        fd.attrs.create('LNhaty', LNhaty)
        fd.attrs.create('LNhatz', LNhatz)
        fd.attrs.create('nhatx', n_hatx)
        fd.attrs.create('nhaty', n_haty)
        fd.attrs.create('nhatz', n_hatz)
        fd.attrs.create('grav_mass1', m1)
        fd.attrs.create('grav_mass2', m2)
        fd.attrs.create('lambda1',tidal_1)
        fd.attrs.create('lambda2',tidal_2)
        fd.attrs.create('fp1', len(h1))
        fd.attrs.create('ip2',ip2)
        fd.attrs.create('shift_time',shift_time)
        fd.attrs.create('hybridize_freq',hh_freq)
        gramp = fd.create_group('amp_l2_m2')
        grphase = fd.create_group('phase_l2_m2')
        times = hybridPN_Num[5][0]
        hplus = hybridPN_Num[5][1]
        hcross = hybridPN_Num[5][2]
        massMpc = total_mass*solar_mass_mpc
        hplusMpc  = pycbc.types.TimeSeries(hplus/massMpc, delta_t=delta_t)
        hcrossMpc = pycbc.types.TimeSeries(hcross/massMpc, delta_t=delta_t)
        times_M = times / (MTSUN_SI * total_mass)
        HlmAmp = wfutils.amplitude_from_polarizations(hplusMpc,hcrossMpc).data
        HlmPhase = wfutils.phase_from_polarizations(hplusMpc, hcrossMpc).data
        sAmph = romspline.ReducedOrderSpline(times_M, HlmAmp,rel=True ,verbose=False)
        sPhaseh = romspline.ReducedOrderSpline(times_M, HlmPhase, rel=True,verbose=False)
        sAmph.write(gramp)
        sPhaseh.write(grphase)
        fd.close()
    return(shift_time,hh_freq) 

#### mostly concerned with ligo noise curve comparison. Keep threshold and write to standard h5 format with. h1-h2 L2 norm is good to do. Look at Lindbolm use and abuse of hybrid
#### look at Data formats for numerical relativity 2011 for hybrid metadata.
#### focus on format = 1. problem with ref_f in format=2.
#### 
#start_time = time.time()
if __name__ == '__main__':
    p = mp.Pool(mp.cpu_count())
    for sim_path in sim_paths:
        for approx  in PN_names:
            for tidal_1 in tidal_range:
                for tidal_2 in tidal_range:                        
                    num = h5py.File(sim_path, 'r')
                    m1 = num.attrs['ADM_mass1']
                    m2 = num.attrs['ADM_mass2']
                    barym1 = num.attrs['Baryon_mass1']
                    barym2 = num.attrs['Baryon_mass2']
                    total_mass = m1 + m2
                    s1x = num.attrs['spin1x']
                    s1y = num.attrs['spin1y']
                    s1z = num.attrs['spin1z']
                    s2x = num.attrs['spin2x']
                    s2y = num.attrs['spin2y']
                    s2z = num.attrs['spin2z']
                    LNhatx = num.attrs['LNhatx']
                    LNhaty = num.attrs['LNhaty']
                    LNhatz = num.attrs['LNhatz']
                    n_hatx = num.attrs['nhatx']
                    n_haty = num.attrs['nhaty']
                    n_hatz = num.attrs['nhatx']
                    num.close()
                    metadata = [(m1,m2),(s1x,s1y,s1z,s2x,s2y,s2z),(LNhatx,LNhaty,LNhatz),(n_hatx,n_haty,n_hatz)]
                    PN_model = hy.getPN(approx,m1=m1,m2=m2,f_low=f_low1,distance=distance,delta_t=delta_t,
                                sAx=s1x,sAy=s1y,sAz=s1z,sBx=s2x,sBy=s2y,sBz=s2z,inclination=inclination,tidal1=tidal_1,tidal2=tidal_2)
                    h1 = PN_model[1]
                    h1_ts = PN_model[0]
                    h1_fs = hy.m_frequency_from_polarizations(np.real(h1),np.imag(h1),delta_t)
                    PN_model2 = hy.getPN(sim_name,m1=m1,m2=m2,f_low=f_low2,distance=distance,delta_t=delta_t,
                                sAx=s1x,sAy=s1y,sAz=s1z,sBx=s2x,sBy=s2y,sBz=s2z,inclination=inclination,tidal1=tidal_1,tidal2=tidal_2)
                    best_numz_indices = []
                    best_PNz_indices = []
                    tidal_1 = tidal_1
                    tidal_2 = tidal_2
                    path_name_data = 'HybridAnnex/'+type+'/'+sim_name+'_'+str(ip2)+'/'+approx+'lambda_'+str(tidal_1)+'/'
                    h2 = PN_model2[1] 
                    h2_ts = PN_model2[0] 
                    h2_fs = hy.m_frequency_from_polarizations(np.real(h2),np.imag(h2),delta_t)	
                    num_iteration = np.arange(1,len(h2)+1,1,dtype=np.int64)
                    PN_iteration = np.arange(len(h1)-len(h2)+1,len(h1)+1,1,dtype=np.int64)
                    func_match_h2 = partial(hy.match_generator_num, h1, h2, match_i)
                    func_match_h1 = partial(hy.match_generator_PN, h1, h2, match_i)
                    match_fp1 = p.map(func_match_h1,PN_iteration)
                    match_fp2 = p.map(func_match_h2,num_iteration)       
                    match_fp2 = np.array(match_fp2)
                    match_fp1 = np.array(match_fp1)
                    #### for best_z, give a minimum match threshold. Uses h1 freq and time data.
                    best_numz = np.where(match_fp2>=match_lower)
                    best_numz_indices_temp = np.argwhere(match_fp2>=match_lower)
                    for k in best_numz_indices_temp:
                        best_numz_indices.append(k[0])
                    best_PNz = np.where(match_fp1>=match_lower)
                    best_PNz_indices_temp = np.argwhere(match_fp1>=match_lower)
                    for k in best_PNz_indices_temp:
                        best_PNz_indices.append(k[0])
                    print best_numz_indices
                    freq_range_num = []
                    time_range_num = []
                    freq_range_PN = []
                    time_range_PN= []
                    for n in best_numz_indices:
                        time_range_num.append(h1_ts[n])
                        freq_range_num.append(h1_fs[n])
                    for n in best_PNz_indices:
                        time_range_PN.append(h1_ts[n])
                        freq_range_PN.append(h1_fs[n-1])
                    if not os.path.exists(os.path.dirname(path_name_data)):
                        try:
                            os.makedirs(os.path.dirname(path_name_data))
                        except OSError as exc:  
                            if exc.errno != errno.EEXIST:
                                raise 
                    with h5py.File(path_name_data+'HybridChars.h5','w') as fd:
                        fd.attrs.create('Read_Group', 'Flynn')
                        fd.attrs.create('type', 'Hybrid:%s'%type)
                        fd.attrs.create('approx', approx)
                        fd.attrs.create('sim_name', sim_name)
                        fd.create_dataset("match_PN_fp", data=match_fp1)
                        fd.create_dataset("match_num_fp", data=match_fp2)
                        fd.create_dataset('time_slices',data=h2_ts)
                        fd.create_dataset('freq_slices',data=h2_fs)
                        fd.create_dataset('tidal_rangeA',data=tidal_range)
                        fd.create_dataset('tidal_rangeB',data=tidal_range)
                        fd.create_dataset('bestFreqRange_sim',data=freq_range_num)
                        fd.create_dataset('bestTimeRange_sim',data=time_range_num)
                        fd.create_dataset('bestFreqRange_approx',data=freq_range_PN)
                        fd.create_dataset('bestTimeRange_approx',data=time_range_PN)
                        fd.close()
'''
                    if len(best_numz_indices[25:])>num_hybrids:
                        index_choices = np.random.choice(best_numz_indices[25:],size=num_hybrids)
                    else:
                        index_choices = best_numz_indices[25:]
                    print index_choices
                    path_name_part = path_name_data+sim_name+'_'+approx 
                    func_writeHybrid_h5 = partial(writeHybrid_h5,path_name_part,metadata,approx,sim_name,h1,h1_ts,h2,h2_ts,delta_t,tidal_1,tidal_2,ip2)
                    hh_times_freqs=p.map(func_writeHybrid_h5,index_choices)
                    shift_times = []
                    hh_freqs = []
                    for n in np.arange(0,len(hh_times_freqs),1):
                        shift_times.append(hh_times_freqs[n][0])
                    for n in np.arange(0,len(hh_times_freqs),1):
                        hh_freqs.append(hh_times_freqs[n][1])
                    with h5py.File(path_name_data+'HybridChars.h5','a') as fd:
                        fd.create_dataset('shift_times',data=shift_times)
                        fd.create_dataset('hybridize_freq',data=hh_freqs)
                        fd.close()
                    #### now make a really long hybrid with highest match
                    #### really cheap way of grabbing the highest match while ignoring the beginning 
                    best_index = np.argmax(match_fp2[100:])
                    long_PN = hy.getPN(approx,m1=m1,m2=m2,f_low=f_low_long,distance=distance,delta_t=delta_t,
                                sAx=s1x,sAy=s1y,sAz=s1z,sBx=s2x,sBy=s2y,sBz=s2z,inclination=inclination,tidal1=tidal_1,tidal2=tidal_2)
                    best_hybrid = hy.hybridize(long_PN[1],long_PN[0],h2,h2_ts,match_i=ip2,match_f=best_index,delta_t=delta_t,M=300,info=1)
                    shift_time = best_hybrid[6]
                    hh_freq = best_hybrid[7]
                    long_hybrid_name = path_name_part+'fp2_'+str(best_index)+'_flow'+str(f_low_long)+'.h5'
                    f_low_M = f_low_long * (TWOPI * total_mass * MTSUN_SI)
                    with h5py.File(long_hybrid_name,'w') as fd:
                        mchirp, eta = pnutils.mass1_mass2_to_mchirp_eta(m1, m2)
                        hashtag = hashlib.md5()
                        fd.attrs.create('type', 'Hybrid:%s'%type)
                        hashtag.update(fd.attrs['type'])
                        fd.attrs.create('hashtag', hashtag.digest())
                        fd.attrs.create('Read_Group', 'Flynn')
                        fd.attrs.create('Format',1)
                        fd.attrs.create('Lmax',2)
                        fd.attrs.create('approx', approx)
                        fd.attrs.create('sim_name', sim_name)
                        fd.attrs.create('f_lower_at_1MSUN', f_low_M)
                        fd.attrs.create('eta', eta)
                        fd.attrs.create('spin1x', s1x)
                        fd.attrs.create('spin1y', s1y)
                        fd.attrs.create('spin1z', s1z)
                        fd.attrs.create('spin2x', s2x)
                        fd.attrs.create('spin2y', s2y)
                        fd.attrs.create('spin2z', s2z)
                        fd.attrs.create('LNhatx', LNhatx)
                        fd.attrs.create('LNhaty', LNhaty)
                        fd.attrs.create('LNhatz', LNhatz)
                        fd.attrs.create('nhatx', n_hatx)
                        fd.attrs.create('nhaty', n_haty)
                        fd.attrs.create('nhatz', n_hatz)
                        fd.attrs.create('ADM_mass1', m1)
                        fd.attrs.create('ADM_mass2', m2)
                        fd.attrs.create('Baryon_mass1',barym1)
                        fd.attrs.create('Baryon_mass2',barym2)
                        fd.attrs.create('lambda1',tidal_1)
                        fd.attrs.create('lambda2',tidal_2)
                        fd.attrs.create('PN_fp1', len(h1))
                        fd.attrs.create('Num_ip1',0)
                        fd.attrs.create('shift_time',shift_time)
                        fd.attrs.create('hybridize_freq',hh_freq)
                        gramp = fd.create_group('amp_l2_m2')
                        grphase = fd.create_group('phase_l2_m2')
                        times = best_hybrid[5][0]
                        hplus = best_hybrid[5][1]
                        hcross = best_hybrid[5][2]
                        massMpc = total_mass*solar_mass_mpc
                        hplusMpc  = pycbc.types.TimeSeries(hplus/massMpc, delta_t=delta_t)
                        hcrossMpc = pycbc.types.TimeSeries(hcross/massMpc, delta_t=delta_t)
                        times_M = times / (MTSUN_SI * total_mass)
                        HlmAmp = wfutils.amplitude_from_polarizations(hplusMpc,hcrossMpc).data
                        HlmPhase = wfutils.phase_from_polarizations(hplusMpc, hcrossMpc).data
                        sAmph = romspline.ReducedOrderSpline(times_M, HlmAmp,rel=True ,verbose=False)
                        sPhaseh = romspline.ReducedOrderSpline(times_M, HlmPhase, rel=True,verbose=False)
                        sAmph.write(gramp)
                        sPhaseh.write(grphase)
                        fd.close()
#end_time = (time.time() - start_time)/60.0
#print 'end_time (mins)', end_time
                            print 'saving plots'
                            with PdfPages(path_name_data+'match_v_maxSimTime.pdf') as pdf:
                                plt.plot(h2[0], match_fp2)
                                plt.legend(loc='best')
                                plt.title('Match vs. Max Simulation Time Window')
                                plt.xlabel('Time Slice (seconds)')
                                plt.ylabel('Match')
                                pdf.savefig()
                                plt.clf()
                            with PdfPages(path_name_data+'match_v_freq_range.pdf') as pdf:
                                plt.plot(h2_fs, match_fp2)
                                plt.legend(loc='best')
                                plt.title('Match vs. Mac Frequency Window')
                                plt.xlabel('Frequency slice (Hz)')
                                plt.ylabel('Match')
                                pdf.savefig()
                                plt.clf()
                            with PdfPages(path_name_data+'match_v_shift_time.pdf') as pdf
                                plt.plot(h2_fs, match_fp2)
                                plt.legend(loc='best')
                                plt.title('Match vs. Mac Frequency Window')
                                plt.xlabel('Frequency slice (Hz)')
                                plt.ylabel('Match')
                                pdf.savefig()
                                plt.clf()
'''          
