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
import romspline
solar_mass_mpc = lal.MRSUN_SI / (1e6*lal.PC_SI)
Sph_Harm = 0.6307831305 ## 2-2 spherical harmoniic at theta = 0 phi = 90 
sample_rate = 4096.0*8
delta_t = 1.0/sample_rate
### PN_names must be a string of a PN from lalsim
PN_names = ['TaylorT2','TaylorT4']
###sim_paths is the array containing all simulations you want to make hybrids of. glob.glob is very useful here.
sim_paths = ['../Simulations/BNS/Haas/Haas_Reformat_rh_CceR2090_l2_m2.h5']
### import lambda values from simulation. here a format is assumed to the same across all sim_paths
wave_read = h5py.File(sim_paths[0], 'r')
lambda_initial1 = wave_read.attrs['lambda1']
lambda_final1 = wave_read.attrs['lambda1']
lambda_initial2 = wave_read.attrs['lambda2']
lambda_final2 = wave_read.attrs['lambda2']
wave_read.close()

### set range of lambda value for PN's to compare to simulation
delta_tidal = 1.0
tidal_range1 = np.arange(lambda_initial1,lambda_final1+1,delta_tidal) 
tidal_range2 = np.arange(lambda_initial2,lambda_final2+1,delta_tidal)
### set initial simulation starting iteration for match maximization. see documentation at https://github.com/eflynn67/HybridTools/wiki for more detail. 
ip1 = 0
final_start_index = 600
delta_start_index = 50
starting_index = np.arange(0,final_start_index,delta_start_index)

###beginning index for hybrid match window 
match_i = 0

type = 'BNS'
### PN params
### Note: REMEMBER TO PICK A MASS under the __main__
sim_name = 'Haas'
f_low = 80 ### lower frequency bound for hybrid characteristics 
f_low_long = 30 #### lower frequency for best hybrid
distance = 1.0 ### in Mpcs
inclination = 0
#### Hybrid Params
match_lower = 0.95
num_hybrids = 100
#####
#writeHybrid_h5 is just a function used for writting short hybrids on multiple processors using python's multiprocessing
def writeHybrid_h5(path_name_part,metadata,approx,sim_name,h1,h1_ts,h2,h2_ts,delta_t,tidal_1,tidal_2,fp2):
    global solar_mass_mpc
    global f_low
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
    f_low_M = f_low * (lal.TWOPI * total_mass * lal.MTSUN_SI)
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
        times_M = times / (lal.MTSUN_SI * total_mass)
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
            for tidal_1 in tidal_range1:
                for tidal_2 in tidal_range2:
                    # First import parameters needed for Pycbc get_td_waveform() a numerical format is assumed. See https://github.com/eflynn67/HybridTools/wiki      
                    num = h5py.File(sim_path, 'r')
                    grav_m1 = num.attrs['grav_mass1']
                    grav_m2 = num.attrs['grav_mass2']  
                    ADM_m1 = num.attrs['ADM_mass1']
                    ADM_m2 = num.attrs['ADM_mass2']
                    barym1 = num.attrs['Baryon_mass1']
                    barym2 = num.attrs['Baryon_mass2']
                    #### Select which mass to use to make the Pycbc waveforms
                    m1 =  ADM_m1 
                    m2 = ADM_m2 
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
                    PN_model = hy.getPN(approx,m1=m1,m2=m2,f_low=f_low,distance=distance,delta_t=delta_t,
                                sAx=s1x,sAy=s1y,sAz=s1z,sBx=s2x,sBy=s2y,sBz=s2z,inclination=inclination,tidal1=tidal_1,tidal2=tidal_2)
                    h1 = PN_model[1]
                    h1_ts = PN_model[0]
                    h1_fs = hy.m_frequency_from_polarizations(np.real(h1),np.imag(h1),delta_t)
                    h2p,h2c = get_td_waveform(approximant='NR_hdf5',numrel_data=sim_path,mass1=m1,mass2=m2,
                                                                      spin1z=s1z,spin1x=s1x,spin1y=s1y,
                                                                      spin2z=s2z,spin2x=s2x,spin2y=s2y,
                                                                      delta_t=delta_t,distance=distance,
                                                                      f_lower=f_low,
                                                                      inclination=inclination)
                    match_funcs = []
                    matches_area = []
                    tidal_1 = tidal_1
                    tidal_2 = tidal_2
                    h2_full = np.array(h2p[:] + 1j*h2c[:])
                    num_times = np.array(h2p.sample_times[:]) 
                    h2_ts_full = num_times - num_times[0]
                    full_freq = hy.m_frequency_from_polarizations(np.real(h2_full),np.imag(h2_full),delta_t)
                    for ip2 in starting_index:
                        best_numz_indices = []
                        best_PNz_indices = []
                        path_name_data = 'HybridAnnex/'+type+'/'+sim_name+'/'+approx+'/'+approx+'ip2_'+str(ip2)+'lambda_'+str(tidal_1)+'/'
                        h2 = h2_full[ip2:]
                        h2_ts = h2_ts_full[ip2:]
                        h2_fs = full_freq[ip2:]
                        starting_freq = np.mean(h2_fs[1:11])
                        time_two_osc = 2.0/starting_freq 
                        time_shift = h2_ts - h2_ts[0]
                        two_osc_index = hy.find_nearest(time_shift,time_two_osc)
                        num_iteration = np.arange(1,len(h2)+1,1,dtype=np.int64)
                        PN_iteration = np.arange(len(h1)-len(h2)+1,len(h1)+1,1,dtype=np.int64)
                        func_match_h2 = partial(hy.match_generator_num, h1, h2, match_i)
                        func_match_h1 = partial(hy.match_generator_PN, h1, h2, match_i)
                        match_fp1 = p.map(func_match_h1,PN_iteration)
                        match_fp2 = p.map(func_match_h2,num_iteration)       
                        match_fp2 = np.array(match_fp2)
                        match_fp1 = np.array(match_fp1)
                        match_area_fp2 = sci.integrate.trapz(match_fp2,h2_ts) 
                        matches_area.append(match_area_fp2)
                        match_funcs.append(match_fp2[two_osc_index[1]:])
                        #### for best_z, give a minimum match threshold. Uses h1 freq and time data
                        best_numz = np.where(match_fp2[two_osc_index[1]:]>=match_lower)
                        best_numz_indices_temp = np.argwhere(match_fp2[two_osc_index[1]:]>=match_lower)
                        for k in best_numz_indices_temp:
                            best_numz_indices.append(k[0])
                        best_PNz = np.where(match_fp1>=match_lower)
                        best_PNz_indices_temp = np.argwhere(match_fp1>=match_lower)
                        for k in best_PNz_indices_temp:
                            best_PNz_indices.append(k[0])
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
                            fd.attrs.create('num_start_freq',starting_freq)
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
                        if len(best_numz_indices[two_osc_index[1]:])>num_hybrids:
                            index_choices = np.random.choice(best_numz_indices[two_osc_index[1]:],size=num_hybrids)
                        else:
                            index_choices = best_numz_indices[two_osc_index[1]:]
                        path_name_part = path_name_data+sim_name+'_'+approx 
                        func_writeHybrid_h5 = partial(writeHybrid_h5,path_name_part,metadata,approx,sim_name,h1,h1_ts,h2,h2_ts,delta_t,tidal_1,tidal_2)
                        hh_times_freqs=p.map(func_writeHybrid_h5,index_choices)
                        shift_times = []
                        hh_freqs = []
                        for n in np.arange(0,len(hh_times_freqs),1):
                            shift_times.append(hh_times_freqs[n][0])
                        for n in np.arange(0,len(hh_times_freqs),1):
                            print hh_times_freqs[n][1]
                            hh_freqs.append(hh_times_freqs[n][1])
                        with h5py.File(path_name_data+'HybridChars.h5','a') as fd:
                            fd.create_dataset('shift_times',data=shift_times)
                            fd.create_dataset('hybridize_freq',data=hh_freqs)
                            fd.close()
#### now make a really long hybrid with highest match
                    best_ip2 = starting_index[np.argmax(matches_area)] 
                    best_fp2 = np.argmax(np.array(match_funcs[np.argmax(matches_area)]))   
                    long_PN = hy.getPN(approx,m1=m1,m2=m2,f_low=f_low_long,distance=distance,delta_t=delta_t,
                                sAx=s1x,sAy=s1y,sAz=s1z,sBx=s2x,sBy=s2y,sBz=s2z,inclination=inclination,tidal1=tidal_1,tidal2=tidal_2)
                    best_hybrid = hy.hybridize(long_PN[1],long_PN[0],h2[best_ip2:],h2_ts[best_ip2:],match_i=0,match_f=best_fp2,delta_t=delta_t,M=300,info=1)
                    shift_time = best_hybrid[6]
                    hh_freq = best_hybrid[7]
                    long_hybrid_name = 'HybridAnnex/'+type+'/'+sim_name+'/'+approx+'/'+sim_name+'_'+approx+'ip2_'+str(best_ip2)+'fp2_'+str(best_fp2)+'_flow'+str(f_low_long)+'.h5'
                    f_low_M = f_low_long * (lal.TWOPI * total_mass * lal.MTSUN_SI)
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
                        fd.attrs.create('grav_mass1', grav_m1)
                        fd.attrs.create('grav_mass2', grav_m2)
                        fd.attrs.create('ADM_mass1', m1)
                        fd.attrs.create('ADM_mass2', m2)
                        fd.attrs.create('Baryon_mass1',barym1)
                        fd.attrs.create('Baryon_mass2',barym2)
                        fd.attrs.create('lambda1',tidal_1)
                        fd.attrs.create('lambda2',tidal_2)
                        fd.attrs.create('PN_fp1', len(h1))
                        fd.attrs.create('num_ip2',best_ip2)
                        fd.attrs.create('num_start_freq',full_freq[best_ip2])
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
                        times_M = times / (lal.MTSUN_SI * total_mass)
                        HlmAmp = wfutils.amplitude_from_polarizations(hplusMpc,hcrossMpc).data
                        HlmPhase = wfutils.phase_from_polarizations(hplusMpc, hcrossMpc).data
                        sAmph = romspline.ReducedOrderSpline(times_M, HlmAmp,rel=True ,verbose=False)
                        sPhaseh = romspline.ReducedOrderSpline(times_M, HlmPhase, rel=True,verbose=False)
                        sAmph.write(gramp)
                        sPhaseh.write(grphase)
                        fd.close()
'''
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
