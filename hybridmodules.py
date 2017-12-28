from functools import partial
import h5py
import cmath
import time
import scipy as sci
import numpy as np
import scipy.signal as sig
import H5Graph
import lal
import re
from scipy import interpolate as inter
import multiprocessing as mp
from multiprocessing import Pool
from pycbc.waveform import get_td_waveform
from pycbc.types.timeseries import TimeSeries

## def _poolinit() is just a profiler. The profiler outputs .out file which shows the time it takes to run functions in the code. Useful 
## for profiling the hybrid characterization script. 
def _poolinit():
	global prof
	prof = cProfile.Profile()
	def finish():
		prof.dump_stats('./Profiles/profile-%s.out' % mp.current_process().pid)
	mp.util.Finalize(None,finish,exitpriority = 1)

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx],idx

def twodimarray_to_2array(array):
    first = []
    second = []
    for i in np.arange(0,len(array),1):
        first.append(array[i][0])
        second.append(array[i][1])
    return (first,second)

def fourdimarray_to_4array(array):
    first = []
    second = []
    third = []
    fourth = []
    for i in np.arange(0,len(array),1):
        first.append(array[i][0])
        second.append(array[i][1])
        third.append(array[i][2])
        forth.append(array[i][3])
    return (first,second)

def rhOverM_to_SI(polarization,total_mass):
        solar_mass_mpc = 1.0/(1.0e6 * lal.PC_SI/lal.MRSUN_SI) 
        h_conversion = total_mass*solar_mass_mpc
        return polarization*h_conversion

def tOverM_to_SI(times,total_mass):
        t_conversion = total_mass*lal.MTSUN_SI
        return times*t_conversion

def SI_to_rhOverM(polarization,total_mass):
	solar_mass_mpc = lal.MRSUN_SI/(1e6*lal.PC_SI)
        h_conversion = solar_mass_mpc/total_mass
        return polarization*h_conversion

def SI_to_rOverM(times,total_mass):
        t_conversion = total_mass*lal.MTSUN_SI
        return times/t_conversion

def getPN(name,m1=10.0,m2=10.0,f_low=50.0,distance=1,delta_t=1.0/4096.0,sAx=0,sAy=0,sAz=0,sBx=0,sBy=0,sBz=0,inclination=0,tidal1=0,tidal2=0):
        Sph_Harm = 0.6307831305
        hp, hc = get_td_waveform(approximant = name, mass1=m1,
                                                mass2=m2,
                                                f_lower=f_low,
                                                distance= distance,
                                                delta_t= delta_t,
                                                spin1x = sAx,
                                                spin1y = sAy,
                                                spin1z = sAz,
                                                spin2x = sBx,
                                                spin2y = sBy,
                                                spin2z = sBz,
                                                lambda1 = tidal1,
                                                lambda2 = tidal2,
                                                inclination= inclination)

        new_hp = np.array(hp)
        new_hc = np.array(hc)
        PN_wave = new_hp + new_hc*1j
        times = np.array(hp.sample_times)
	shift_times = times - times[0]
	return (shift_times,PN_wave)

def m_phase_from_polarizations(hp,hc,remove_start_phase=True):
	p_wrapped = np.arctan2(hp,hc)
	p = np.unwrap(p_wrapped)
	if remove_start_phase:
		p += -p[0]	
	return np.abs(p)

def m_frequency_from_polarizations(hp,hc,delta_t):
	phase = m_phase_from_polarizations(hp,hc)
	freq = np.diff(phase)/(2.0*np.pi*delta_t)
	return freq
	
def matchfft(h1,h2):
	z_fft = sci.signal.fftconvolve(np.conj(h1),h2[::-1])
	abs_z_fft = np.abs(z_fft)
	w = np.argmax(abs_z_fft) - len(h2) + 1
	delta_w =  w + len(h2)
	h2_norm = np.linalg.norm(h2)
	h1_norm = np.linalg.norm(h1[w:delta_w])
	norm_z = abs_z_fft/(h1_norm*h2_norm)
	return np.amax(norm_z)

def match_generator_num(h1,h2,match_i,fp2):
	try:	
        	match_f = fp2
        	h2_seg = h2[match_i:match_f]
		z_fft = sci.signal.fftconvolve(h1,np.conj(h2_seg[::-1]))
        	abs_z_fft = np.abs(z_fft)
        	w = np.argmax(abs_z_fft) - len(h2_seg) + 1
        	delta_w =  w + len(h2_seg)
        	h2_norm = np.linalg.norm(h2_seg)
       		h1_norm = np.linalg.norm(h1[w:delta_w])
		norm_z = abs_z_fft/(h1_norm*h2_norm)
        	return np.amax(norm_z)
	except RuntimeWarning:
		raise
##### Needs debugging
def match_generator_PN(h1,h2,match_i,fp1):
        if len(h1) > len(h2):
		match_f = fp1
                h1_seg = h1[match_i:match_f]
                z_fft = sci.signal.fftconvolve(h1_seg,np.conj(h2[::-1]))
                abs_z_fft = np.abs(z_fft)
                w = np.argmax(abs_z_fft) - len(h1_seg) + 1
                delta_w =  w + len(h1_seg)
                h1_norm = np.linalg.norm(h1_seg)
                h2_norm = np.linalg.norm(h2[w:delta_w])
		norm_z = abs_z_fft/(h1_norm*h2_norm)
                return np.amax(norm_z)
        elif  len(h1) <= len(h2):
                match_f = fp1
                h2_seg = h2[match_i:match_f]
                z_fft = sci.signal.fftconvolve(h2_seg,np.conj(h1[::-1]))
                abs_z_fft = np.abs(z_fft)
                w = np.argmax(abs_z_fft) - len(h1) + 1
                delta_w =  w + len(h1)
                h2_norm = np.linalg.norm(h2_seg)
                h1_norm = np.linalg.norm(h1[w:delta_w])
                norm_z = abs_z_fft/(h1_norm*h2_norm)
                return np.amax(norm_z)
        else:
                return 'error'

def corrintegral(h1,h2,initial,f):
	match_i = initial
        match_f = f
        z = sci.correlate(h1,h2[match_i:match_f],mode='full')
        abs_z = np.abs(z)
        w = np.argmax(abs_z) - len(h2[match_i:match_f]) + 1
        delta_w = w + len(h2[match_i:match_f])
        h2p_norm = np.linalg.norm(h2[match_i:match_f])
	h1p_norm = np.linalg.norm(h1[w:delta_w])
        norm_z = abs_z/(h1p_norm*h1p_norm)
        return  np.amax(norm_z),w
### match function takes in two waveforms with corresponding parameters and can either perform a simple match using function build = 0 (with output 
### (w,delta_w,np.amax(norm_z),phi,h2_phase_shift) where w is the match index, delta_w, the match number, the phase angle, and the corresponding phase shift for h2 respectively) or 
### using build = 1 constructs a full hybrid with windowing length M (an integer). Windowing function used: hann function
def hybridize(h1,h1_ts,h2,h2_ts,match_i,match_f,delta_t=1/4096.0,M=200,info=0):	
	h2_seg = h2[match_i:match_f]
        z = sci.signal.fftconvolve(h1,np.conj(h2_seg[::-1]))
	abs_z = np.abs(z)
	w = np.argmax(abs_z) - len(h2_seg) + 1
	delta_w = w + len(h2_seg)
	h2_norm = np.linalg.norm(h2_seg)
	h1_norm = np.linalg.norm(h1[w:delta_w])
	norm_z = abs_z/(h1_norm*h2_norm)
	phi = np.angle(z[np.argmax(abs_z)])
	h2_phase_shift = np.exp(1j*phi)*h2
	shift_time  = (w - match_i)*delta_t
        h2_tc = h2_ts - h2_ts[0] + shift_time
	off_len = (M-1)/2 + 1
	on_len = (M+1)/2
	window = sig.hann(M)
	##Initialize off and on arrays
	off_hp = np.zeros(off_len)
	on_hp = np.zeros(on_len)
	off_hc = np.zeros(off_len)
	on_hc = np.zeros(on_len)
	##Bounds for windowing functions
	lb= w
	mid=off_len + w
	ub = M-1 + w
	##multiply each off and on section by appropriate window section
	for i in range(on_len):
		on_hp[i] = np.real(h2_phase_shift[match_i+i])*window[i]
	for i in range(off_len):
		off_hp[i] = np.real(h1[w+i])*window[i+off_len-1]
	for i in range(on_len):
		on_hc[i] = np.imag(h2_phase_shift[match_i+i])*window[i]
	for i in range(off_len):
		off_hc[i] = np.imag(h1[w+i]*window[i+off_len-1])
	 ##Next add the on and off sections together
	mix_hp = on_hp + off_hp
	mix_hc = on_hc + off_hc
	h1_hp_split = np.real(h1[:w])
	h1_hc_split = np.imag(h1[:w])	
	h1_ts_split = h1_ts[:w]
	hybrid_t = np.concatenate((np.real(h1_ts_split),np.real(h2_tc[match_i:])), axis =0)
	hybrid_hp = np.concatenate((h1_hp_split,mix_hp,np.real(h2_phase_shift[match_i+off_len:])),axis = 0)
	hybrid_hc = np.concatenate((h1_hc_split,mix_hc,np.imag(h2_phase_shift[match_i+off_len:])),axis =0)
	hybrid = (hybrid_t, hybrid_hp, hybrid_hc)
	freq = m_frequency_from_polarizations(hybrid_hp,hybrid_hc,delta_t=delta_t)
        hh_freq = freq[w]
        if info == 0:
		return hybrid
	if info == 1:
		return(np.max(norm_z),w,phi,h2_phase_shift,h2_tc,hybrid,h2_tc[0],hh_freq)
	else: 
		return 'use info = 0 or 1'

def getFormatSXSData(filepath,total_m,delta_t=1.0/4096.0,l=2,m=2,N=4):
        num = h5py.File(filepath, 'r')
### We only care about 2-2 mode for now.
        harmonics = 'Y_l%d_m%d.dat' %(l,m)
        order = 'Extrapolated_N%d.dir'%N
        ht = num[order][harmonics][:,0][500:]
        hre = num[order][harmonics][:,1][500:]
        him = num[order][harmonics][:,2][500:]
        ht_SI = tOverM_to_SI(ht,total_m)
        hre_SI = rhOverM_to_SI(hre,total_m)
        him_SI = rhOverM_to_SI(him,total_m)
        sim_name = re.search('/BBH(.+?)/L(.+?)/',filepath).group(0) 
        interpo_hre = sci.interpolate.interp1d(ht_SI,hre_SI, kind = 'linear')
        interpo_him = sci.interpolate.interp1d(ht_SI,him_SI, kind = 'linear')
##### interpolate the numerical hp and hc with PN timeseries
        hts = np.arange(ht_SI[0],ht_SI[-1],delta_t)
        #num_t_zeros = np.concatenate((num_ts,np.zeros(np.absolute(len(PN_tc)-len(num_t)))),axis = 0)
        new_hre = interpo_hre(hts)
        new_him = interpo_him(hts)
        num_wave = new_hre - 1j*new_him
#### Cast waves into complex form and take fft of num_wave
        num.close()
        return (sim_name,hts,num_wave)
'''
def iter_hybridize_write(h1,h1_ts,h2,h2_ts,match_i,delta_t,M):
    hybridPN_Num = hybridize(h1,h1_ts,h2,h2_ts,match_i=match_i,match_f=l,delta_t=delta_t,M=300,info=1)
    shift_times.append(hybridPN_Num[6])
    hh_freqs.append(h1_fs[hybridPN_Num[1]])
    h5file= path_name_data+name+'_'+sim_name+'.h5'
    f_low_M = f_low * (lal.TWOPI * total_mass * lal.MTSUN_SI)
    with h5py.File(path_name_data+sim_hybrid_name+'_'+approx+'fp2_'+str(l)+'.h5','w') as fd:
        mchirp, eta = pnutils.mass1_mass2_to_mchirp_eta(m_1, m_2)
        hashtag = hashlib.md5()
        fd.attrs.create('type', 'Hybrid:%s'%type)
        hashtag.update(fd.attrs['type'])
        fd.attrs.create('hashtag', hashtag.digest())
        fd.attrs.create('Read_Group', 'Flynn')
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
        fd.attrs.create('coa_phase', coa_phase)
        fd.attrs.create('mass1', m_1)
        fd.attrs.create('mass2', m_2)
        fd.attrs.create('lambda1',tidal_1)
        fd.attrs.create('lambda2',tidal_2)
        fd.attrs.create('PN_fp2', len(h1))
        fd.attrs.create('Num_begin_window_index',0 )
        #fd.attrs.create('check_match',hybridPN_Num[0])
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
    with h5py.File(path_name_data+'HybridChars.h5','a') as fd:
        fd.create_dataset('shift_times',data=shift_times)
        fd.create_dataset('hybridize_freq',data=hh_freqs)
        fd.close()

#### writes a given hybrid to a format 1 h5 file and writes it to disk 
def writeHybridtoSplineH5():
    with h5py.File(path_name_data+name+'_'+sim_name,'w') as fd:
                                mchirp, eta = pnutils.mass1_mass2_to_mchirp_eta(mass1, mass2)
                                hashtag = hashlib.md5()
                                hashtag.update(fd.attrs['name'])
                                fd.attrs.create('hashtag', hashtag.digest())
                                fd.attrs.create('Read_Group', 'Flynn')
                                fd.attrs.create('name', 'Hybrid:B0:%s'%simname)
                                fd.attrs.create('f_lower_at_1MSUN', f_low_M)
                                fd.attrs.create('eta', eta)
                                fd.attrs.create('Name of Simulation', num_waves[0])
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
                                fd.attrs.create('coa_phase', coa_phase)
                                fd.attrs.create('mass1', m_1)
                                fd.attrs.create('mass2', m_2)            
                                gramp = fd.create_group('amp_l2_m2')
                                grphase = fd.create_group('phase_l2_m2')
                                times = hybridPN_Num[0]
                                native_delta_t = delta_t
                                hplus = hybridPN_Num[1]
                                hcross = hybridPN_Num[2]
                                massMpc = total_mass*solar_mass_mpc
                                hplusMpc  = pycbc.types.TimeSeries(hplus/massMpc, delta_t=delta_t)
                                hcrossMpc = pycbc.types.TimeSeries(hcross/massMpc, delta_t=delta_t)
                                times_M = times / (lal.MTSUN_SI * total_mass)
                                HlmAmp   = wfutils.amplitude_from_polarizations(hplusMpc,
                                        hcrossMpc).data
                                HlmPhase = wfutils.phase_from_polarizations(hplusMpc, hcrossMpc).data
                        #       if l!=2 or abs(m)!=2:
                        #           HlmAmp = np.zeros(len(HlmAmp))
                        #           HlmPhase = np.zeros(len(HlmPhase))

                                sAmph = romspline.ReducedOrderSpline(times_M, HlmAmp,rel=True ,verbose=False)
                                sPhaseh = romspline.ReducedOrderSpline(times_M, HlmPhase, rel=True,verbose=False)
                                sAmph.write(gramp)
                                sPhaseh.write(grphase)

'''
def delta_h(h1,h2):
        h1 = np.array(h1)
        h2 = np.array(h2)
        if len(h1) > len(h2):
                h2 = np.append(h2,np.zeros(np.abs(len(h2)-len(h1))))
        if len(h1) < len(h2):
                h1 = np.append(h1,np.zeros(np.abs(len(h2)-len(h1))))
	norm_diff = np.divide(np.linalg.norm(np.subtract(h1,h2)),np.linalg.norm(h2))
	return norm_diff 
'''		

########################################################################################
## Bandpass filter will only vary the high and low cutoffs at the same time. The filter cutoff
#  currently corresponds to where the middle of the filter is.          

def match_bpfil(h1,h2,order,sample_rate,cutoff,center=250):
        h1p = np.real(h1)
        h2p = np.real(h2)
        nyq = 0.5*sample_rate
        high = (center + cutoff)/nyq
        low = (center - cutoff)/nyq
        b, a = sig.butter(order,[low,high],btype='bandpass', analog=False)
        #w,h = sig.freqz(b,a) 
        h2p_filter = sig.lfilter(b, a, h2p)
        z_fft = sci.signal.fftconvolve(np.conj(h1p),h2p_filter[::-1])
        abs_z_fft = np.abs(z_fft)
        w = np.argmax(abs_z_fft) - len(h2p) + 1
        delta_w =  w + len(h2p)
        h2p_norm = np.linalg.norm(h2p)
        h1p_norm = np.linalg.norm(h1p[w:delta_w])
        norm_z = abs_z_fft/(h1p_norm*h2p_norm)
        return np.amax(norm_z)

def match_lpfil(h1,h2,order,sample_rate,cutoff):
        h1p = np.real(h1)
        h2p = np.real(h2)
        nyq = 0.5*sample_rate
        normal_cutoff = cutoff/nyq
        b,a = sig.butter(order,normal_cutoff,btype='lowpass', analog=False)
        #w,h = sig.freqz(b,a)
        h1p_fil = sig.lfilter(b, a, h1p)
        z_fft = sci.signal.fftconvolve(np.conj(h1p_fil),h2p[::-1])
        abs_z_fft = np.abs(z_fft)
        w = np.argmax(abs_z_fft) - len(h2p) + 1
        delta_w =  w + len(h2p)
        h2p_norm = np.linalg.norm(h2p)
        h1p_norm = np.linalg.norm(h1p[w:delta_w])
        norm_z = abs_z_fft/(h1p_norm*h2p_norm)
        return np.amax(norm_z)
## cond wave argumenet is the wave to be filtered
def match_hpfil(h1,h2,order,sample_rate,cutoff):
        h1p = np.real(h1)
        h2p = np.real(h2)
        nyq = 0.5*sample_rate
        normal_cutoff = cutoff/nyq
        b, a = sig.butter(order,normal_cutoff,btype='highpass', analog=False)
        #w,h = sig.freqz(b,a) 
        if len(h1) >= len(h2):
                h2p_filter = sig.lfilter(b, a, h2p)
                z_fft = sci.signal.fftconvolve(np.conj(h1p),h2p_filter[::-1])
                abs_z_fft = np.abs(z_fft)
                w = np.argmax(abs_z_fft) - len(h2p) + 1
                delta_w =  w + len(h2p)
                h2p_norm = np.linalg.norm(h2p)
                h1p_norm = np.linalg.norm(h1p[w:delta_w])
                norm_z = abs_z_fft/(h1p_norm*h2p_norm)
                return np.amax(norm_z)
        elif len(h1) < len(h2):
                h2p_filter = sig.lfilter(b, a, h2p)
                z_fft = sci.signal.fftconvolve(np.conj(h2p_filter),h1p[::-1])
                abs_z_fft = np.abs(z_fft)
                w = np.argmax(abs_z_fft) - len(h2p) + 1
                delta_w =  w + len(h2p)
                h2p_norm = np.linalg.norm(h2p)
                h1p_norm = np.linalg.norm(h1p[w:delta_w])
                norm_z = abs_z_fft/(h1p_norm*h2p_norm)
                return np.amax(norm_z)
'''
