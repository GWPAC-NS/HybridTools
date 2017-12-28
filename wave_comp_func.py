### assume the following conventions
### function will take arrays of all the parameters. if single values want to be used either give it an array containing that single point 
### or give it a single point and the function will detect it as a single point and will put it in to an array.
### remove degenercies? how do i know that the function is symmetric with respect to mass, spins, and lambdas between models.  
### single processor should be ok since we assume each array is small
### spins are input as (sx,sy,sz) or arrays of these tuples.
### lambda1 and lambda2 can be scalar or array.
### mass1 and mass2 are input as scalar or array,
### names is the array containing all the names of the approx you want to compare.
### psd is assumed to be in pycbc FrequencySeries
def wave_comp(names,,mass1,mass2,spin1,spin2,lambda1,lambda2,psd='None',f_low=70,distance=1,inclination=0,delta_t=1/4096.0):
    td_names = []
    fd_names = []
    mass1_array = np.array(mass1,ndmin=1)
    mass2_array = np.array(mass2,ndmin=1)
    spin1_array = np.array(spin1,ndmin=1)
    spin2_array = np.array(spin2,ndmin=1) 
    lambda1_array = np.array(lambda1,ndmin=1)
    lambda2_array = np.array(lambda2,ndmin=1)
    for n in names:
        if n==print_td_waveforms():
            td_names.append(n)
        elif n==print_fd_waveforms():
            fd_names.append(n)
    if len(td_names) > 0 and len(fd_names) == 0:
        for i,name1 in enumerate(td_names): 
            shift_index = np,arange(i,len(td_names)-1,1) 
            for j in shift_index:
                    name2 = td_names[j]
                for m1_index,mass1 in enumerate(mass1_array):
                    for m2_index, mass2 in enumerate(mass2_array):
                        for spin1_index,spin1 in enumerate(spins1_array):
                            for spin2_index,spin2 in enumerate(spins2_array):
                                for lambda1_index,lambda1 in enumerate(lambda1_array):
                                    for lambda2_index,lambda2 in enumerate(lambda2_array):
                                         h1p,h1c = get_td_waveform(approximant=name1,mass1=mass1,
                                                                  mass2=mass2,
                                                                  spin1z=spin1[2],spin1x=spin1[0],spin1y=spin1[1],
                                                                  spin2z=spin2[2],spin2x=spin2[0],spin2y=spin2[1],
                                                                  lambda1=lambda1,lambda2=lambda2,delta_t=delta_t,distance=distance,
                                                                  f_lower=f_low,
                                                                  inclination=inclination)
                                        h2p,h2c = get_td_waveform(approximant=name2,mass1=mass1,
                                                                  mass2=mass2,
                                                                  spin1z=spin,spin1x=s1x,spin1y=s1y,
                                                                  spin2z=s2z,spin2x=s2x,spin2y=s2y,
                                                                  lambda1=lambda1,lambda2=lambda2,delta_t=delta_t,distance=distance,
                                                                  f_lower=f_low,
                                                                  inclination=inclination)
                                        t_len = max(len(h1p), len(h2p))
                                        h1p.resize(t_len)
                                        h2p.resize(t_len)
                                        delta_f = 1.0 /h2p.duration
                                        f_len = t_len/2 + 1
                                        f_cutoff = 30.0
                                        h1p = np.array(h1p)
                                        h2p = np.array(h2p)
                                        h1p_tilde = np.fft.fft(h1p,norm=None)/sample_rate
                                        h2p_tilde = np.fft.fft(h2p,norm=None)/sample_rate
                                        h1p_tilde_new = FrequencySeries(h1p_tilde[:len(h1p)/2], delta_f=delta_f)
                                        h2p_tilde_new = FrequencySeries(h2p_tilde[:len(h2p)/2], delta_f=delta_f)
                                        h1p_tilde_new.resize(f_len)
                                        h2p_tilde_new.resize(f_len)
                                        match_plus, index1 = match(h1p_tilde_new,h2p_tilde_new,psd=psd,low_frequency_cutoff=f_cutoff)
                                        return(match_plus)
    elif len(td_names) > 0 and len(fd_names) > 0:
        if len(td_names) > len(fd_names):
            longest=td_names
        else:
            longest=fd_names
        for i,name1 in enumerate(longest): 
            shift_index = np,arange(i,len(names)-1,1) 
            for j in shift_index:
                    name2 = [j]
                for m1_index,mass1 in enumerate(mass1_array):
                    for m2_index, mass2 in enumerate(mass2_array):
                        for spin1_index,spin1 in enumerate(spins1_array):
                            for spin2_index,spin2 in enumerate(spins2_array):
                                for lambda1_index,lambda1 in enumerate(lambda1_array):
                                    for lambda2_index,lambda2 in enumerate(lambda2_array):
                                        htp,htc = get_td_waveform(approximant=name1,mass1=m1,
                                                                  mass2=m2,
                                                                  spin1z=s1z,spin1x=s1x,spin1y=s1y,
                                                                  spin2z=s2z,spin2x=s2x,spin2y=s2y,
                                                                  delta_t=delta_t,distance=distance,
                                                                  f_lower=f_lower_hh1,
                                                                  inclination=inclination)
                                        hfp,hfc = get_fd_waveform(approximant=name2,mass1=m1,
                                                                  mass2=m2,
                                                                  spin1z=s1z,spin1x=s1x,spin1y=s1y,
                                                                  spin2z=s2z,spin2x=s2x,spin2y=s2y,
                                                                  delta_t=delta_t,distance=distance,
                                                                  f_lower=f_lower_hh1,
                                                                  inclination=inclination)
                                        t_len = max(len(h1p), len(h2p))
                                        htp.resize(t_len)
                                        htp.resize(t_len)
                                        delta_f = 1.0 /h2p.duration
                                        f_len = t_len/2 + 1
                                        f_cutoff = 30.0
                                        h1p = np.array(h1p)
                                        h2p = np.array(h2p)
                                        h1p_tilde = np.fft.fft(h1p,norm=None)/sample_rate
                                        h2p_tilde = np.fft.fft(h2p,norm=None)/sample_rate
                                        h1p_tilde_new = FrequencySeries(h1p_tilde[:len(h1p)/2], delta_f=delta_f)
                                        h2p_tilde_new = FrequencySeries(h2p_tilde[:len(h2p)/2], delta_f=delta_f)
                                        h1p_tilde_new.resize(f_len)
                                        h2p_tilde_new.resize(f_len)
                                        match_plus, i = match(h1p_tilde_new,h2p_tilde_new,psd=psd,low_frequency_cutoff=f_cutoff)
    elif len(td_names) == 0 and len(fd_names) > 0:
        for i,name1 in enumerate(names): 
            shift_index = np,arange(i,len(names)-1,1) 
            for j in shift_index:
                    name2 = names[j]
                for m1_index,mass1 in enumerate(mass1_array):
                    for m2_index, mass2 in enumerate(mass2_array):
                        for spin1_index,spin1 in enumerate(spins1_array):
                            for spin2_index,spin2 in enumerate(spins2_array):
                                for lambda1_index,lambda1 in enumerate(lambda1_array):
                                    for lambda2_index,lambda2 in enumerate(lambda2_array):
                                        h1p,h1c = get_td_waveform(approximant=name1,mass1=m1,
                                                                  mass2=m2,
                                                                  spin1z=s1z,spin1x=s1x,spin1y=s1y,
                                                                  spin2z=s2z,spin2x=s2x,spin2y=s2y,
                                                                  delta_t=delta_t,distance=distance,
                                                                  f_lower=f_lower_hh1,
                                                                  inclination=inclination)
                                        h2p,h2c = get_td_waveform(approximant=name2,mass1=m1,
                                                                  mass2=m2,
                                                                  spin1z=s1z,spin1x=s1x,spin1y=s1y,
                                                                  spin2z=s2z,spin2x=s2x,spin2y=s2y,
                                                                  delta_t=delta_t,distance=distance,
                                                                  f_lower=f_lower_hh1,
                                                                  inclination=inclination)
                                        h1p_tilde = np.fft.fft(h1p,norm=None)/sample_rate
                                        h2p_tilde = np.fft.fft(h2p,norm=None)/sample_rate
                                        h1p_tilde_new = FrequencySeries(h1p_tilde[:len(h1p)/2], delta_f=delta_f)
                                        h2p_tilde_new = FrequencySeries(h2p_tilde[:len(h2p)/2], delta_f=delta_f)
                                        h1p_tilde_new.resize(f_len)
                                        h2p_tilde_new.resize(f_len)
                                        match_plus, i = match(h1p_tilde_new,h2p_tilde_new,psd=psd,low_frequency_cutoff=f_cutoff)  
    elif: len(td_names) == 0 and len(fd_names) ==: 
        return 'name list is empty' 
    
