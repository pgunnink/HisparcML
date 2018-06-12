import tables
import h5py
import numpy as np
#from sapphire.utils import pbar
from sapphire.analysis.find_mpv import FindMostProbableValueInSpectrum
import pdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def update_timings(t, std):
    """
    Normalises data using the std and mean
    (t* = (t-mean(this subset of t))/std(all t))
    :param t: ND Numpy array of arrival times
    :return: ND Numpy array of normalised arrival times, with 0 if the original
    arrival time was 0
    """
    t_temp = t[t != 0.]
    avg = np.mean(t_temp)
    t_new = (t - avg) / std
    t_new[t == 0.] = 0
    return t_new

def update_all_timings(t):
    idx = t != 0.
    u = t[idx]
    t -= np.mean(u,)
    t /= np.std(u)
    t[~idx] = 0.
    return t
    
    
def read_sapphire_simulation(file_location, new_file_location, N_stations,
                             find_mips=True, uniform_dist=False, filter_detectors=False, 
                             no_gamma_peak=False, trigger=3):
    """
    read a h5 file made by merge.py from all individual simulation files
    :param file_location: location of h5 file made by merge.py
    :param new_file_location: location of h5 file made by read_sapphire_simulation
    :param N_stations: number of stations in h5 file
    :param find_mips: if True create pulseheight histogram and calculate MIPS from that (default: True)
    :param uniform_dist: if True force uniform distribution
    :return: nothing
    """

    # Open .h5 file, assuming N_stations stations are in this file, with 4
    # detectors each
    with tables.open_file(file_location, 'r') as data:
        entries = len(data.root.traces.Traces) # count number of samples

        # create an uniform distribution by looking at what zenith angle has the lowest
        #  number of events and using only that amount of angles
        if uniform_dist:
            available_zeniths = np.linspace(0., 60., 17, dtype=np.float32)
            events = []
            for angle in available_zeniths:
                settings = {'zenith_upper_bound': np.radians(angle + 1),
                            'angle_lower_bound': np.radians(angle - 1)}
                res = data.root.traces.Traces.read_where(
                    "(zenith>angle_lower_bound) & (zenith<zenith_upper_bound)", settings)
                events.append(len(res))
            events = np.array(events)
            print({k:v for k, v in zip(available_zeniths,events)})
            min_val = np.amin(events)
            entries = min_val*len(available_zeniths)
        # create empty arrays
        traces = np.empty([entries,N_stations,4,80])
        labels = np.empty([entries,3])
        timings = np.empty([entries,N_stations,4])
        pulseheights = np.empty([entries, N_stations, 4])
        rec_z = np.empty([entries])
        rec_a = np.empty([entries])
        zenith = np.empty([entries])
        azimuth = np.empty([entries])

        if uniform_dist:
            i = 0
            #for angle in pbar(available_zeniths):
            for angle in available_zeniths:
                settings = {'zenith_upper_bound': np.radians(angle + 1),
                            'angle_lower_bound': np.radians(angle - 1)}
                res = data.root.traces.Traces.read_where(
                    "(zenith>angle_lower_bound) & (zenith<zenith_upper_bound)", settings)
                for row in res[:min_val]:
                    if np.count_nonzero(row['timings']!=0.) >= trigger:
                        traces[i, :] = row['traces']
                        labels[i, :] = np.array([[row['x'], row['y'], row['z']]])
                        timings[i, :] = row['timings']
                        rec_z[i] = row['zenith_rec']
                        rec_a[i] = row['azimuth_rec']
                        pulseheights[i] = row['pulseheights']
                        zenith[i] = row['zenith']
                        azimuth[i] = row['azimuth']
                        i += 1
        else:
            # loop over all entries and fill them
            i = 0
            #for row in pbar(data.root.traces.Traces.iterrows(), entries):
            for row in data.root.traces.Traces.iterrows():
                if np.count_nonzero(row['timings']!=0.) >= trigger:
                    traces[i,:] = row['traces']
                    labels[i,:] = np.array([[row['x'],row['y'],row['z']]])
                    timings[i,:] = row['timings']
                    rec_z[i] = row['zenith_rec']
                    rec_a[i] = row['azimuth_rec']
                    pulseheights[i] = row['pulseheights']
                    zenith[i] = row['zenith']
                    azimuth[i] = row['azimuth']
                    i += 1

        # remove part of the array that was not filled
        # ( in case some filter criterium was used )
        traces = traces[:i,]
        labels = labels[:i,]
        timings = timings[:i,]
        rec_z = rec_z[:i,]
        rec_a = rec_a[:i,]


    
    if find_mips:


        # From this data determine the MiP peak
        # first create a pulseheight histogram
        pulseheights_flat = pulseheights.flatten()
        pulseheights_flat = pulseheights_flat.compress(np.abs(pulseheights_flat)>0)
        h, bins, patches = plt.hist(pulseheights_flat, bins=100)
        
        find_m_p_v = FindMostProbableValueInSpectrum(h,bins) # use the in-built search from Sapphire

        # if there is no gamma peak than the first guess algorithm fails, so make our own guess
        if no_gamma_peak:
            mpv_guess = bins[h.argmax()]
            try:
                mpv = (find_m_p_v.fit_mpv(mpv_guess),True)
            except:
                mpv = (-999, False)
        else:
            # find the peak in this pulseheight histogram
            mpv = find_m_p_v.find_mpv()     # mpv is a set, with first the mpv peak  and
                                            # second a boolean that is False if the search failed

        # ensure that the algorithm did not fail:
        if mpv[1]:
            mpv = mpv[0]
        else:
            raise AssertionError('No MPV found!')

        # calculate number of mips per trace
        pulseheights = np.reshape(pulseheights, [-1, N_stations * 4, 1])
        mips = np.zeros(pulseheights.shape)
        for i in range(pulseheights.shape[0]):
            mips[i,:] = pulseheights[i,:]/mpv
        plt.plot([mpv],[np.max(h)],'x')
        plt.savefig('Pulseheights.png')
        print('mpv %s' % mpv)

    # reshape traces such that for every coincidence we have N_stations*4 traces
    traces = np.reshape(traces, [-1, N_stations * 4, 80, 1])
    # calculate total trace (aka the pulseintegral)
    total_traces = np.reshape(np.sum(np.abs(traces), axis=2), [-1, N_stations * 4, 1])
    
    
    # take the log of a positive trace
    traces = np.log10(-1*traces+1)
    total_traces = np.log10(total_traces+1)
    total_traces -= np.mean(total_traces,axis=1)[:,np.newaxis]
    total_traces /= np.std(total_traces)
    
    
    # normalize the timings
    timings = np.reshape(timings, [-1, N_stations * 4, 1])
    idx = timings != 0.
    timings[~idx] = np.nan
    timings -= np.nanmean(timings,axis=1)[:,np.newaxis]
    timings /= np.nanstd(timings)
    timings[~idx] = 0.
    
    timings = np.reshape(timings, [-1,N_stations,4])
    
    #for i in pbar(range(timings.shape[0])):
    #    timings[i,:] = update_timings(timings[i,:],std)
    
    
    # again reshape the timings
    timings = np.reshape(timings, [-1, N_stations * 4, 1])
    # concatenate the pulseintegrals and timings
    input_features = np.concatenate((total_traces,timings),axis=2)


    # shuffle everything
    permutation = np.random.permutation(traces.shape[0])
    traces = traces[permutation,:]
    labels = labels[permutation,:]
    input_features = input_features[permutation,:]
    if find_mips:
        mips = mips[permutation,:]
    rec_z = rec_z[permutation]
    rec_a = rec_a[permutation]
    zenith = zenith[permutation]
    azimuth = azimuth[permutation]

    # Save everything into a h5 file
    with h5py.File(new_file_location, 'w') as f:
        traces_set = f.create_dataset('traces',data=traces)
        labels_set = f.create_dataset('labels', data=labels)
        input_features_set = f.create_dataset('input_features',data=input_features)
        if find_mips:
            mips_set = f.create_dataset('mips',data=mips)
        rec_z_set = f.create_dataset('rec_z', data=rec_z)
        rec_a_set = f.create_dataset('rec_a', data=rec_a)
        zenith_set = f.create_dataset('zenith', data=zenith)
        azimuth_set = f.create_dataset('azimuth', data=azimuth)
