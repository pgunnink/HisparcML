import tables
import h5py
import numpy as np
from sapphire.analysis.find_mpv import FindMostProbableValueInSpectrum
import pdb
import matplotlib
from fast_histogram import histogram1d
import timeit


# determine where to plot to
try:
    cfg = get_ipython().config 
    if cfg['IPKernelApp']['kernel_class'] == 'google.colab._kernel.Kernel':
        pass
    else:
        matplotlib.use('Agg')
except NameError:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt


    

def read_sapphire_simulation(file_location, new_file_location, N_stations,
                             find_mips=True, uniform_dist=False, rossi_dist=False,
                             no_gamma_peak=False, trigger=3, energy_low=9.9**16.5, 
                             energy_high=10.1**16.5, verbose=True,
                             max_samples=1, CHUNK_SIZE=10**4, skip_nonreconstructed=True):
    """
    read a h5 file made by merge.py from all individual simulation files

    traces (and resulting variables pulseheights and pulseintegras) are scaled by the
    position of the mip peak

    :param file_location: location of h5 file made by merge.py
    :param new_file_location: location of h5 file made by read_sapphire_simulation
    :param N_stations: number of stations in h5 file
    :param find_mips: if True create pulseheight histogram and calculate MIPS from that (default: True)
    :param uniform_dist: if True force uniform distribution
    :param no_gamma_peak: if using simulation data, which has no big first peak in the pulseheight histogram
    :param trigger: the number of detectors that must have triggered
    :param energy_low: low energy cut
    :param energy_high: high energy cut
    :param max_samples: ratio of the number of samples to cut (use only with uniform_dist=True)
    :param CHUNK_SIZE: size of individual CHUNKS that are writte together to disk
    :param verbose: if True write output and create plots
    :return: nothing
    """
    start_time = timeit.default_timer()
    # Open .h5 file, assuming N_stations stations are in this file, with 4
    # detectors each
    with tables.open_file(file_location, 'r') as data:
        #entries = len(data.root.traces.Traces) # count number of samples
        settings = {'energy_lower_bound': energy_low,
                    'energy_upper_bound': energy_high}
        res = data.root.traces.Traces.read_where(
            "(energy > energy_lower_bound) & (energy < energy_upper_bound)",
            settings)
        entries = len(res)
        with h5py.File(new_file_location, 'w') as f:
            # create the h5py dataset files
            traces = f.create_dataset('traces', shape=(entries, N_stations*4, 80),
                                      chunks=(CHUNK_SIZE, N_stations*4, 80),
                                      dtype='float32')
            labels = f.create_dataset('labels', shape=(entries, 3),
                                      chunks=(CHUNK_SIZE,3), dtype='float32')
            input_features = f.create_dataset('input_features', shape=(entries,N_stations*4, 2),
                                       chunks=(CHUNK_SIZE, N_stations*4, 2),
                                       dtype='float32')
            pulseheights = f.create_dataset('pulseheights',
                                            shape=(entries, N_stations*4),
                                            chunks=(CHUNK_SIZE,N_stations*4),
                                            dtype='int32')
            rec_z = f.create_dataset('rec_z', shape=(entries,), chunks=(CHUNK_SIZE,),
                                     dtype='float32')
            rec_a = f.create_dataset('rec_a', shape=(entries,), chunks=(CHUNK_SIZE,),
                                     dtype='float32')
            zenith = f.create_dataset('zenith', shape=(entries,), chunks=(CHUNK_SIZE,),
                                     dtype='float32')
            azimuth = f.create_dataset('azimuth', shape=(entries,), chunks=(CHUNK_SIZE,),
                                     dtype='float32')
            energy = f.create_dataset('energies', shape=(entries,), chunks=(CHUNK_SIZE,),
                                     dtype='float32')

            available_zeniths = np.linspace(0., 60., 17, dtype=np.float32)
            filled = np.zeros(17)
            # create an uniform distribution by looking at what zenith angle has the lowest
            #  number of events and using only that amount of angles
            if uniform_dist:
                events = []
                for angle in available_zeniths:
                    settings = {'zenith_upper_bound': np.radians(angle + 1),
                                'angle_lower_bound': np.radians(angle - 1),
                                'energy_lower_bound': energy_low,
                                'energy_upper_bound': energy_high}
                    res = data.root.traces.Traces.read_where(
                        "(zenith>angle_lower_bound) & (zenith<zenith_upper_bound) & "
                        "(energy > energy_lower_bound) & (energy < energy_upper_bound)",
                        settings)
                    events.append(len(res))
                events = np.array(events)
                if verbose:
                    print({k:v for k, v in zip(available_zeniths,events)})
                min_val = np.ones(available_zeniths.shape)*np.amin(events)*max_samples
                total_entries_max = entries
            elif rossi_dist:
                events = []
                for angle in available_zeniths:
                    settings = {'zenith_upper_bound': np.radians(angle + 1),
                                'angle_lower_bound': np.radians(angle - 1),
                                'energy_lower_bound': energy_low,
                                'energy_upper_bound': energy_high}
                    res = data.root.traces.Traces.read_where(
                        "(zenith>angle_lower_bound) & (zenith<zenith_upper_bound) & "
                        "(energy > energy_lower_bound) & (energy < energy_upper_bound)",
                        settings)
                    events.append(len(res))
                events = np.array(events)
                scale = events[5]/.19
                new_dist = lambda theta,scaling : np.sin(theta)*np.cos(
                    theta)**8*scaling+0.1*scaling
                while ((events - new_dist(np.radians(available_zeniths),scale))<0).any():
                    scale *= 0.9
                else:
                    if verbose:
                        plt.figure()
                        plt.plot(available_zeniths,events,'o',label='Available events')
                        plt.plot(available_zeniths,new_dist(np.radians(available_zeniths),
                                                            scale),'-',
                                 label='New dist')
                        plt.legend()
                        plt.savefig('New_dist.png')
                    min_val = new_dist(np.radians(available_zeniths), scale)*max_samples
                    total_entries_max = entries


            else:
                min_val = np.ones(available_zeniths.shape)*entries
                total_entries_max = entries*max_samples
            if verbose:
                print('Maximum number of entries: %s' % total_entries_max)

            # create temporary chunk sizes that can then be written to the h5 file
            traces_temp = np.empty([CHUNK_SIZE, N_stations*4, 80])
            labels_temp = np.empty([CHUNK_SIZE, 3])
            timings_temp = np.empty([CHUNK_SIZE, N_stations*4])
            pulseheights_temp = np.empty([CHUNK_SIZE, N_stations*4])
            rec_z_temp = np.empty([CHUNK_SIZE])
            rec_a_temp = np.empty([CHUNK_SIZE])
            zenith_temp = np.empty([CHUNK_SIZE])
            azimuth_temp = np.empty([CHUNK_SIZE])
            energy_temp = np.empty([CHUNK_SIZE])



            i = 0
            i_chunk = 0
            chunk_count = 0
            # loop over all entries and fill them
            for row in data.root.traces.Traces.iterrows():
                # first filter on the trigger
                if np.count_nonzero(row['timings']!=0.) >= trigger:
                    # now filter on energy
                    if row['energy']>=energy_low and row['energy']<=energy_high:
                        # and filter on zenith angle (if a distribution is wanted)
                        idx = (np.abs(np.radians(available_zeniths) - row['zenith'])).argmin()
                        if filled[idx]<min_val[idx] and i<total_entries_max:
                            if np.isnan(row['zenith_rec']) and skip_nonreconstructed:
                                continue



                            # read neccessary data from h5 file and create the
                            # temporary chunks
                            t = row['traces'].reshape((4*N_stations,80))
                            t = np.log10(-1 * t  + 1)

                            traces_temp[i_chunk,:] = t
                            labels_temp[i_chunk,:] = np.array([[row['x'],row['y'],row['z']]])
                            timings_temp[i_chunk,:] = row['timings'].reshape((4*N_stations,))
                            if np.isnan(row['zenith_rec'])
                                rec_z_temp[i_chunk] = np.nan
                                rec_a_temp[i_chunk] = np.nan
                            else:
                                rec_z_temp[i_chunk] = row['zenith_rec']
                                rec_a_temp[i_chunk] = row['azimuth_rec']

                            pulseheights_temp[i_chunk] = row['pulseheights'].reshape((4*N_stations,))
                            zenith_temp[i_chunk] = row['zenith']
                            azimuth_temp[i_chunk] = row['azimuth']
                            energy_temp[i_chunk] = row['energy']

                            i_chunk += 1
                            i += 1
                            # when we have gathered enough events write them all to
                            # disk at once
                            if i_chunk==CHUNK_SIZE:
                                chunk_count += 1
                                if verbose:
                                    print('Writing chunk %s' % chunk_count)
                                i_chunk = 0

                                traces[i-CHUNK_SIZE:i,] = traces_temp
                                labels[i-CHUNK_SIZE:i,] = labels_temp
                                input_features[i - CHUNK_SIZE:i,:,0] = timings_temp
                                rec_z[i - CHUNK_SIZE:i] = rec_z_temp
                                rec_a[i - CHUNK_SIZE:i] = rec_a_temp
                                pulseheights[i - CHUNK_SIZE:i,] = pulseheights_temp
                                zenith[i - CHUNK_SIZE:i] = zenith_temp
                                azimuth[i - CHUNK_SIZE:i] = azimuth_temp
                                energy[i - CHUNK_SIZE:i] = energy_temp
                            filled[idx] += 1 # in order to keep track of zenith
                            # distribution
                        elif i>total_entries_max:
                            break
            # make sure to write the last, half-filled chunk as well
            if i_chunk>0: # make sure that the last chunk is actually filled
                traces[i - i_chunk:i,] = traces_temp[:i_chunk,]
                labels[i - i_chunk:i,] = labels_temp[:i_chunk,]
                input_features[i - i_chunk:i,:,0] = timings_temp[:i_chunk,]
                rec_z[i - i_chunk:i] = rec_z_temp[:i_chunk,]
                rec_a[i - i_chunk:i] = rec_a_temp[:i_chunk,]
                pulseheights[i - i_chunk:i,] = pulseheights_temp[:i_chunk,]
                zenith[i - i_chunk:i] = zenith_temp[:i_chunk,]
                azimuth[i - i_chunk:i] = azimuth_temp[:i_chunk,]
                energy[i - i_chunk:i] = energy[:i_chunk,]

            if verbose:
                print('Filling datasets %s'% (timeit.default_timer() - start_time))
                print('Out of %.2d items %.2d remained' % (entries, i))

            # remove part of the array that was not filled
            traces.resize(i,axis=0)
            labels.resize(i,axis=0)
            input_features.resize(i,axis=0)
            rec_z.resize(i,axis=0)
            rec_a.resize(i,axis=0)
            pulseheights.resize(i,axis=0)
            zenith.resize(i,axis=0)
            azimuth.resize(i,axis=0)
            energy.resize(i, axis=0)
            new_entries = i

            if find_mips:
                # From this data determine the MiP peak
                # first create a pulseheight histogram
                pulseheights_flat = pulseheights[:].flatten()
                pulseheights_flat = pulseheights_flat.compress(np.abs(pulseheights_flat)>0)
                r = [0,np.max(pulseheights_flat)]
                number_of_bins = 100
                bins = np.linspace(r[0],r[1],number_of_bins)
                h = histogram1d(pulseheights_flat, range=r, bins= number_of_bins)

                # plot the pulseheights
                if verbose:
                    plt.figure()
                    plt.bar(bins, h, width=(bins[1] - bins[0]))




                del pulseheights_flat # clear some memory
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
                    plt.plot([mpv], [np.max(h)], 'x')
                    print('mpv %s' % mpv)
                    plt.savefig('Pulseheights.png')
                else:
                    plt.savefig('Pulseheights.png')
                    raise AssertionError('No MPV found!')
                # calculate number of mips per pulse
                mips = f.create_dataset('mips', shape=(len(traces), N_stations*4),
                                            chunks=(CHUNK_SIZE,N_stations*4),
                                            dtype='float64')
                # loop over the pulseheights and convert to mips
                for i in range(int(np.floor(new_entries/CHUNK_SIZE))):
                    pulseheights_temp = pulseheights[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE,:]
                    mips[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE,:] = pulseheights_temp
                else:
                    remaining = new_entries % CHUNK_SIZE
                    pulseheights_temp = pulseheights[i*CHUNK_SIZE:i*CHUNK_SIZE+remaining,:]
                    mips[i*CHUNK_SIZE:i*CHUNK_SIZE+remaining,:] = pulseheights_temp



            if verbose:
                print('Finding MiP %s'% (timeit.default_timer() - start_time))

            # calculate total trace (aka the pulseintegral) and rescale by mpv
            total_traces = np.reshape(np.sum(np.abs(10**traces[:]-1) / mpv, axis=2),
                                      [-1, N_stations * 4])
            total_traces = np.log10(total_traces+1)
            total_traces -= np.mean(total_traces, axis=1)[:, np.newaxis]
            print("Std of integrals: %s" % np.std(total_traces))
            total_traces /= np.std(total_traces)
            if verbose:
                print('Creating total_traces %s'% (timeit.default_timer() - start_time))

            # normalize the timings
            timings = input_features[:,:,0]
            idx = timings != 0.
            timings[~idx] = np.nan
            timings -= np.nanmean(timings,axis=1)[:,np.newaxis]
            if verbose:
                plt.figure()
                plt.hist(np.extract(~np.isnan(timings.flatten()),timings.flatten()),
                         bins=np.linspace(-100,100,50))
                plt.savefig('histogram_timings.png')
            print('Std of timings: %s' % np.nanstd(timings))
            timings /= np.nanstd(timings)
            timings[~idx] = 0.
            if verbose:
                print('Normalizing input_features %s'% (timeit.default_timer() - start_time))

            input_features[:,:,:] = np.stack((timings,total_traces),axis=2)

            # shuffle everything
            permutation = np.random.permutation(new_entries)
            traces[:] = np.log10((10**traces[:][permutation,:]-1)/mpv + 1)
            labels[:] = labels[:][permutation,:]
            input_features[:] = input_features[:][permutation,:]
            if find_mips:
                mips[:] = mips[:][permutation,]
            rec_z[:] = rec_z[:][permutation]
            rec_a[:] = rec_a[:][permutation]
            zenith[:] = zenith[:][permutation]
            azimuth[:] = azimuth[:][permutation]
            energy[:] = energy[:][permutation]
            if verbose:
                print('Shuffling everything %s'% (timeit.default_timer() - start_time))
