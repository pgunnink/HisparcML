import numpy as np
import tables
import re
import os
from sapphire import HiSPARCStations
from HisparcML.DegRad import azimuth_zenith_to_cartestian
from sapphire.analysis.reconstructions import ReconstructSimulatedEvents, \
    ReconstructSimulatedCoincidences
from tqdm import tqdm
import pdb

MAX_VOLTAGE = 4096*0.57

def filter_timings(timings):
    std = np.std(np.extract(timings>0,timings))
    if std>200:
        return False
    else:
        return True


def merge(stations, output = None, orig_stations=None, directory='.', verbose=True,
          overwrite=False, reconstruct=False, save_coordinates=False,
          only_original=False, coincidences=1, cluster=None, photontimes=False):
    """
    Merges the simulation data from individual 'the_simulation.h5' files from the
    core.* directories inside a certain directory

    PS: Deze functie is ook een beetje een draak en heeft ook nodig onderhoud nodig.
    Feitelijk gebeurt er niet heel veel: je leest alleen alle the_simulation.h5
    bestanden uit en combineert ze. Het is dan waarschijnlijker ook makkelijker als je
    zelf even iets klust.

    :param stations:        List of stations to use
    :param output:          The output file, by default main_data[503_505_...].h5
    :param orig_stations:   If there are more stations than those listed in the parameter
                            in the simulation files you need to list them here. By
                            default this is None, indicating that the stations are the
                            same
    :param directory:       The directory to look in, by default the current directory
    :param only_original:   If True only look for directories in the format coreXX,
                            else look in every directory
    :param verbose:         If True print more output statements
    :param overwrite:       If True overwrite the Sapphire reconstruction in the
                            individual the_simulation.h5 files
    :param reconstruct:     If True reconstruct directions using Sapphire
    :param save_coordinates:If True store the impact coordinates of the first particle
                            that hit the detector (requires a version of sapphire that
                            saves this information)
    :param coincidences:    Minimum number of coincidences to include
    :param cluster:         A Cluster object, if None creates a Cluster object itself
                            using the list of stations provided.
    :param photontimes:     If True stores the photontimes histogram. Also adds the
                            individual timings to the photontimes, so traces are correctly
                            timed relative to each other.
    :return: nothing
    """

    # process parameters
    STATIONS = stations
    if orig_stations is not None:
        ORIG_STATIONS = orig_stations
    else:
        ORIG_STATIONS = STATIONS

    ORIG_LENGTH = len(ORIG_STATIONS)
    if output is not None:
        output_file = output
    else:
        output_file = 'main_data_%s.h5' % str(STATIONS).replace(', ', '_')

    # if the length of the original stations is 1 then sapphire will not have included
    # a coincidence table, so just grab all the events from the one station
    if ORIG_LENGTH==1:
        IGNORE_COINCIDENCES = True
    else:
        IGNORE_COINCIDENCES = False

    combined_regex = "(" + ")|(".join([str(a) for a in STATIONS]) + ")"
    N = len(STATIONS)
    if cluster is None:
        cluster = HiSPARCStations(STATIONS)

    if only_original:
        core_re = re.compile(r"core.*\d$")
    else:
        core_re = re.compile(r"core.*")
    dirs = [os.path.join(directory, o) for o in os.listdir(directory) if
            os.path.isdir(os.path.join(directory, o)) and core_re.match(o) is not None]

    # The pytables class that describes the table to store in the h5 file
    class Traces(tables.IsDescription):
        id = tables.Int32Col()
        N = tables.Int32Col()
        azimuth = tables.Float32Col()
        zenith = tables.Float32Col()
        traces = tables.Int16Col(shape=(len(STATIONS), 4, 80))
        energy = tables.Float32Col()
        timings = tables.Float32Col(shape=(len(STATIONS), 4))
        pulseheights = tables.Int16Col(shape=(len(STATIONS), 4))
        x = tables.Float32Col()
        y = tables.Float32Col()
        z = tables.Float32Col()
        azimuth_rec = tables.Float32Col(dflt=np.nan)
        zenith_rec = tables.Float32Col(dflt=np.nan)
        core_distance = tables.Float32Col()
        core_position = tables.Float32Col(shape=(2,))
        photontimes = tables.Float32Col(shape=(len(STATIONS), 4, 80,))
        arrivaltimes_particles = tables.Float32Col(shape=(len(STATIONS), 4))
        if save_coordinates:
            inslag_coordinates = tables.Float32Col((4,2))
            n_electron_muons = tables.Int16Col(shape=4)

    with tables.open_file(output_file, mode='w',
                          title='Collected data from %s' % STATIONS) as collected_traces:

        # create the table /traces/traces
        group = collected_traces.create_group('/', 'traces',
                                              'Traces with azimuth and zenith information')
        table = collected_traces.create_table(group, 'Traces', Traces, 'Traces')
        row = table.row

        # keep track of numbers in order to print those at the end
        total = 0
        throwing_away = 0
        # loop over all core* directories
        for d in tqdm(dirs):
            # wrap in try except block, because sometimes a core* dir exists,
            # but without the_simulation.h5 file
            try:
                template = '%s/the_simulation.h5' % d
                # open only in append mode if reconstructing direction and if there is
                # a sim.py.e1091201 file in the directory, which indicates that the
                # simulation is done running and we can safely open the file in 'a' mode
                list_of_files = os.listdir(d)
                re_sim = re.compile('sim\.py\.e[0-9]*')
                reconstruct_local = False
                if reconstruct:
                    if np.count_nonzero([re_sim.match(x) for x in list_of_files])>0:
                        reconstruct_local = True


                if reconstruct_local:
                    fmode = 'a'
                else:
                    fmode = 'r'
                with tables.open_file(template, fmode) as data:
                    if IGNORE_COINCIDENCES and reconstruct_local:
                        # only possible if there is 1 station (for now)

                        station_path = '/cluster_simulations/station_%s/' % STATIONS[0]
                        station = STATIONS[0]
                        rec = ReconstructSimulatedEvents(data, station_path, station,
                                                         verbose=False, overwrite=overwrite,
                                                         progress=False)
                        try:
                            rec.reconstruct_and_store()
                        except:
                            if verbose:
                                print('Already reconstructed')
                        recs = data.get_node(station_path).reconstructions
                    if IGNORE_COINCIDENCES: # create one entry for every event in every
                        #  station
                        for station in data.root.cluster_simulations:
                            if photontimes:
                                photontimes_table = station.photontimes
                            for station_event in station.events:

                                timings_station = np.array(
                                    [station_event['t1'], station_event['t2'],
                                     station_event['t3'], station_event['t4']])
                                # due to a bug in the simulation sometimes a timing
                                #  of -999 is included, so filter these out (bug is
                                # fixed now, so should no longer be neccessary unless
                                # when working with old data)
                                if filter_timings(timings_station):
                                    # create empty arrays to fill
                                    trace = np.zeros([4, 80], dtype=np.float32)
                                    timings = np.zeros([4], dtype=np.float32)
                                    pulseheights = np.zeros([4], dtype=np.int16)

                                    # fill using data from h5 file
                                    trace_local = station_event['traces']
                                    trace_local[trace_local < -MAX_VOLTAGE] = -MAX_VOLTAGE
                                    trace[:, :] = trace_local
                                    zenith = station_event['zenith']
                                    azimuth = station_event['azimuth']
                                    energy = station_event['shower_energy']
                                    distance_core = station_event['core_distance']

                                    if np.count_nonzero(np.isnan(timings_station))>0:
                                        print(timings_station)

                                    # remove the -999 timings and set to 0
                                    timings_station[timings_station < 0] = 0
                                    timings[:] = timings_station
                                    pulseheights_local = station_event['pulseheights']
                                    pulseheights_local[pulseheights_local >
                                                       MAX_VOLTAGE] = MAX_VOLTAGE
                                    pulseheights[:] = pulseheights_local

                                    if np.count_nonzero(
                                            (pulseheights>((4096*0.57/10e3)*1e3-1))
                                            &
                                            (pulseheights<((4096*0.57/10e3)*1e3+1))
                                            )>0:
                                        throwing_away += 1
                                        continue
                                    # write to new h5 file

                                    # do some magic to recreate timings and shift
                                    # photontimes (this is really a bit of a pointless
                                    # exercise, since you should just do this right in
                                    # the simulation anyway)
                                    particle_timings = np.zeros((4,))
                                    earliest_particle = np.inf
                                    non_zero_idx = []
                                    for i, (t, t0) in enumerate(zip(trace, timings)):
                                        old_trigger_delay = 0
                                        for i_local, value in enumerate(t):
                                            if value < -30.0:
                                                old_trigger_delay = i_local * 2.5
                                                break
                                        else:
                                            particle_timings[i] = np.nan
                                            continue
                                        non_zero_idx.append(i)
                                        particle_timings[i] = timings[i] - old_trigger_delay
                                        if particle_timings[i] < earliest_particle:
                                            earliest_particle = particle_timings[i]
                                    particle_timings -= np.nanmin(particle_timings)
                                    row['arrivaltimes_particles'] = particle_timings
                                    if photontimes:
                                        row_photontimes = np.zeros((4,80))
                                        for i, idx in enumerate(station_event['photontimes_idx']):
                                            if np.isnan(particle_timings[i]): #there is
                                                #  no incident particle or the
                                                # particles that were incidident did
                                                # not produce a sufficiently high trace
                                                #  to trigger
                                                continue
                                            pt = photontimes_table[idx] + \
                                                 particle_timings[i]
                                            local_hist, _ = np.histogram(pt,
                                                                       bins=np.linspace(0,200,81))
                                            row_photontimes[i,:] = local_hist
                                        row['photontimes'] = row_photontimes



                                    row['traces'] = trace
                                    row['N'] = 1
                                    row['azimuth'] = azimuth
                                    row['zenith'] = zenith
                                    row['energy'] = energy
                                    row['timings'] = timings
                                    # convert zenith and azimuth to x y z
                                    x, y, z = azimuth_zenith_to_cartestian(zenith, azimuth)
                                    row['x'] = x
                                    row['y'] = y
                                    row['z'] = z
                                    if reconstruct:
                                        row['azimuth_rec'] = recs.col('azimuth')[station_event['event_id']]
                                        row['zenith_rec'] = recs.col('zenith')[station_event['event_id']]
                                    row['pulseheights'] = pulseheights
                                    row['core_distance'] = distance_core
                                    row['id'] = total

                                    if save_coordinates:
                                        row['inslag_coordinates'] = station_event[
                                            'coordinates']
                                        row['n_electron_muons'] = \
                                            [station_event["n_electrons1"] +
                                             station_event["n_muons1"],
                                             station_event["n_electrons2"] +
                                             station_event["n_muons2"],
                                             station_event["n_electrons3"] +
                                             station_event["n_muons3"],
                                             station_event["n_electrons4"] +
                                             station_event["n_muons4"]]
                                    row.append()
                                    total += 1
                                else:
                                    throwing_away += 1
                        data.close()
                    else:
                        if reconstruct and reconstruct_local:

                            rec = ReconstructSimulatedCoincidences(data,
                                                                   destination='reconstructions',
                                                                   overwrite=overwrite,
                                                                   progress=False)
                            rec.reconstruct_and_store()
                            recs = data.get_node('/coincidences/reconstructions')
                        # recreating coincidences, so per coincidence a list of the
                        # traces etc. per station

                        # for every incoming shower the coincidences are saved for some
                        # reason, but only if there is a hit the timestamp>0

                        # WARNING: I have not used this for a long time,
                        # so it probably is broken (or does not the same things as with
                        #  the single station). Basically do not use.
                        for coin in data.root.coincidences.coincidences.where(
                                'timestamp>0'):
                            if coin['N'] >= coincidences:
                                trace = np.zeros([len(STATIONS), 4, 80], dtype=np.float32)
                                timings = np.zeros([len(STATIONS), 4], dtype=np.float32)
                                pulseheights = np.zeros([len(STATIONS), 4], dtype=np.int16)

                                # the c_index maps the coincidences to the events in
                                # the format [[station_id, event_id],...]. You then use
                                # the
                                # s_index table to look up the path to the station
                                # using the station_id
                                c_index = data.root.coincidences.c_index[coin['id']]
                                for station, event_idx in c_index:
                                    station_path = data.root.coincidences.s_index[station].\
                                        decode('UTF-8')
                                    # make sure the path exists
                                    if re.search(combined_regex, station_path) is not None:
                                        station_event = data.get_node(station_path, 'events')[
                                            event_idx]
                                        station = STATIONS.index(ORIG_STATIONS[station])
                                        trace_local = station_event['traces']
                                        trace_local[trace_local<-MAX_VOLTAGE] = -MAX_VOLTAGE
                                        trace[station, :, :] = trace_local
                                        zenith = station_event['zenith']
                                        azimuth = station_event['azimuth']
                                        energy = station_event['shower_energy']
                                        timings_station = np.array(
                                            [station_event['t1'], station_event['t2'],
                                             station_event['t3'], station_event['t4']])

                                        timings_station[timings_station < 0] = 0
                                        timings[station, :] = timings_station
                                        pulseheights_local = station_event['pulseheights']
                                        pulseheights_local[pulseheights_local>MAX_VOLTAGE] = MAX_VOLTAGE
                                        pulseheights[station, :] = station_event['pulseheights']
                                # due to stupidity I set the cap to 4096*0.57/10e3
                                # mV instead of 4096*0.57/1e3 ...
                                if np.count_nonzero(
                                        (pulseheights>((4096*0.57/10e3)*1e3-1))
                                        &
                                        (pulseheights<((4096*0.57/10e3)*1e3+1))
                                        )>0:
                                    throwing_away += 1
                                    continue



                                row['traces'] = trace
                                row['N'] = coin['N']
                                row['azimuth'] = azimuth
                                row['zenith'] = zenith
                                row['energy'] = energy
                                row['timings'] = timings
                                x, y, z = azimuth_zenith_to_cartestian(zenith, azimuth)
                                row['x'] = x
                                row['y'] = y
                                row['z'] = z
                                if reconstruct:
                                    row['azimuth_rec'] = recs.col('azimuth')[coin['id']]
                                    row['zenith_rec'] = recs.col('zenith')[coin['id']]
                                row['pulseheights'] = pulseheights
                                row['id'] = total
                                row['core_position'] = [coin['x'], coin['y']]

                                row.append()
                                total += 1
                            else:
                                throwing_away += 1
            except Exception as e:
                if verbose:
                    print('Error occurred in %s' % (d))
                    print(e)
        table.flush()
        print('Total entries: %d' % total)
        print('Thrown away: %s' % throwing_away)