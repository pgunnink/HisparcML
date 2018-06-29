import numpy as np
import tables
import re
import os
from sapphire import HiSPARCStations
from ProcessDataML.DegRad import azimuth_zenith_to_cartestian
from sapphire.analysis.reconstructions import ReconstructSimulatedEvents, \
    ReconstructSimulatedCoincidences
import pdb

def filter_timings(timings):
    std = np.std(np.extract(timings>0,timings))
    if std>200:
        return False
    else:
        return True


def merge(stations, output = None, orig_stations=None, directory='.', verbose=True,
          overwrite=False, reconstruct=False):
    """
    Merges the simulation data from individual 'the_simulation.h5' files from the
    core.* directories inside a certain directory
    :param stations:        List of stations to use
    :param output:          The output file, by default main_data[503_505_...].h5
    :param orig_stations:   If there are more stations than those listed in the parameter
                            in the simulation files you need to list them here. By
                            default this is None, indicating that the stations are the
                            same
    :param directory:       The directory to look in, by default the current directory
    :return:
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
    cluster = HiSPARCStations(STATIONS)

    core_re = re.compile(r"core.*")
    dirs = [os.path.join(directory, o) for o in os.listdir(directory) if
            os.path.isdir(os.path.join(directory, o)) and core_re.match(o) is not None]

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
        azimuth_rec = tables.Float32Col()
        zenith_rec = tables.Float32Col()
        core_distance = tables.Float32Col()
        core_position = tables.Float32Col(shape=(2,))




    with tables.open_file(output_file, mode='w',
                          title='Collected data from %s' % STATIONS) as collected_traces:

        # create the table /traces/traces
        group = collected_traces.create_group('/', 'traces',
                                              'Traces with azimuth and zenith information')
        table = collected_traces.create_table(group, 'Traces', Traces, 'Traces')
        row = table.row


        if reconstruct:
            fmode = 'a'
        else:
            fmode = 'r'

        total = 0
        throwing_away = 0
        # loop over all core* directories
        for d in dirs:
            # wrap in try except block, because sometimes a core* dir exists,
            # but without the_simulation.h5 file
            try:
                template = '%s/the_simulation.h5' % d
                # open only in append mode if reconstructing direction
                with tables.open_file(template, fmode) as data:
                    if IGNORE_COINCIDENCES and reconstruct: # only possible if there is 1
                        # station (for now)
                        station_path = '/cluster_simulations/station_%s/' % STATIONS[0]
                        station = STATIONS[0]
                        rec = ReconstructSimulatedEvents(data, station_path, station,
                                                         verbose=False, overwrite=overwrite,
                                                         progress=False)
                        rec.reconstruct_and_store()
                        recs = data.get_node(station_path).reconstructions
                    if IGNORE_COINCIDENCES: # create one entry for every event in every
                        #  station
                        for station in data.root.cluster_simulations:
                            for station_event in station.events:

                                timings_station = np.array(
                                    [station_event['t1'], station_event['t2'],
                                     station_event['t3'], station_event['t4']])
                                # due to a bug in the GEANT simulation sometimes a timing
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
                                    trace_local[trace_local < -2] = -2
                                    trace[station, :, :] = trace_local
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
                                    pulseheights_local[pulseheights_local > 2] = 2
                                    pulseheights[:] = pulseheights_local

                                    # write to new h5 file
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
                                    row.append()
                                    total += 1
                                else:
                                    throwing_away += 1
                        data.close()
                    else:
                        if reconstruct:
                            rec = ReconstructSimulatedCoincidences(data,
                                                                   destination='reconstructions',
                                                                   overwrite=overwrite)
                            rec.reconstruct_and_store()
                            recs = data.get_node('/coincidences/reconstructions')
                        # recreating coincidences, so per coincidence a list of the
                        # traces etc. per station

                        # for evey incoming shower the coincidences are saved for some
                        # reason, but only if there is a hit the timestamp>0
                        for coin in data.root.coincidences.coincidences.where(
                                'timestamp>0'):
                            if coin['N'] >= 1:
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
                                        trace_local[trace_local<-2] = -2
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
                                        pulseheights_local[pulseheights_local>2] = 2
                                        pulseheights[station, :] = station_event['pulseheights']
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
            except Exception as e:
                if verbose:
                    print('Error occurred in %s' % (d))
                    print(e)
        table.flush()
        print('Total entries: %d' % total)
        print('Thrown away: %s' % throwing_away)