import numpy as np
import tables
import re
import os
from sapphire import HiSPARCStations
from ProcessDataML.DegRad import azimuth_zenith_to_cartestian
from sapphire.analysis.reconstructions import ReconstructSimulatedEvents
import pdb

def merge(stations, output = None, orig_stations=None, directory='.', verbose=True,
          overwrite=False):
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
    cluster = HiSPARCStations(STATIONS, force_stale=True)

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
        core_distance = tables.Float32Col(shape=(len(STATIONS)))


    with tables.open_file(output_file, mode='w',
                          title='Collected data from %s' % STATIONS) as collected_traces:
        group = collected_traces.create_group('/', 'traces',
                                              'Traces with azimuth and zenith information')
        table = collected_traces.create_table(group, 'Traces', Traces, 'Traces')
        row = table.row

        total_ignored = 0
        total = 0
        #for d in pbar(dirs):
        for d in dirs:
            try:
                template = '%s/the_simulation.h5' % d
                with tables.open_file(template, 'a') as data:
                    if IGNORE_COINCIDENCES: # only possible if there is 1 station (for now)
                        station_path = '/cluster_simulations/station_%s/' % STATIONS[0]
                        station = STATIONS[0]
                        rec = ReconstructSimulatedEvents(data, station_path, station,
                                                         verbose=False, overwrite=overwrite,
                                                         progress=False)
                        # rec.direction = CoincidenceDirectionReconstructionDetectors(cluster)
                        rec.reconstruct_and_store()
                        recs = data.get_node(station_path).reconstructions
                    if IGNORE_COINCIDENCES:
                        for station in data.root.cluster_simulations:
                            for station_event in station.events:
                                trace = np.zeros([4, 80], dtype=np.float32)
                                timings = np.zeros([4], dtype=np.float32)
                                pulseheights = np.zeros([4], dtype=np.int16)
                                trace[:, :] = station_event['traces']
                                zenith = station_event['zenith']
                                azimuth = station_event['azimuth']
                                energy = station_event['shower_energy']
                                distance_core = station_event['core_distance']
                                timings_station = np.array(
                                    [station_event['t1'], station_event['t2'],
                                     station_event['t3'], station_event['t4']])
                                timings_station[timings_station < 0] = 0
                                timings[:] = timings_station
                                pulseheights[:] = station_event['pulseheights']
                                row['traces'] = trace
                                row['N'] = 1
                                row['azimuth'] = azimuth
                                row['zenith'] = zenith
                                row['energy'] = energy
                                row['timings'] = timings
                                x, y, z = azimuth_zenith_to_cartestian(zenith, azimuth)
                                row['x'] = x
                                row['y'] = y
                                row['z'] = z
                                row['azimuth_rec'] = recs.col('azimuth')[station_event['event_id']]
                                row['zenith_rec'] = recs.col('zenith')[station_event['event_id']]
                                row['pulseheights'] = pulseheights
                                row['core_distance'] = distance_core
                                row['id'] = total
                                row.append()
                                total += 1
                        data.close()
                    else:
                        for coin in data.root.coincidences.coincidences:
                            if coin['N'] >= 1:
                                trace = np.zeros([len(STATIONS), 4, 80], dtype=np.float32)
                                timings = np.zeros([len(STATIONS), 4], dtype=np.float32)
                                pulseheights = np.zeros([len(STATIONS), 4], dtype=np.int16)
                                core_distance = np.zeros((len(STATIONS),),
                                                         dtype=np.float32)
                                c_index = data.root.coincidences.c_index[coin['id']]
                                for station, event_idx in c_index:
                                    station_path = data.root.coincidences.s_index[station].decode(
                                        'UTF-8')
                                    if re.search(combined_regex, station_path) is not None:
                                        station_event = data.get_node(station_path, 'events')[
                                            event_idx]
                                        station = STATIONS.index(ORIG_STATIONS[station])
                                        trace[station, :, :] = station_event['traces']
                                        zenith = station_event['zenith']
                                        azimuth = station_event['azimuth']
                                        energy = station_event['shower_energy']
                                        timings_station = np.array(
                                            [station_event['t1'], station_event['t2'],
                                             station_event['t3'], station_event['t4']])
                                        timings_station[timings_station < 0] = 0
                                        timings[station, :] = timings_station
                                        pulseheights[station, :] = station_event['pulseheights']
                                        core_distance[station] = station_event['core_distance']
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
                                #row['azimuth_rec'] = recs.col('azimuth')[coin['id']]
                                #row['zenith_rec'] = recs.col('zenith')[coin['id']]
                                row['pulseheights'] = pulseheights
                                row['id'] = total
                                row.append()
                                total += 1
            except Exception as e:
                if verbose:
                    print('Error occurred in %s' % (d))
                    print(e)
                # os.system('rm -r %s' % d)
        table.flush()
        print('Total entries: %d' % (total))