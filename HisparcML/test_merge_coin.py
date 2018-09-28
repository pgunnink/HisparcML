from HisparcML.MergeData import merge
from sapphire.clusters import CompassStations
import numpy as np
DIR = '/data/hisparc/pgunnink/MachineLearning/Simulation/CustomDriehoek'
STATIONS = [9001, 9002, 9003]
H5FILE = DIR+"/main_data_test.h5"

# r,alpha,z,beta taken from the 501 station
compass_coordinates = [(6.09, -158.07,  0.0, 45.0),
                       (5.0,    73.49,  0.0,    45.0),
                       (14.51, 95.63,   0.0,    315.0),
                       (11.06,  138.13, 0.0,    315.0)]

driehoek_zijdes = 100
cluster = CompassStations()
cluster._add_station((0, driehoek_zijdes/np.sqrt(3), 0), compass_coordinates, number=9001)
cluster._add_station((-driehoek_zijdes/2,0,0), compass_coordinates, number=9002)
cluster._add_station((driehoek_zijdes/2,0,0), compass_coordinates, number=9003)



merge(STATIONS, H5FILE, directory=DIR, verbose=True,
          overwrite=True, reconstruct=False, coincidences=3, cluster=cluster)