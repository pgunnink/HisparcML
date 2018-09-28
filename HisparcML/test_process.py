from HisparcML.ProcessData import read_sapphire_simulation
import numpy as np
DIR = '/data/hisparc/pgunnink/MachineLearning/Simulation/CustomDriehoek'
H5FILE = DIR+"/main_data_[kascade].h5"
new_H5FILE = DIR+"/driehoek_test.h5"

#N_stations = len([501, 502, 503, 504, 505, 506, 510, 511, 508])


N_stations = 1
ENERGY_LOW = 0
ENERGY_HIGH = 11**17

weights = np.ones(17)
weights[10] = weights[10]*0.7


#merge(STATIONS, H5FILE, directory=DIR, verbose=True,
#          overwrite=False, reconstruct=False)
read_sapphire_simulation(H5FILE, new_H5FILE, N_stations,
                             find_mips=True, uniform_dist=False,
                             no_gamma_peak=False, trigger=3, energy_low=ENERGY_LOW,
                             energy_high=ENERGY_HIGH, verbose=True,
                             max_samples=1, CHUNK_SIZE=10**4, zenith_weights=None,)