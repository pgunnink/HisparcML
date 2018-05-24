from MergeData import merge
from ProcessData import read_sapphire_simulation


STATIONS = [503, 504, 506]
DIR = '/data/hisparc/pgunnink/MachineLearning/Simulation/DriehoekSneller'
H5FILE = DIR+"/main_data_[503_504_506].h5"
new_H5FILE = DIR+"/driehoek.h5"

merge(STATIONS, H5FILE, directory=DIR)
read_sapphire_simulation(H5FILE, new_H5FILE, len(STATIONS),
                         find_mips=True,
                         uniform_dist=True)