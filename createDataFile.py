from ProcessData import read_sapphire_simulation


H5FILE = "C:\\Users\\piete\\OneDrive\\Documenten\\Nikhef\\Losse sims" \
         "\\main_data_[503_504_506].h5"
new_H5FILE = "C:\\Users\\piete\\OneDrive\\Documenten\\Nikhef\\ML\\driehoek.h5"
read_sapphire_simulation(H5FILE, new_H5FILE, 3, find_mips=True, uniform_dist=True)