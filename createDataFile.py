from ProcessData import read_sapphire_simulation


H5FILE = "C:\\Users\\piete\\OneDrive\\Documenten\\Nikhef\\Losse " \
         "sims\\driehoekzonderfilter.h5"
new_H5FILE = "C:\\Users\\piete\\OneDrive\\Documenten\\Nikhef\\ML\\driehoek.h5"
read_sapphire_simulation(H5FILE, new_H5FILE, 3, find_mips=False)