This is a pip-installable git directory which contains the code I used during my intern at HiSPARC. Installation is simple:

```
pip install git+https://github.com/pgunnink/ProcessDataML.git
```

`Example Jupyter Notebook.ipynb`is an example notebookt that contains an example of training a neural network. This was intented to be run using Google Colab, but should run in any normal Jupyter environment.

`CustromMetrics.py` contains a custom metric that can be used during training of a neural network by Keras to display progress. It gives the mean of the inproduct between predicted and actual angle.

`DegRad.py` contains multiple convencience functions.

`MergeData.py` contains a function for merging multiple the_simulation.h5 files, which are produced by the SAPPHiRE simulation.

`PMT.py` contains a implementation of the new PMT parametrization

`PMT_tf.py` contains an implementation of the new PMT parametrization in Tensorflow

`ProcessData.py` contains a function that creates a .h5 file that is ready to be used by Keras for training.

