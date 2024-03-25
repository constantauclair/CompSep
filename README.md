# CompSep : statistical component separations using Wavelet Phase Harmonics (WPH) statistics
CompSep contains Python codes to perform statistical component separations on 2D data using WPH statistics.
The codes allow to separate a unknown signal of interest from a contamination of which we have a set of realizations. \
\
The Jupyter Notebook Separation_analysis.ipynb gives some tools to analyse the results of the component separation. \
\
The Library folder contains four files: the component separation algorithm component_separation.py, the component separation functions comp_sep_functions.py, the plot tools tools.py and a component separation tutorial notebook tutorial.ipynb. \
\
The Data folder contains a map which can be used as a template for a non-Gaussian process. It comes from the total intensity Planck observation at 353 GHz of the Chameleon-Musca region. \
\
The Results folder contains two results of component_separation.py. \
\
All the codes have been written with Python 3.9.1 and using PyWPH 1.1 (https://github.com/bregaldo/pywph), NumPy 1.20.1, PyTorch 2.0.0, and SciPy 1.6.0.

