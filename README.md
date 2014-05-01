# plat
Script to compare the effect of array platinization

(C) 2014 Benjamin Naecker

Usage
-----
`import plat` into a Python shell (e.g., IPython). The main function to run
is `plat.run_comparison`, which takes a bin-file with data before platinization
and a bin-file with data after, and computes the average power spectrum of the two.

Check out the following functions to visualize differences.

+ `plot_channels` - plots the actual data from the requested channels
+ `plot_spectra`  - plots a single spectrum from the requested channels
+ `comp_spectra`  - plot the power spectrum of the given channels before and after platinization on the same axis
   
Requirements
------------
+ Python 3
+ NumPy
+ Matplotlib
+ Pyret (Also from baccus-lab GitHub acct)
