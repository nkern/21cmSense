"""
example.py
----------

- An example of using py21cmsense with a class structure.

- It is assumed one already has a "calibration file" that
contains the specifics of the telescope of interest, and that
the calibration file is in the current working directory (see below).

Directory Structure:
working_direc/
    hera127.py
    ps_no_halos_nf0.521457_z9.50_useTs0_zetaX-1.0e+00_200_400Mpc_v2

- py21cmsense requires that aipy (https://github.com/AaronParsons/aipy)
and scipy are installed.

Nicholas Kern
January 2017
"""
## Import Modules
import numpy as np
from py21cmsense import Calc_Sense

## Instantiate Class
CS = Calc_Sense()

## Make an aipy array file from a calibration file
## Analogous to mk_array_file.py
## See help(CS.make_arrayfile) for details on arguments
cal_filename    = 'hera127'
out_direc       = './'
out_filename    = 'hera127_arrayfile'
verbose         = False
CS.make_arrayfile(cal_filename, outdir=out_direc,
                    out_fname=out_filename, verbose=verbose)


## Calculate 1D Thermal Sensitivity
## Analogous to calc_sense.py
## See help(CS.calc_sense_1D) for details on arguments
array_filename  = 'hera127_arrayfile.npz'
out_direc       = './'
out_filename    = 'hera127_sense1D'
model           = 'mod'
buff            = 0.1
freq            = 0.150
eor             = 'ps_no_halos_nf0.521457_z9.50_useTs0_zetaX-1.0e+00_200_400Mpc_v2'
ndays           = 180.0
n_per_day       = 6.0
bwidth          = 0.01
nchan           = 82
no_ns           = False
CS.calc_sense_1D(array_filename, outdir=out_direc, out_fname=out_filename,
                    model=model, buff=buff, freq=freq, eor=eor, ndays=ndays,
                    n_per_day=n_per_day, bwidth=bwidth, nchan=nchan, no_ns=no_ns)


