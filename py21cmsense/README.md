### py21cmsense
<br>
py21cmsense is a Python module for calculating the sensitivity of a radio interferometer the 21cm power spectrum.
For more information, please see

Parsons et al. 2012ApJ...753...81P

Pober et al. 2013AJ....145...65P

Pober et al. 2014ApJ...782...66P

The code repository can be found at https://github.com/jpober/21cmSense

Questions, comments, or feedback should be directed to jpober <at> uw <dot> edu.
If you use this code in any of your work, please acknowledge 
Pober et al. 2013AJ....145...65P and Pober et al. 2014ApJ...782...66P 
and provide a link to the code repository.

INSTALLATION:

Simple installation is to add the directory where py21cmsense lives to your PYTHONPATH.
In your .bash_profile or .bashrc file, add
```bash
#!/bin/bash 
cd <working_directory>
PYTHONPATH=<working_directory>:$PYTHONPATH
export PYTHONPATH
```

