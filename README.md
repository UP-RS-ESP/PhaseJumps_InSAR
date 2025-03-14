# PhaseJumps_ISCE

Detecting phase jumps in unwrapped interferograms.

The code `calculate_phasejumps_from_mintpystack.py` identifies phase jumps in
an existing Mintpy stack.

The package contains three additional scripts that aid in processing InSAR
stacks. These are 
- `verify_IPF_version.py` which looks through your Sentinel1 SAFE files and identifies the IPF version. Files with IPF version below 2.6 are listed and should be removed. It has been previously suggested to not use these dates for interferogram generation.
- `create_list2exclude_interferograms_mintpy.py` removes interferograms listed
  in a file from an existing Mintpy stack and writes a new stack.
- `ESD_calculate_stats_at_burst_overlap_level.py` calculates statistics from Enhanced Spectral Diversity (ESD) files used to proper align the sences (stackSentinel.py tool from ISCE).  The script processes ESD files to extract median values, standard deviations, and coherence points, at burst overlaps levels.

## Installation
This is typically installed into an existing mintpy environment. Please see
[mintpy](https://github.com/insarlab/MintPy) to install mintpy and associated packages.

This is typicall done with
```
conda create -n mintpy
conda activate mintpy
conda install -c conda-forge mintpy
```

The Python packages used for running the phase jump detection require the
following
```
conda activate mintpy
conda install -c conda-forge xarray pandas packaging netCDF4
```

The python files `verify_IPF_version.py`, `create_list2exclude_interferograms_mintpy.py`, `ESD_calculate_stats_at_burst_overlap_level.py`, and `calculate_phasejumps_from_mintpystack.py` can be copied to a directory included in the path (or you add the PhaseJumps_ISCE directory to your path).

# 
