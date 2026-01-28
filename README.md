# discontinuous-layered-lead-bromide-perovskites
Python scripts to process data for ''(add reference)

Requirements

* numpy
* matplotlib
* scipy
* pandas
* statsmodels


Usage

Inside the 'data' folder, add the data found in '' (add repository). The data should be organized into the following subfolders for the analysis:
'absorption', 'anisotropic_diffusion', 'diffusion', 'double_peak'->('polarization', 'pow_dep','sp_res'), 'polarization', 'power' and 'spectra'.

After that is ready you can use an IDE to run the code or use the command window by typing: python your\_folder/process_*.py 

for the respective figures.

Note:
-  The data contained in anisotropic_diffusion were analyzed using the diffusion scripts. Scan angle directions and their corresponding diffusivity values are stored in the Python script process_anisotropic_diff, which is used to obtain the polar diffusivity plot.
 
- Some terminal printouts and graphical elements (e.g., additional legends not shown in the paper figures) have been intentionally left in the code/output for clarity.

To access the data: 
%check how write it 
