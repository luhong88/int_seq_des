This folder contains the code to reproduce and analyze the PapD multistate design benchmark results, as well as the source data to reproduce all PapD analysis figures in the paper.

# Simulation code

* `config.py`: protein system and simulation setups required for all PapD design simulations. The user needs to update file/folder path strings in this file according to the local setups.
* `run_ga.py`: performs multistate PapD sequence design with NSGA-II. The script is setup to sweep through a set of mutation rates, mutation operator setups, and objective function setups with the `batch_settings_dict` variable and the `batch_ind` command line argument.
* `run_ad.py`: performs multistate RfaH sequence design in ProteinMPNN and computes additional metric/objective functions for the redesigned sequences; an option is provided to score the WT sequence only.
* `pdb_files/`: Rosetta relaxed PDB files.

The design simulation results will be outputted as pickle files in the `output/` folder. `run_ga.py` is setup for calculation over a single GPU, and `run_ad.py` is setup to be submitted as SGE array jobs (pass the array job ID to `batch_ind` as a command line argument).

# Analysis code

* `analysis.ipynb`: jupyter notebook used to generate all PapD-related figures in the main text and supplementary material, except for the structure visualizations.
* `data/`: the PapD benchmark data; required to run the analysis notebook.
    * `benchmark_collated.gz.parquet`: a `pandas` DataFrame containing all multistate simulation results. The `pyarrow` package is required for `pandas` to parse the `parquet` file format.