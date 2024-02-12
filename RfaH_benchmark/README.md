This folder contains the code to reproduce and analyze the RfaH multistate design benchmark results, as well as the source data to reproduce all analysis figures in the paper.

# Simulation code

* `config.py`: protein system and simulation setups required for all design simulations. The user needs to update file/folder path strings in this file according to the local setups.
* `run_ga.py`: performs multistate RfaH sequence design with NSGA-II. The script is setup to sweep through a set of mutation rates, mutation operator setups, and objective function setups with the `batch_settings_dict` variable and the `batch_ind` command line argument.
* `run_ad.py`: performs multistate RfaH sequence design in ProteinMPNN and computes additional metric/objective functions for the redesigned sequences; an option is provided to score the WT sequence only.
* `run_sd.py`: performs single-state RfaH sequence design in ProteinMPNN and computes additional metric/objective functions for the redesigned sequences. The user needs to specify which RfaH state to redesign as a command line argument, although the metric/objective functions will be calculated for both states.
* `pdb_files/`: Rosetta relaxed PDB files.

The design simulation results will be outputted as pickle files in the `output/` folder. `run_ga.py` is setup for parallelization over an SGE job scheduler, and `run_ad.py` and `run_sd.py` are setup to be submitted as SGE array jobs (pass the array job ID to `batch_ind` as a command line argument).

# Analysis code

* `analysis.ipynb`: jupyter notebook used to generate all figures in the main text and supplementary material, except for the structure visualizations.
* `data/`: the RfaH benchmark data; required to run the analysis notebook.
    * `benchmark_collated.gz.parquet`: a `pandas` DataFrame containing all single-state and multistate simulation results. The `pyarrow` package is required for `pandas` to parse the `parquet` file format.
    * `41467_2022_31532_MOESM3_ESM.csv`: A NusG-like sequence database containing computational foldswitching predictions; retrieved from the supplementary materials of the paper [Many dissimilar NusG protein domains switch between α-helix and β-sheet folds](https://www.nature.com/articles/s41467-022-31532-9#Sec26).
    * `idmapping_active_true_2023_12_17.fasta` and `porter_full_seqs_clustal_omega.fa`: intermediate analysis sequence files; see the jupyter notebook for more information.
