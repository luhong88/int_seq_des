This folder contains the code to reproduce and analyze the CaM multistate design benchmark results, as well as the source data to reproduce all CaM analysis figures in the paper.

# Simulation code

* `config.py`: protein system and simulation setups required for all CaM design simulations. The user needs to update file/folder path strings in this file according to the local setups.
* `run_ga.py`: performs multistate CaM sequence design with NSGA-III. The script is setup to sweep through a set of mutation rates, mutation operator setups, and objective function setups with the `batch_settings_dict` variable and the `batch_ind` command line argument.
* `run_ad.py`: performs multistate CaM sequence design in ProteinMPNN and computes additional metric/objective functions for the redesigned sequences; an option is provided to score the WT sequence only.
* `pdb_files/`: Rosetta relaxed PDB files.

The design simulation results will be outputted as pickle files in the `output/` folder. `run_ga.py` is setup for calculation over a single GPU, and `run_sd.py` is setup to be submitted as SGE array jobs (pass the array job ID to `batch_ind` as a command line argument).

# Analysis code

* `analysis.ipynb`: jupyter notebook used to generate all CaM-related figures in the main text and supplementary material, except for the structure visualizations.
* `data/`: the CaM benchmark data; required to run the analysis notebook.
    * `benchmark_collated.gz.parquet`: a `pandas` DataFrame containing all single-state and multistate simulation results. The `pyarrow` package is required for `pandas` to parse the `parquet` file format.
    * `CaM_AD_pMPNN_logits/`: pMPNN logit vectors for each designable position, conditioned on the rest of the WT sequence. Each `.pt` file contains the logit vectors for all CaM states at a design position. The `.pt` files are serialized torch tensors.
    * `charge_complementarity_pdb_files/`: the PDB files/structural models for a GA[pMPNN] sequence, used to analyze how additional charge substitutions in GA[pMPNN] may impact CaM binding interfaces. The `mutant/` folder contains the structural models of the redesigned sequence, and the `wt/` folder contains the structural models of the WT sequence, but subjected to the same relax protocol.
    * `pdb_files_single_chain_trimmed`: the WT PDB files for computing pairwise TM-scores. The PDB files are trimmed to retain only the CaM chain residues 6-145; all binding partners are removed, and additional CaM chains are removed in case of CaM dimers. The `tmscore.py` script is used to compute and collect pairwise TM-scores; a compiled binary of the `TMscore` program is required to run this script. The `pairwise_tms.txt` file contains the output of `tmscore.py`.
