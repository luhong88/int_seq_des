# Integrative protein sequence design with evolutionary multiobjective optimization

Code and benchmark data associated with the preprint "An Integrative Approach to Protein Sequence Design through Multiobjective Optimization".

## Installation

Clone the repo and `pip install .` from the repo root directory. The repo can also be packaged into a `.whl` file using `python -m build --wheel` from the root directory.

### Optional dependencies

* To use the AF2Rank objective function, the following dependency requirements need to be properly configured:
    * [`colabdesign`](https://github.com/sokrypton/ColabDesign/tree/main) (version 1.1.0); see the linked github repo for installation instruction.
    * AlphaFold2 parameters, which can be downloaded [here](https://storage.googleapis.com/alphafold/alphafold_params_2022-03-02.tar). Note that this link will download the 2022-03-02 version that contains `alphafold_multimer_v2` model parameters. Newer versions of the multimer model parameters can be found through the [`alphafold`](https://github.com/google-deepmind/alphafold) github repo, but these parameters have not been tested with the current code.
    * TMscore; download the source code [here](https://zhanggroup.org/TM-score/TMscore.cpp) and compile with, e.g.,  `g++ -static -O3 -ffast-math -lm -o TMscore TMscore.cpp`.
    * The folder containing the unpacked AlphaFold2 parameters and the path to the compiled `TMscore` binary file will need to be passed as arguments to a `wrapper.ObjectiveAF2Rank` object.

* To use the ESM models, install the [`pgen`/`protein_gibbs_sampler`](https://github.com/seanrjohnson/protein_gibbs_sampler) package, following the instructions therein. The ESM model parameters will be downloaded automatically the first time the model is called. If the package is not installed in a directory searched by `sys.path`, then the path to `likelihood_esm.py` needs to be explicitly provided to `wrapper.ObjectiveESM` and/or `ga_operator.MutationMethod`.

* To parallelize calculations using MPI, install the [`mpi4py`](https://github.com/mpi4py/mpi4py) package.

Note that this repo contains a vendorized version of [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) (version 1.0.1) and [AF2Rank](https://github.com/jproney/AF2Rank).


### Debugging mode

Change the line `logger.setLevel(logging.WARN)` in `utils.get_logger()` to `logger.setLevel(logging.DEBUG)` to print out debugging information.

### Parallelization

As long as `torch` and `jax` are properly configured, the code should automatically detect and utilize available GPUs. To force CPU computation, set the `device` argument to `cpu` for the relevant wrapper objects.

The code supports three modes of parallelization:
* No parallelization: all calculations are processed serially over a single Python process.
* Parallelization over MPI: to enable this, pass an `mpi4py` communicator object (e.g., `mpi4py.MPI.COMM_WORLD`) to the `comm` argument (and set the `cluster_parallelization` argument to False, if necessary), and then wrap the Python interpreter call with the MPI environment (e.g., `mpirun -np <population_size> python3 <job_script>.py`). In theory, MPI should be able to understand hybrid compute environment with multiple available CPU and GPU cores, if it has been compiled properly, but this setup has not been tested.
* Parallelization over a job scheduler: the code is setup to parallelize computations over an SGE job scheduler by allowing the main job script to generate, submit, monitor, and manage smaller SGE job scripts that contain parallelizable units of computation. Set `cluster_parallelization` (and possibly `cluster_parallelize_metrics`) to `True` to enable behavior. The user needs to modify the `utils.sge_write_submit_script()` function directly to configure the submission scripts in accordance with the available local compute environment. It should be possible to adapt the code to utilize other job schedulers, such as SLURM, by updating the appropriate commands in `utils.sge_write_submit_script()`, `utils.sge_submit_job()`, and `utils.cluster_manage_job()`.

*Caveats*
* Due to the way AlphaFold2 and ESM models are initialized, parallelization may lead to significant disk I/O activity due to the need to read in the model parameter files each time the models are (re)initialized.
* Parallelization over a job scheduler may fail using the system default tmp folder; in this case, an alternative folder for writing temporary job files can be specified with the `temp_dir` argument.

## RfaH benchmark

The code for performing RfaH benchmark design simulations is located in `/RfaH_benchmark`:
* `config.py`: protein system and simulation setups required for all design simulations. The user needs to update file/folder path strings in this file according to the local setups.
* `run_ga.py`: performs multistate RfaH sequence design with NSGA-II. The script is setup to sweep through a set of mutation rates, mutation operator setups, and objective function setups with the `batch_settings_dict` variable and the `batch_ind` command line argument.
* `run_ad.py`: performs multistate RfaH sequence design in ProteinMPNN and computes additional metric/objective functions for the redesigned sequences; an option is provided to score the WT sequence only.
* `run_sd.py`: performs single-state RfaH sequence design in ProteinMPNN and computes additional metric/objective functions for the redesigned sequences. The user needs to specify which RfaH state to redesign as a command line argument, although the metric/objective functions will be calculated for both states.

The Rosetta relaxed PDB files are located at `pdb_files/`. The design simulation results will be outputted as pickle files in the `output/` folder. `run_ga.py` is setup for parallelization over an SGE job scheduler, and `run_ad.py` and `run_sd.py` are setup to be submitted as SGE array jobs (pass the array job ID to `batch_ind` as a command line argument).

