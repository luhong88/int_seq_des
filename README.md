# Integrative protein sequence design with evolutionary multiobjective optimization

Code and benchmark data associated with the paper "[An integrative approach to protein sequence design through multiobjective optimization](https://doi.org/10.1371/journal.pcbi.1011953)".

## TL;DR

### Motivation

This package provides a demonstration of how evolutionary multiobjective optimization techniques can be used to coherently integrate multiple models into the computational protein sequence design process, by 1) directly embedding models into the mutation operator to bias sampling in the sequence space, and 2) explicitly approximating the Pareto front in a user-specified objective space. The main advantage of this approach is that it outperforms and obviates the need for post hoc filtering in a multiobjective protein design problem; we anticipate this approach to be broadly relevant for problems with complex design specifications that cannot be easily encapsulated by a single model or objective function.

### Getting started with the repo

Clone the repo and `pip install .` from the repo root directory to install the package. Take a look at the RfaH benchmark code in [`RfaH_benchmark/`](benchmarks/RfaH_benchmark/) and the docstrings in [`__init__.py`](int_seq_des/__init__.py), which provides the primary user interface for setting up a simulation.

## Installation

Besides `pip install`, the repo can also be packaged into a `.whl` file using `python -m build --wheel` from the root directory.

### Optional dependencies

* To use the AF2Rank objective function, the following dependency requirements need to be properly configured:
    * [`colabdesign`](https://github.com/sokrypton/ColabDesign/tree/main) (version 1.1.1): `pip -q install git+https://github.com/sokrypton/ColabDesign.git@v1.1.1`; see the linked repo for more information.
    * AlphaFold2 parameters, which can be downloaded [here](https://storage.googleapis.com/alphafold/alphafold_params_2022-03-02.tar). Note that this link will download the 2022-03-02 version that contains `alphafold_multimer_v2` model parameters. Newer versions of the multimer model parameters can be found through the [`alphafold`](https://github.com/google-deepmind/alphafold) github repo, but these parameters have not been tested with the current code.
    * TMscore; download the source code [here](https://zhanggroup.org/TM-score/TMscore.cpp) and compile with, e.g.,  `g++ -static -O3 -ffast-math -lm -o TMscore TMscore.cpp`.
    * The folder containing the unpacked AlphaFold2 parameters and the path to the compiled `TMscore` binary file will need to be passed as arguments to a `wrapper.ObjectiveAF2Rank` object.

* To use the ESM models, install the [`pgen`/`protein_gibbs_sampler`](https://github.com/seanrjohnson/protein_gibbs_sampler) package, following the instructions therein. The ESM model parameters will be downloaded automatically the first time the model is called.

* To parallelize calculations using MPI, install the [`mpi4py`](https://github.com/mpi4py/mpi4py) package.

Note that this repo contains a vendorized version of [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) (version 1.0.1) and [AF2Rank](https://github.com/jproney/AF2Rank).


### Debugging mode

Change the line `logger.setLevel(logging.WARN)` in `utils.get_logger()` to `logger.setLevel(logging.DEBUG)` to print out debugging information.

## Parallelization

As long as `torch` and `jax` are properly configured, the code should automatically detect and utilize available GPUs. To force CPU computation, set the `device` argument to `cpu` for the relevant wrapper objects.

The code supports three modes of parallelization:
* No parallelization: all calculations are processed serially over a single Python process. This is the recommended way to run the genetic algorithm simulations, provided that a GPU with a large enough memory to load all relevant model parameters is available.
* Parallelization over MPI: to enable this, pass an `mpi4py` communicator object (e.g., `mpi4py.MPI.COMM_WORLD`) to the `comm` argument (and set the `cluster_parallelization` argument to False, if necessary), and then wrap the Python interpreter call with the MPI environment (e.g., `mpirun -np <population_size> python3 <job_script>.py`). In theory, MPI should be able to understand hybrid compute environment with multiple available CPU and GPU cores, if it has been compiled properly, but this setup has not been tested.
* Parallelization over a job scheduler: the code is setup to parallelize computations over an SGE job scheduler by allowing the main job script to generate, submit, monitor, and manage smaller SGE job scripts that contain parallelizable units of computation. Set `cluster_parallelization` (and possibly `cluster_parallelize_metrics`) to `True` to enable behavior. The user needs to modify the `utils.sge_write_submit_script()` function directly to configure the submission scripts in accordance with the available local compute environment. It should be possible to adapt the code to utilize other job schedulers, such as SLURM, by updating the appropriate commands in `utils.sge_write_submit_script()`, `utils.sge_submit_job()`, and `utils.cluster_manage_job()`.

*Caveats*
* Due to the way AlphaFold2 and ESM models are initialized, parallelization may lead to significant disk I/O activity due to the need to read in the model parameter files each time the models are (re)initialized.
* Parallelization over a job scheduler may fail using the system default tmp folder; in this case, an alternative folder for writing temporary job files can be specified with the `temp_dir` argument.

## Benchmarks

The genetic algorithms in this repo have been benchmarked against three model systems: RfaH, PapD, and CaM. See the [benchmarks](benchmarks/) folder for more information on the data and code for the benchmark anlaysis of each of these model systems.
