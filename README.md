# Integrative protein sequence design with evolutionary multiobjective optimization

Code and benchmark data associated with the preprint "An Integrative Approach to Protein Sequence Design through Multiobjective Optimization".

## Installation

Clone the repo and `pip install .` from the repo root directory. The repo can also be packaged into a `.whl` file using `python -m build --wheel` from the root directory.

### Optional dependencies

* To use the AF2Rank objective function, the following dependency requirements need to be properly configured:
    * [`colabdesign`](https://github.com/sokrypton/ColabDesign/tree/main) (version 1.1.0); see the linked github repo for installation instruction.
    * AlphaFold2 parameters, which can be downloaded [here](https://storage.googleapis.com/alphafold/alphafold_params_2022-03-02.tar). Note that this link will download the 2022-03-02 version that contains `alphafold_multimer_v2` model parameters. Newer versions of the multimer model parameters can be found through the [`alphafold`](https://github.com/google-deepmind/alphafold) github repo, but these parameters have not been tested with the current code.
    * `TMscore`; download the source code [here](https://zhanggroup.org/TM-score/TMscore.cpp) and compile with, e.g.,  `g++ -static -O3 -ffast-math -lm -o TMscore TMscore.cpp`.
    * The folder containing the unpacked AlphaFold2 parameters and the path to the compiled `TMscore` binary file will need to be passed as arguments to a `wrapper.ObjectiveAF2Rank` object.

* To use the ESM models, install the [`pgen`/`protein_gibbs_sampler`](https://github.com/seanrjohnson/protein_gibbs_sampler) package, following the instructions therein. The ESM model parameters will be downloaded automatically the first time the model is called. If the package is installed in a directory searched by `sys.path`, then the path to `likelihood_esm.py` needs to be explicitly provided to `wrapper.ObjectiveESM` and/or `ga_operator.MutationMethod`.

* To parallelize calculations using MPI, install the [`mpi4py`](https://github.com/mpi4py/mpi4py) package.

Note that this repo contains a vendorized version of [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) (version 1.0.1) and [AF2Rank](https://github.com/jproney/AF2Rank).

## RfaH benchmark
