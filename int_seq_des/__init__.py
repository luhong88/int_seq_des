import time, pickle

import numpy as np, pandas as pd

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from ga_operator import (
    ProteinSampling, MultistateSeqDesignProblem
)
from utils import (
    get_logger, class_seeds, LoadPop, SavePop, DumpPop
)

logger= get_logger(__name__)

def run_single_pass(
    protein,
    protein_mpnn,
    design_mode,
    metrics_list,
    num_seqs,
    protein_mpnn_batch_size,
    root_seed,
    out_file_name,
    comm= None,
    score_wt_only= False
):
    '''
    A wrapper for calling ProteinMPNN to perform sequence design.

    Input
    -----
    protein (protein.Protein): details of the protein system and design parameters.

    protein_mpnn (wrapper.ProteinMPNNWrapper): ProteinMPNN setups as a wrapper object.

    design_mode (str): how to perform multistate design: 'ProteinMPNN-AD' to perform
    average decoding for each tied positions, or 'ProteinMPNN-PD' to perform uniform
    sampling conditioned on the Pareto front at each tied position. 'ProteinMPNN-AD'
    is the method described in the original ProteinMPNN paper, and should be used
    for single-state designs.
    
    metrics_list (list): a list of metric objects against which the designed
    sequences will be scored.
    
    num_seqs (int): how many designed sequences to generate; same as the 'num_seq_per_target'
    option in ProteinMPNN.
    
    protein_mpnn_batch_size (int): ProteinMPNN decoding batch size; same as the
    'batch_size' option in ProteinMPNN.
    
    root_seed (int): a random seed to initialize the numpy random generator.
    
    out_file_name (str): the name for the output file, without suffixes.
    
    comm (mpi4py.MPI.Comm, None): a mpi4py communicator object (e.g., mpi4py.MPI.COMM_WORLD).
    Set to None (default) to disable parallelization with mpi4py.
    
    score_wt_only (bool): set to True to score the WT sequence with the metrics
    only; set to False (default) to perform sequence design.

    Output
    -----
    None. The function writes a pd.DataFrame object to a pickle object. The DataFrame 
    always contain two columns: 'seq', for the new sequences of the designable chains,
    separated by '/', and 'candidate', for the new residues at the designable positions;
    in addition, each metric in 'metrics_list' generates a new column containing
    the corresponding scores for the designs. 
    '''
    class_seed= class_seeds['run_single_pass']
    outputs= {}
    base_candidate= protein.get_candidate()

    if design_mode not in ['ProteinMPNN-PD', 'ProteinMPNN-AD']:
        raise KeyError(f'Unknown {design_mode} mode.')

    if comm is None:
        rng= np.random.default_rng([class_seed, root_seed])
        rank= None

        t0= time.time()
        design_fa, chains_to_design= protein_mpnn.design(
            method= design_mode,
            base_candidate= base_candidate,
            proposed_des_pos_list= np.arange(protein.design_seq.n_des_res),
            num_seqs= num_seqs if not score_wt_only else 1,
            batch_size= protein_mpnn_batch_size if not score_wt_only else 1,
            seed= rng.integers(1000000000)
        )
        t1= time.time()
        logger.info(f'ProteinMPNN total run time: {t1 - t0} s.')

        if score_wt_only:
            design_candidates= [base_candidate]
            outputs['seq']= [str(design_fa[0].seq)]
        else:
            design_candidates= protein_mpnn.design_seqs_to_candidates(
                design_fa, 
                chains_to_design, 
                base_candidate
            )
            outputs['seq']= [str(fa.seq) for fa in design_fa[1:]]

        outputs['candidate']= [''.join(candidate) for candidate in design_candidates]

        for metric in metrics_list:
            t0= time.time()
            outputs[str(metric)]= metric.apply(design_candidates, protein)
            t1= time.time()
            logger.info(f'{(str(metric))} total run time: {t1 - t0} s.')
        
        outputs_df= pd.DataFrame(outputs)
        pickle.dump(outputs_df, open(out_file_name + '.p', 'wb'))
    
    else:
        rank= comm.Get_rank()
        size= comm.Get_size()
        rng= np.random.default_rng([class_seed, rank, root_seed])

        chunk_size= num_seqs/size
        if not chunk_size.is_integer():
            raise ValueError(
                f'It is not possible to evenly divide {num_seqs} sequences into {size} processes.'
            )
        else:
            chunk_size= int(chunk_size)
        
        batch_size= min(protein_mpnn_batch_size, chunk_size)

        t0= time.time()
        design_fa, chains_to_design= protein_mpnn.design(
            method= 'ProteinMPNN-PD',
            base_candidate= base_candidate,
            proposed_des_pos_list= np.arange(protein.design_seq.n_des_res),
            num_seqs= chunk_size,
            batch_size= batch_size,
            seed= rng.integers(1000000000)
        )
        t1= time.time()
        logger.info(f'ProteinMPNN (rank {rank}) total run time: {t1 - t0} s.')

        design_candidates= protein_mpnn.design_seqs_to_candidates(
            design_fa, 
            chains_to_design, 
            base_candidate
        )
        outputs['seq']= [str(fa.seq) for fa in design_fa[1:]]
        outputs['candidate']= [''.join(candidate) for candidate in design_candidates]

        for metric in metrics_list:
            t0= time.time()
            outputs[str(metric)]= metric.apply(design_candidates, protein)
            t1= time.time()
            logger.info(f'{str(metric)} (rank {rank}) total run time: {t1 - t0} s.)')
        
        outputs_df= pd.DataFrame(outputs)

        outputs_df_list= comm.gather(outputs_df, root= 0)
        if rank == 0:
            outputs_df= pd.concat(outputs_df_list, ignore_index= True)
            pickle.dump(outputs_df, open(out_file_name + '.p', 'wb'))

def run_nsga(
        protein,
        protein_mpnn,
        pop_size, 
        n_generation,
        mutation_operator, 
        crossover_operator, 
        metrics_list,
        pkg_dir, 
        root_seed, 
        out_file_name, 
        saving_method,
        observer_metrics_list= None,
        nsga_version= 2,
        comm= None, 
        cluster_parallelization= False, 
        cluster_parallelize_metrics= False,
        cluster_time_limit_str= None, 
        cluster_mem_free_str= None,
        restart= False, 
        init_pop_file= None, 
        init_mutation_rate= 0.1
    ):
    '''
    Perform sequence design using either NSGA-II or NSGA-II.

    Input
    -----
    protein (protein.Protein): details of the protein system and design parameters.

    protein_mpnn (wrapper.ProteinMPNNWrapper): ProteinMPNN setups as a wrapper object.

    pop_size (int): size of the genetic algorithm population.

    n_generation (int): how many generations/iterations before termination.

    mutation_operator (ga_operator.ProteinMutation): the mutation operator to be
    used in the genetic algorithm simulation.

    crossover_operator (pymoo.core.crossover.Crossover): the crossover operator
    to be used in the genetic algorithm simulation. The n-point crossover operator
    is given by the class pymoo.operators.crossover.pntx.PointCrossover and the
    uniform crossover operator is given by the classpymoo.operators.crossover.ux.UniformCrossover.
    
    metrics_list (list): a list of metric objects/objective functions against
    which the designed sequences will be scored.

    pkg_dir (str): the absolute path to the directory containing this package.
    
    root_seed (int): a random seed that controls all downstream random number generators.
    
    out_file_name (str): the name for the output file, without suffixes; see the
    'saving_method' argument.

    saving_method (str): how to save simulation outputs. If 'by_generation',
    then write the population at each generation to file (the output files will
    have the generation index concatenated to the end of 'out_file_name'). If
    'by_termination', then save the population at each generation to a
    pymoo.core.callback.Callback object, which will be accessible in the returned
    results object.

    observer_metrics_list (str): similar to the metrics_list argument; however,
    the metrics provided are not used to define the objective space.

    nsga_version (str, int): which version of the NSGA algorithm to call. By
    default call NSGA-II, but also supports NSGA-III, which is more appropriate
    for higher-dimensional objective spaces.
    
    comm (mpi4py.MPI.Comm, None): a mpi4py communicator object (e.g., mpi4py.MPI.COMM_WORLD).
    Set to None (default) to disable parallelization with mpi4py.

    cluster_parallelization (bool): whether to parallelize calculations for each
    candidate as a job on a compute cluster. If this is set to True, the comm 
    argument is ignored. Set to False by default.

    cluster_parallelize_metrics (bool): whether to split the calculation of each
    metric for a candidate as separate jobs on a compute cluster. False by default.

    cluster_time_limit_str (str): how much time to be allowed for each job on
    the compute cluster. Required for cluster parallelization, but by default 
    set to None. The str should be formatted in a way that can be parsed by the
    SGE job scheduler.

    cluster_mem_free_str (str): how much memory to be allocated for each job
    on the computer cluster. Required for cluster parallelization, but by default 
    set to None. The str should be formatted in a way that can be parsed by the
    SGE job scheduler.

    restart (bool): whether to initialize the simulation with a new or an existing
    sequence population. If set to True, start with an existing population, and
    the argument 'init_pop_file' is required; if False (default), start with a new
    population by performing random resetting on the WT sequence, and the argument
    init_mutation_rate is required.

    init_pop_file (str): path to a pickle file containing the initial population.
    The pickle file should contain either a pd.DataFrame or a list thereof, and
    the DataFrame should have a 'candidate' column; if a list of DataFrames is
    provided, only the last element of the list will be read. This argument is not
    used if 'restart' is set to False.

    init_mutation_rate (float): a float within [0., 1.] that controls the mutation
    rate for the random resetting operator used to initialize the population. This
    argument is not used if 'restart' is set to True.

    Output
    -----
    results (pymoo.core.result.Result): the results of the genetic algorithm
    simualation. see 'saving_method' argument for additional information.
    '''
    if cluster_parallelization: comm= None

    design_problem= MultistateSeqDesignProblem(
        protein, 
        metrics_list,
        pkg_dir,
        comm, 
        cluster_parallelization, 
        cluster_parallelize_metrics,
        cluster_time_limit_str, 
        cluster_mem_free_str)

    if restart:
        pop_initializer= LoadPop(init_pop_file)
    else:
        pop_initializer= ProteinSampling(init_mutation_rate, root_seed, comm)

    if nsga_version in ['III', '3', 3]:
        ref_dirs= get_reference_directions(
            name= 'energy',
            n_dim= len(metrics_list),
            n_points= pop_size,
            seed= 1
        )
        algorithm= NSGA3(
            ref_dirs= ref_dirs,
            pop_size= pop_size,
            sampling= pop_initializer,
            crossover= crossover_operator,
            mutation= mutation_operator,
            eliminate_duplicates= False
        )
    elif nsga_version in ['II', '2', 2]:
        algorithm= NSGA2(
            pop_size= pop_size,
            sampling= pop_initializer,
            crossover= crossover_operator,
            mutation= mutation_operator,
            eliminate_duplicates= False
        )
    else:
        raise ValueError('Unknown NSGA version number.')

    if saving_method == 'by_generation':
        t0= time.time()
        results= minimize(
            design_problem,
            algorithm,
            ('n_gen', n_generation),
            seed= root_seed if isinstance(root_seed, int) else sum(root_seed),
            verbose= False,
            callback= DumpPop(
                protein, 
                metrics_list, 
                observer_metrics_list, 
                out_file_name, 
                pkg_dir, 
                comm, 
                cluster_parallelization, 
                cluster_parallelize_metrics, 
                cluster_time_limit_str, 
                cluster_mem_free_str
            ),
            copy_algorithm= False
        )
        t1= time.time()
        logger.info(f'NSGA-{nsga_version} total run time: {t1 - t0} s.')

    elif saving_method == 'by_termination':
        t0= time.time()
        results= minimize(
            design_problem,
            algorithm,
            ('n_gen', n_generation),
            seed= root_seed if isinstance(root_seed, int) else sum(root_seed),
            verbose= False,
            callback= SavePop(
                protein, 
                metrics_list, 
                observer_metrics_list, 
                pkg_dir, 
                comm, 
                cluster_parallelization, 
                cluster_parallelize_metrics, 
                cluster_time_limit_str, 
                cluster_mem_free_str
            ),
            copy_algorithm= False
        )
        t1= time.time()
        logger.info(f'NSGA-{nsga_version} total run time: {t1 - t0} s.')

        if (comm is None) or (comm.Get_rank() == 0):
            pickle.dump(
                results.algorithm.callback.data['pop'], 
                open(out_file_name + '.p', 'wb')
            )
    else:
        raise KeyError(f'Unknown saving_method {saving_method}')

    return results