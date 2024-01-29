import time, pickle

import numpy as np, pandas as pd

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from multistate_methods.protein_mpnn_ga.ga_operator import (
    ProteinSampling, MultistateSeqDesignProblem
)
from multistate_methods.protein_mpnn_ga.utils import (
    get_logger, class_seeds, LoadPop, SavePop, DumpPop
)

logger= get_logger(__name__)

def run_single_pass(
    protein,
    protein_mpnn, design_mode,
    metrics_list,
    num_seqs, protein_mpnn_batch_size,
    root_seed,
    out_file_name,
    comm= None,
    score_wt_only= False
):
    '''
    This is equivalent to calling ProteinMPNN-PD
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
    
def run_nsga2(
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
        comm= None, 
        cluster_parallelization= False, 
        cluster_parallelize_metrics= False,
        cluster_time_limit_str= None, 
        cluster_mem_free_str= None,
        restart= False, 
        init_pop_file= None, 
        init_mutation_rate= 0.1
    ):
    
    if cluster_parallelization: comm= None

    design_problem= MultistateSeqDesignProblem(
        protein, 
        protein_mpnn, 
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

    algorithm= NSGA2(
        pop_size= pop_size,
        sampling= pop_initializer,
        crossover= crossover_operator,
        mutation= mutation_operator,
        eliminate_duplicates= False
    )

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
        logger.info(f'NSGA2 total run time: {t1 - t0} s.')

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
        logger.info(f'NSGA2 total run time: {t1 - t0} s.')

        if (comm is None) or (comm.Get_rank() == 0):
            pickle.dump(
                results.algorithm.callback.data['pop'], 
                open(out_file_name + '.p', 'wb')
            )
    else:
        raise KeyError(f'Unknown saving_method {saving_method}')

def run_nsga3(
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
        comm= None, 
        cluster_parallelization= False, 
        cluster_parallelize_metrics= False,
        cluster_time_limit_str= None, 
        cluster_mem_free_str= None,
        restart= False, 
        init_pop_file= None, 
        init_mutation_rate= 0.1
    ):
    
    if cluster_parallelization: comm= None

    design_problem= MultistateSeqDesignProblem(
        protein, 
        protein_mpnn, 
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
                cluster_time_limit_str, 
                cluster_mem_free_str
            ),
            copy_algorithm= False
        )
        t1= time.time()
        logger.info(f'NSGA3 total run time: {t1 - t0} s.')

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
                cluster_time_limit_str, 
                cluster_mem_free_str
            ),
            copy_algorithm= False
        )
        t1= time.time()
        logger.info(f'NSGA3 total run time: {t1 - t0} s.')

        if (comm is None) or (comm.Get_rank() == 0):
            pickle.dump(
                results.algorithm.callback.data['pop'], 
                open(out_file_name + '.p', 'wb')
            )
    else:
        raise KeyError(f'Unknown saving_method {saving_method}')