import pickle, numpy as np, pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

from multistate_methods.protein_mpnn_ga.ga_operator import ProteinSampling, MultistateSeqDesignProblem
from multistate_methods.protein_mpnn_ga.utils import class_seeds, LoadPop, SavePop, DumpPop

def run_single_pass(
        protein,
        protein_mpnn, design_mode,
        metrics_list,
        num_seqs, protein_mpnn_batch_size,
        root_seed,
        out_file_name,
        comm= None):
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

        design_fa, chains_to_design= protein_mpnn.design(
            method= design_mode,
            base_candidate= base_candidate,
            proposed_des_pos_list= np.arange(protein.design_seq.n_des_res),
            num_seqs= num_seqs,
            batch_size= batch_size,
            seed= rng.integers(10000000000)
        )

        design_candidates= protein_mpnn.design_seqs_to_candidates(design_fa, chains_to_design, base_candidate)

        outputs['seq']= [str(fa.seq) for fa in design_fa[1:]]
        outputs['candidate']= [''.join(candidate) for candidate in design_candidates]

        for metric in metrics_list:
            outputs[str(metric)]= metric.apply(design_candidates, protein)
        
        outputs_df= pd.DataFrame(outputs)
        pickle.dump(outputs_df, open(out_file_name + '.p', 'rb'))
    
    else:
        rank= comm.Get_rank()
        size= comm.Get_size()
        rng= np.random.default_rng([class_seed, rank, root_seed])

        chunk_size= num_seqs/size
        if not chunk_size.is_integer():
            raise ValueError(f'It is not possible to evenly divide {num_seqs} sequences into {size + 1} processes.')
        else:
            chunk_size= int(chunk_size)
        
        batch_size= min(protein_mpnn_batch_size, chunk_size)

        design_fa, chains_to_design= protein_mpnn.design(
            method= 'ProteinMPNN-PD',
            base_candidate= base_candidate,
            proposed_des_pos_list= np.arange(protein.design_seq.n_des_res),
            num_seqs= chunk_size,
            batch_size= batch_size,
            seed= rng.integers(10000000000)
        )

        design_candidates= protein_mpnn.design_seqs_to_candidates(design_fa, chains_to_design, base_candidate)
        outputs['seq']= [str(fa.seq) for fa in design_fa[1:]]
        outputs['candidate']= [''.join(candidate) for candidate in design_candidates]

        for metric in metrics_list:
            outputs[str(metric)]= metric.apply(design_candidates, protein)
        
        outputs_df= pd.DataFrame(outputs)

        outputs_df_list= comm.gather(outputs_df, root= 0)
        if rank == 0:
            outputs_df= pd.concat(outputs_df_list, ignore_index= True)
            pickle.dump(outputs_df, open(out_file_name + '.p', 'rb'))

def run_nsga2(
        protein, protein_mpnn,
        pop_size, n_generation,
        mutation_operator, crossover_operator, metrics_list,
        root_seed, out_file_name, saving_method,
        comm= None,
        restart= False, init_pop_file= None
        ):
    
    if restart:
        pop_initializer= LoadPop(init_pop_file)
    else:
        pop_initializer= ProteinSampling(root_seed= root_seed, comm= comm)

    algorithm= NSGA2(
        pop_size= pop_size,
        sampling= pop_initializer,
        crossover= crossover_operator,
        mutation= mutation_operator,
        eliminate_duplicates= False
    )

    design_problem= MultistateSeqDesignProblem(protein, protein_mpnn, metrics_list, comm)

    if saving_method == 'by_generation':
        results= minimize(
            design_problem,
            algorithm,
            ('n_gen', n_generation),
            seed= root_seed,
            verbose= False,
            callback= DumpPop(metrics_list, out_file_name),
            copy_algorithm= False
        )

    elif saving_method == 'by_termination':
        results= minimize(
            design_problem,
            algorithm,
            ('n_gen', n_generation),
            seed= root_seed,
            verbose= False,
            callback= SavePop(metrics_list),
            copy_algorithm= False
        )
        if (comm is None) or (comm.Get_rank() == 0):
            pickle.dump(results.algorithm.callback.data['pop'], open(out_file_name + '.p', 'wb'))
    else:
        raise KeyError(f'Unknown saving_method {saving_method}')