import textwrap, sys, multiprocessing, itertools
from copy import deepcopy

import numpy as np, pandas as pd
from scipy.spatial import distance_matrix

from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
from pymoo.core.problem import Problem

from int_seq_des.wrapper import ObjectiveESM
from int_seq_des.utils import (
    sort_order, 
    get_logger, 
    sep, 
    class_seeds, 
    get_array_chunk, 
    evaluate_candidates, 
    cluster_act_single_candidate
)


logger= get_logger(__name__)

class MutationMethod(object):
    # all private methods are written to handle a single candidate at each call
    def __init__(
        self, choose_pos_method, choose_AA_method, prob,
        mutation_rate, sigma= None,
        protein_mpnn= None,
        esm_script_loc= None, esm_model= None, esm_device= None
    ):
        
        self.choose_pos_method= choose_pos_method
        self.choose_AA_method= choose_AA_method
        self.prob= prob
        self.mutation_rate= mutation_rate
        self.sigma= sigma
        self.protein_mpnn= protein_mpnn
        self.esm_script_loc= esm_script_loc
        self.esm_model= esm_model
        self.esm_device= esm_device

        self.name= f'{choose_pos_method}+{choose_AA_method}_{prob}'

    def __str__(self):
        return self.name
    
    def set_rng(self, rng):
        if hasattr(self, 'rng'):
            raise RuntimeError(
                f'The numpy rng generator for MutationMethod {str(self)} is already set.'
            )
        else:
            self.rng= rng
    
    def choose_pos(self, candidate, protein, allowed_pos_list):
        if self.choose_pos_method == 'random':
            return self._random_sampling(allowed_pos_list, self.mutation_rate)
        elif self.choose_pos_method == 'spatial_coupling':
            return self._spatially_coupled_sampling(
                protein,
                self._random_chain_picker(protein),
                allowed_pos_list,
                self.rng.permuted(allowed_pos_list),
                self.mutation_rate,
                self.sigma
            )
        elif self.choose_pos_method == 'likelihood_ESM':
            return self._esm_then_cutoff(
                ObjectiveESM(
                    chain_id= self._random_chain_picker(protein),
                    script_loc= self.esm_script_loc,
                    model_name= self.esm_model if self.esm_model is not None else 'esm1v',
                    device= self.esm_device if self.esm_device is not None else 'cpu',
                    sign_flip= False
                ),
                candidate,
                protein,
                allowed_pos_list,
                self.mutation_rate
            )
        elif self.choose_pos_method == 'ESM+spatial':
            return self._esm_then_spatial(
                ObjectiveESM(
                    chain_id= self._random_chain_picker(protein),
                    script_loc= self.esm_script_loc,
                    model_name= self.esm_model if self.esm_model is not None else 'esm1v',
                    device= self.esm_device if self.esm_device is not None else 'cpu',
                    sign_flip= False
                ),
                candidate,
                protein,
                allowed_pos_list,
                self.mutation_rate,
                self.sigma
            )
        else:
            raise ValueError('Unknown choose_pos_method')
    
    def choose_AA(self, candidate, protein, proposed_des_pos_list):
        if self.choose_AA_method == 'random':
            return self._random_resetting(candidate, protein, proposed_des_pos_list)
        elif self.choose_AA_method in ['ProteinMPNN-AD', 'ProteinMPNN-PD']:
            return self._protein_mpnn(
                self.protein_mpnn, 
                self.choose_AA_method, 
                candidate, 
                proposed_des_pos_list
            )
        else:
            raise ValueError('Unknown choose_AA_method')

    def apply(self, candidate, protein, allowed_pos_list= None):
        if allowed_pos_list is None:
            # allowed_pos_list is a list of indices of elements in the candidate array
            # by default, allow all designable positions to be redesigned
            allowed_pos_list= np.arange(protein.design_seq.n_des_res)
        
        if len(allowed_pos_list) == 0:
            raise ValueError(
                f'allowed_pos_list for MutationMethod ({str(self)}) is empty.'
            )

        return self.choose_AA(
            candidate, 
            protein, 
            self.choose_pos(candidate, protein, allowed_pos_list)
        )

    def _random_sampling(self, allowed_pos_list, mutation_rate):
        des_pos_list= []
        for pos in allowed_pos_list:
            if self.rng.random() < mutation_rate:
                des_pos_list.append(pos)

        if len(des_pos_list) == 0:
            logger.warning(
                '_random_sampling() des_pos_list returned an empty list; ' + \
                'consider increasing the mutation rate or the number of designable positions. ' + \
                'A random position will be chosen from the allowed_pos_list for mutation.'
            )
            des_pos_list= self.rng.choice(allowed_pos_list, size= 1)

        logger.debug(
            textwrap.dedent(
                f'''\
                _random_sampling() returned the following des_pos_list:
                {sep}
                allowed_pos_list:{allowed_pos_list}
                des_pos_list: {des_pos_list}
                {sep}
                '''
            )
        )
        return des_pos_list
    
    def _spatially_coupled_sampling(
        self, 
        protein, 
        chain_id, 
        allowed_pos_list, 
        hotspot_allowed_des_pos_ranked_list, 
        mutation_rate, 
        sigma
    ):
        '''
        hotspot_allowed_des_pos_ranked_list is a subset of allowed_pos_list, likely with a different ordering
        this function is complicated because each allowed position may correspond to multiple physical residues, 
        but we want to know the shortest possible distances between each pairs of allowed positions
        '''
        CA_coords_df= {}
        neighbor_chain_ids= protein.chains_dict[chain_id].neighbors_list
        designable_chain_ids= protein.design_seq.chains_to_design
        # find the interaction of the two lists
        chain_ids= list(set(neighbor_chain_ids) & set(designable_chain_ids)) 
        chain_ids.sort(key= sort_order)

        for id in chain_ids:
            CA_coords= protein.get_CA_coords(id)
            chain_allowed_pos_list= protein.candidate_to_chain_des_pos(
                candidate_des_pos_list= allowed_pos_list,
                chain_id= id,
                drop_terminal_missing_res= True,
                drop_internal_missing_res= False
            )
            CA_coords_subset= {}
            for chain_pos, candidate_pos in zip(chain_allowed_pos_list, allowed_pos_list):
                if chain_pos is None:
                    CA_coords_subset[candidate_pos]= [np.nan]*3
                else:
                    CA_coords_subset[candidate_pos]= CA_coords[chain_pos]
            CA_coords_subset= pd.Series(CA_coords_subset)

            CA_coords_df[id]= CA_coords_subset
        # each column is a chain, each row is a tied_position, and each element is a CA position
        CA_coords_df= pd.DataFrame(CA_coords_df) 
        logger.debug(
            textwrap.dedent(
                f'''\
                _spatially_coupled_sampling() returned the following CA_coords_df:
                {sep}\n{CA_coords_df}\n{sep}
                '''
            )
        )
        
        if len(chain_ids) == 1:
            # if the chosen chain has no neighbors other than itself, then compute pairwise dist within the chain
            min_all_pairwise_dist_mat= distance_matrix(
                *[np.vstack(CA_coords_df[chain_id])]*2, 
                p= 2
            )
            logger.debug(
                textwrap.dedent(
                    f'''\
                    _spatially_coupled_sampling() returned the following min_all_pairwise_dist_mat:
                    {sep}\n{min_all_pairwise_dist_mat}\n{sep}
                    '''
                )
            )
        else:
            all_pairwise_dist_mat= []
            for chain_pair in itertools.combinations_with_replacement(chain_ids, 2):
                chain_A, chain_B= chain_pair
                dist_mat= distance_matrix(
                    np.vstack(CA_coords_df[chain_A]), 
                    np.vstack(CA_coords_df[chain_B]), 
                    p= 2
                )
                min_dist_mat= np.nanmin([dist_mat, dist_mat.T], axis= 0)
                all_pairwise_dist_mat.append(min_dist_mat)
            # find the shortest possible distance between each tied_position
            min_all_pairwise_dist_mat= np.nanmin(all_pairwise_dist_mat, axis= 0) 
            logger.debug(
                textwrap.dedent(
                    f'''\
                    _spatially_coupled_sampling() returned the following min_all_pairwise_dist_mat:
                    {sep}\n{min_all_pairwise_dist_mat}\n{sep}
                    '''
                )
            )
            if not np.allclose(
                min_all_pairwise_dist_mat, 
                min_all_pairwise_dist_mat.T
            ):
                raise ValueError(
                    f'_spatially_coupled_sampling() returned a non-symmetric min_all_pairwise_dist_mat.'
                )

        min_dist_kernel_df= pd.DataFrame(
            np.exp(-min_all_pairwise_dist_mat**2/sigma**2), 
            columns= allowed_pos_list, 
            index= allowed_pos_list
        )
        logger.debug(
            textwrap.dedent(
                f'''\
                _spatially_coupled_sampling() returned the following min_dist_kernel_df:
                {sep}\n{min_dist_kernel_df}\n{sep}
                '''
            )
        )

        chain_hotspot_allowed_des_pos_ranked_list= protein.candidate_to_chain_des_pos(
            candidate_des_pos_list= hotspot_allowed_des_pos_ranked_list,
            chain_id= chain_id,
            drop_terminal_missing_res= True,
            drop_internal_missing_res= False
        )
        hotspot_dict= dict(
            zip(
                hotspot_allowed_des_pos_ranked_list, 
                chain_hotspot_allowed_des_pos_ranked_list
            )
        )

        # at least pick one position for mutation
        n_mutations= max(1, self.rng.binomial(len(allowed_pos_list), mutation_rate)) 
        des_pos_list= []
        for des_pos in hotspot_allowed_des_pos_ranked_list:
            if hotspot_dict[des_pos] == None:
                pass
            else:
                n_mutations_remaining= n_mutations - len(des_pos_list)
                if n_mutations_remaining > 0:
                    probs= min_dist_kernel_df[des_pos].to_numpy()
                    probs= np.nan_to_num(probs, nan= 0.)
                    probs= probs/probs.sum()
                    n_mutations_to_gen= min(probs.size, n_mutations_remaining)
                    new_des_pos= self.rng.choice(
                        allowed_pos_list, 
                        size= n_mutations_to_gen, 
                        replace= False, 
                        p= probs
                    ).tolist()
                    des_pos_list+= new_des_pos
                    des_pos_list= [*set(des_pos_list)] # remove duplicates
                    logger.debug(
                        textwrap.dedent(
                            f'''\
                            _spatially_coupled_sampling() returned the following des_pos_list for pos {des_pos}:
                            {sep}\n{new_des_pos}\n{sep}
                            '''
                        )
                    )
                else:
                    break
        logger.debug(
            textwrap.dedent(
                f'''\
                _spatially_coupled_sampling() returned the following combined des_pos_list:
                {sep}\n{des_pos_list}\n{sep}
                '''
            )
        )
        return des_pos_list
    
    def _likelihood_esm_score_rank(
        self, 
        objective_esm, 
        candidate, 
        protein, 
        allowed_pos_list
    ):
        chain_id= objective_esm.chain_id

        esm_scores= np.squeeze(
            objective_esm.apply([candidate], protein, position_wise= True)
        )
        chain_des_pos= protein.candidate_to_chain_des_pos(
            allowed_pos_list, 
            chain_id, 
            drop_terminal_missing_res= False, 
            drop_internal_missing_res= False
        )
        esm_scores_des_pos= []
        for des_pos in chain_des_pos:
            if des_pos is None:
                # give np.nan if the designable position does not map onto a residue in the chain
                esm_scores_des_pos.append(np.nan) 
            else:
                esm_scores_des_pos.append(esm_scores[des_pos])
        # sort in ascending order for the ESM log likelihood scores (i.e., worst to best)
        # np.argsort() will will put the np.nan positions at the end of the list, which is then removed through slicing
        esm_scores_des_pos_argsort= np.argsort(esm_scores_des_pos)
        n_nan= sum(np.isnan(esm_scores_des_pos))
        if n_nan > 0:
            esm_scores_des_pos_argsort= esm_scores_des_pos_argsort[:-n_nan]

        esm_des_pos_rank= np.asarray(allowed_pos_list)[esm_scores_des_pos_argsort]

        logger.debug(
            textwrap.dedent(
                f'''\
                _likelihood_esm_score_rank() returned the following results:
                {sep}
                chain_id: {chain_id}
                all_scores: {esm_scores}
                des_pos_scores: {esm_scores_des_pos}
                des_pos_rank: {esm_des_pos_rank}
                {sep}
                '''
            )
        )
        
        # the returned list of scores and indices may be shorter than the length of the allowed_pos_list
        return chain_id, esm_scores_des_pos, esm_des_pos_rank
    
    def _esm_then_cutoff(
        self, 
        objective_esm, 
        candidate, 
        protein, 
        allowed_pos_list, 
        mutation_rate
    ):
        chain_id, esm_scores, esm_ranks= self._likelihood_esm_score_rank(
            objective_esm, 
            candidate, 
            protein, 
            allowed_pos_list
        )
        n_des_pos= len(allowed_pos_list)
        n_mutations= max(
            1, min(
                len(esm_ranks), 
                self.rng.binomial(n_des_pos, mutation_rate)
            )
        )
        des_pos_list= list(esm_ranks[:n_mutations])
        logger.debug(
            textwrap.dedent(
                f'''\
                _esm_then_cutoff() picked {n_mutations}/{n_des_pos} sites:
                {sep}\n{des_pos_list}\n{sep}
                '''
            )
        )
        return des_pos_list

    def _esm_then_spatial(
        self, 
        objective_esm, 
        candidate, 
        protein, 
        allowed_pos_list, 
        mutation_rate, 
        sigma
    ):
        chain_id, esm_scores, esm_ranks= self._likelihood_esm_score_rank(
            objective_esm, 
            candidate, 
            protein, 
            allowed_pos_list
        )
        des_pos_list= self._spatially_coupled_sampling(
            protein, 
            chain_id, 
            allowed_pos_list, 
            esm_ranks, 
            mutation_rate, 
            sigma
        )
        logger.debug(
            textwrap.dedent(
                f'''\
                _esm_then_spatial() returned the following des pos list:
                {sep}\n{des_pos_list}\n{sep}
                '''
            )
        )
        return des_pos_list
    
    def _random_resetting(self, candidate, protein, proposed_des_pos_list):
        new_candidate= np.copy(candidate)
        for des_pos in proposed_des_pos_list:
            alphabet= list(protein.design_seq.tied_residues[des_pos].allowed_AA)
            new_candidate[des_pos]= self.rng.choice(alphabet)
        logger.debug(
            textwrap.dedent(
                f'''\
                _random_resetting() returned the following results:
                {sep}
                old_candidate: {candidate}
                des_pos_list: {proposed_des_pos_list}
                new_candidate: {new_candidate}
                {sep}
                '''
            )
        )
        return new_candidate

    def _protein_mpnn(self, protein_mpnn, method, candidate, proposed_des_pos_list):
        protein_mpnn_seed= self.rng.integers(1000000000)
        new_candidate= np.squeeze(
            protein_mpnn.design_and_decode_to_candidates(
                method, 
                candidate, 
                proposed_des_pos_list, 
                num_seqs= 1, 
                batch_size= 1, 
                seed= protein_mpnn_seed
            )
        )
        logger.debug(
            textwrap.dedent(
                f'''\
                _protein_mpnn() returned the following results:
                {sep}
                old_candidate: {candidate}
                des_pos_list: {proposed_des_pos_list}
                new_candidate: {new_candidate}
                {sep}
                '''
            )
        )
        return new_candidate

    def _random_chain_picker(self, protein):
        assert hasattr(self, 'rng'), f'{self} has no rng attr'
        # only draw from the set of designable chains
        new_chain= self.rng.choice(list(protein.design_seq.chains_to_design))
        logger.debug(
            f'_random_chain_picker() returned a new chain_id: {new_chain}\n'
        )
        return new_chain

class ProteinMutation(Mutation):
    def __init__(
        self, 
        method_list, 
        root_seed, 
        pkg_dir, 
        pop_size= None, 
        comm= None, 
        cluster_parallelization= False, 
        cluster_time_limit_str= None, 
        cluster_mem_free_str= None, 
        **kwargs
    ):
        super().__init__(prob=1.0, prob_var=None, **kwargs)
        self.uninitialized_method_list= method_list
        self.root_seed= root_seed
        self.pkg_dir= pkg_dir
        self.pop_size= pop_size
        self.comm= comm
        self.cluster_parallelization= cluster_parallelization
        self.cluster_time_limit_str= cluster_time_limit_str
        self.cluster_mem_free_str= cluster_mem_free_str
        self.class_seed= class_seeds[self.__class__.__name__]
        self.call_counter= 0 # increment this counter each time the method _do() is called; for setting RNG.

    def _do(self, problem, candidates, **kwargs):
        logger.debug(
            textwrap.dedent(
                f'''\
                ProteinMutation (call_counter = {self.call_counter}) received the following input candidates:
                {sep}\n{candidates}\n{sep}
                '''
            )
        )

        # reset RNG
        if self.cluster_parallelization:
            # it seems that if the population size is an odd number, sometimes len(candidates) > pop_size
            size= max(self.pop_size, len(candidates))
        else:
            if self.comm is None:
                size= 1
            else:
                size= self.comm.Get_size()
        self.rng_list= [
            np.random.default_rng(
                [
                    self.class_seed, 
                    rank, 
                    self.call_counter, 
                    self.root_seed
                ]
            ) for rank in range(size)
        ]

        method_list_rng_list= []
        for rank in range(size):
            method_list_rng= deepcopy(self.uninitialized_method_list)
            for method_ind, method in enumerate(method_list_rng):
                method.set_rng(
                    np.random.default_rng(
                        [
                            self.class_seed, 
                            rank, 
                            method_ind, 
                            self.call_counter, 
                            self.root_seed
                        ]
                    )
                )
            method_list_rng_list.append(method_list_rng)
        method_list= method_list_rng_list
        self.call_counter+= 1


        if self.cluster_parallelization:
            jobs= []
            result_queue= multiprocessing.Queue()
            
            for candidate_ind, candidate in enumerate(candidates):
                rank= candidate_ind
                chosen_method= self.rng_list[rank].choice(
                    method_list[rank], 
                    p= [method.prob for method in method_list[rank]]
                )
                proc= multiprocessing.Process(
                    target= cluster_act_single_candidate, 
                    args= (
                        [chosen_method], 
                        candidate, 
                        problem.protein, 
                        self.cluster_time_limit_str, 
                        self.cluster_mem_free_str, 
                        self.pkg_dir, 
                        candidate_ind, 
                        result_queue
                    )
                )
                jobs.append(proc)
                proc.start()
            
            # fetch results from the queue
            results_list= []

            for proc in jobs:
                results= result_queue.get()
                # exceptions are passed back to the main process as the result
                if isinstance(results, Exception):
                    sys.exit('%s (found in %s)' %(results, proc))
                results_list.append(results)

            for proc in jobs:
                proc.join()
            
            # unscramble the returned results
            new_results_order= [result[0] for result in results_list]
            assert sorted(new_results_order) == list(range(len(candidates))), \
                'some scores are not returned by multiprocessing!'
            results_list_sorted= sorted(zip(new_results_order, results_list))
            Xp= np.squeeze([result[1][1] for result in results_list_sorted])
            logger.debug(
                textwrap.dedent(
                    f'''\
                    ProteinMutation (cluster) received the following broadcasted Xp:
                    {sep}\n{Xp}\n{sep}
                    '''
                )
            )
            return Xp

        else:
            Xp= []

            if self.comm is None:
                rank= 0
                for candidate in candidates:
                    chosen_method= self.rng_list[rank].choice(
                        method_list[rank], 
                        p= [method.prob for method in method_list[rank]]
                    )
                    proposed_candidate= chosen_method.apply(candidate, problem.protein)
                    Xp.append(proposed_candidate)
                    logger.debug(
                        textwrap.dedent(
                            f'''\
                            ProteinMutation returned the following results:
                            {sep}
                            old_candidate: {candidate}
                            chosen_method: {str(chosen_method)}
                            new_candidate: {proposed_candidate}
                            {sep}
                            '''
                        )
                    )
                return np.asarray(Xp)
            
            else:
                rank= self.comm.Get_rank()
                size= self.comm.Get_size()
                
                candidates_subset= get_array_chunk(candidates, rank, size)
                chosen_method= self.rng_list[rank].choice(
                    method_list[rank], 
                    p= [method.prob for method in method_list[rank]]
                )
                
                for candidate in candidates_subset:
                    proposed_candidate= chosen_method.apply(candidate, problem.protein)
                    Xp.append(proposed_candidate)
                    logger.debug(
                        textwrap.dedent(
                            f'''\
                            ProteinMutation (rank {rank}/{size}) returned the following results:
                            {sep}
                            old_candidate: {candidate}
                            chosen_method: {str(chosen_method)}
                            new_candidate: {proposed_candidate}
                            {sep}
                            '''
                        )
                    )
                
                Xp= self.comm.gather(Xp, root= 0)
                if rank == 0: Xp= np.vstack(Xp)
                Xp= self.comm.bcast(Xp, root= 0)
                logger.debug(
                    textwrap.dedent(
                        f'''\
                        ProteinMutation (rank {rank}/{size}) received the following broadcasted Xp:
                        {sep}\n{Xp}\n{sep}
                        '''
                    )
                )
                
            return Xp

class ProteinSampling(Sampling):
    def __init__(self, mutation_rate, root_seed, comm= None):
        super().__init__()

        self.mutation_rate= mutation_rate
        self.comm= comm
        self.class_seed= class_seeds[self.__class__.__name__]

        self.rng= np.random.default_rng([self.class_seed, root_seed])

    def _do(self, problem, n_samples, **kwargs):
        # use only one process to generate the initial candidates
        if problem.comm == None:
            method= MutationMethod(
                choose_pos_method= 'random',
                choose_AA_method= 'random',
                mutation_rate= self.mutation_rate,
                prob= 1.0
            )
            method.set_rng(self.rng)
            WT_candidate= problem.protein.get_candidate()
            proposed_candidates= np.array(
                [
                    method.apply(WT_candidate, problem.protein) 
                    for _ in range(n_samples)
                ]
            )
            logger.debug(
                textwrap.dedent(
                    f'''\
                    ProteinSampling returned the following results:
                    {sep}
                    old_candidate: {WT_candidate}
                    new_candidates: {proposed_candidates}
                    {sep}
                    '''
                )
            )
            return proposed_candidates
        else:
            rank= self.comm.Get_rank()
            size= self.comm.Get_size()

            if rank == 0:
                method= MutationMethod(
                    choose_pos_method= 'random',
                    choose_AA_method= 'random',
                    mutation_rate= self.mutation_rate,
                    prob= 1.0
                )
                method.set_rng(self.rng)
                WT_candidate= problem.protein.get_candidate()
                proposed_candidates= np.array(
                    [
                        method.apply(WT_candidate, problem.protein) 
                        for _ in range(n_samples)
                    ]
                )
            else:
                WT_candidate= problem.protein.get_candidate()
                proposed_candidates= None

            proposed_candidates= self.comm.bcast(proposed_candidates, root= 0)
            logger.debug(
                textwrap.dedent(
                    f'''\
                    ProteinSampling (rank {rank}/{size}) returned the following broadcasted results:
                    {sep}
                    old_candidate: {WT_candidate}
                    new_candidates: {proposed_candidates}
                    {sep}
                    '''
                )
            )
            return proposed_candidates

class MultistateSeqDesignProblem(Problem):
    def __init__(
            self, 
            protein, 
            protein_mpnn_wrapper, 
            metrics_list,
            pkg_dir,
            comm= None, 
            cluster_parallelization= False, 
            cluster_parallelize_metrics= False,
            cluster_time_limit_str= None, 
            cluster_mem_free_str= None, 
            **kwargs
        ):
        self.protein= protein
        self.protein_mpnn= protein_mpnn_wrapper
        self.metrics_list= metrics_list
        self.pkg_dir= pkg_dir
        self.comm= comm
        self.cluster_parallelization= cluster_parallelization
        self.cluster_parallelize_metrics= cluster_parallelize_metrics
        self.cluster_time_limit_str= cluster_time_limit_str
        self.cluster_mem_free_str= cluster_mem_free_str

        super().__init__(
            n_var= protein.design_seq.n_des_res, 
            n_obj= len(self.metrics_list), 
            n_ieq_constr= 0, 
            xl= None, 
            xu= None, 
            **kwargs
        )

    def _evaluate(self, candidates, out, *args, **kwargs):
        scores= evaluate_candidates(
            self.metrics_list,
            candidates,
            self.protein,
            self.pkg_dir,
            comm= self.comm,
            cluster_parallelization= self.cluster_parallelization,
            cluster_parallelize_metrics= self.cluster_parallelize_metrics,
            cluster_time_limit_str= self.cluster_time_limit_str, 
            cluster_mem_free_str= self.cluster_mem_free_str
        )
        out['F']= scores