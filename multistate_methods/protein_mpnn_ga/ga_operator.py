import itertools, logging, numpy as np, pandas as pd
from scipy.spatial import distance_matrix
from multistate_methods.protein_mpnn_ga.wrapper import ObjectiveESM
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback

logger= logging.getLogger(__name__)
logger.propagate= False
logger.setLevel(logging.DEBUG)
c_handler= logging.StreamHandler()
c_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(c_handler)
sep= '-'*50

def _get_array_chunk(arr, rank, size):
    chunk_size= len(arr)/size
    if not chunk_size.is_integer():
        raise ValueError(f'It is not possible to evenly divide an array of length {len(arr)} into {size + 1} processes.')
    else:
        chunk_size= int(chunk_size)
        
    if rank < size - 1:
        return arr[rank*chunk_size:(rank + 1)*chunk_size]
    else:
        return arr[rank*chunk_size:]

# TODO: get RNG to work with MPI properly
#rng= np.random.default_rng()

class MutationMethod(object):
    # TODO: check the rate of n-point crossover
    # all private methods are written to handle a single candidate at each call
    def __init__(
            self, choose_pos_method, choose_AA_method, prob,
            mutation_rate, sigma= None,
            protein_mpnn= None,
            esm_script_loc= None, esm_model= None, esm_device= None):
        self.prob= prob
        self.rng= None

        if choose_pos_method == 'random':
            self.choose_pos= lambda candidate, protein, allowed_pos_list: self._random_sampling(allowed_pos_list, mutation_rate)
        elif choose_pos_method == 'spatial_coupling':
            self.choose_pos= lambda candidate, protein, allowed_pos_list: self._spatially_coupled_sampling(
                protein,
                self._random_chain_picker(protein),
                allowed_pos_list,
                self.rng.permuted(allowed_pos_list),
                mutation_rate,
                sigma)
        elif choose_pos_method == 'likelihood_ESM':
            self.choose_pos= lambda candidate, protein, allowed_pos_list: self._esm_then_cutoff(
                ObjectiveESM(
                    chain_id= self._random_chain_picker(protein),
                    script_loc= esm_script_loc,
                    model_name= esm_model if esm_model is not None else 'esm1v',
                    device= esm_device if esm_device is not None else 'cpu',
                    sign_flip= False
                ),
                candidate,
                protein,
                allowed_pos_list,
                mutation_rate
            )
        elif choose_pos_method == 'ESM+spatial':
            self.choose_pos= lambda candidate, protein, allowed_pos_list: self._esm_then_spatial(
                ObjectiveESM(
                    chain_id= self._random_chain_picker(protein),
                    script_loc= esm_script_loc,
                    model_name= esm_model if esm_model is not None else 'esm1v',
                    device= esm_device if esm_device is not None else 'cpu',
                    sign_flip= False
                ),
                candidate,
                protein,
                allowed_pos_list,
                mutation_rate,
                sigma
            )
        else:
            raise ValueError('Unknown choose_pos_method')

        if choose_AA_method == 'random':
            self.choose_AA= lambda candidate, protein, proposed_des_pos_list: self._random_resetting(candidate, protein, proposed_des_pos_list)
        elif choose_AA_method in ['ProteinMPNN-AD', 'ProteinMPNN-PD']:
            self.choose_AA= lambda candidate, protein, proposed_des_pos_list: \
                self._protein_mpnn(protein_mpnn, choose_AA_method, candidate, proposed_des_pos_list)
        else:
            raise ValueError('Unknown choose_AA_method')

        self.name= f'choose_pos_method: {choose_pos_method}, choose_AA_method: {choose_AA_method}, prob: {prob}'

    def __str__(self):
        return self.name
    
    def set_rng(self, rng):
        if self.rng is None:
            self.rng= rng
        else:
            raise RuntimeError(f'The rng generator for MutationMethod {str(self)} is already set.')

    def apply(self, candidate, protein, allowed_pos_list= None):
        if allowed_pos_list is None:
            # allowed_pos_list is a list of indices of elements in the candidate array
            # by default, allow all designable positions to be redesigned
            allowed_pos_list= np.arange(protein.design_seq.n_des_res)
        
        if len(allowed_pos_list) == 0:
            raise ValueError(f'allowed_pos_list for MutationMethod ({str(self)}) is empty.')

        return self.choose_AA(candidate, protein, self.choose_pos(candidate, protein, allowed_pos_list))

    def _random_sampling(self, allowed_pos_list, mutation_rate):
        des_pos_list= []
        for pos in allowed_pos_list:
            if self.rng.random() < mutation_rate:
                des_pos_list.append(pos)

        if len(des_pos_list) == 0:
            logger.warning(f'_random_sampling() des_pos_list returned an empty list; consider increasing the mutation rate or the number of designable positions. A random position will be chosen from the allowed_pos_list for mutation.')
            des_pos_list= self.rng.choice(allowed_pos_list, size= 1)

        logger.debug(f'_random_sampling() returned the following des_pos_list:\n{sep}\nallowed_pos_list:{allowed_pos_list}\ndes_pos_list: {des_pos_list}\n{sep}\n')
        return des_pos_list
    
    def _spatially_coupled_sampling(self, protein, chain_id, allowed_pos_list, hotspot_allowed_des_pos_ranked_list, mutation_rate, sigma):
        '''
        hotspot_allowed_des_pos_ranked_list is a subset of allowed_pos_list, likely with a different ordering
        this function is complicated because each allowed position corresponds to multiple physical residues, but we want to know the shortest possible distances between each pairs of allowed positions
        '''
        CA_coords_df= {}
        neighbor_chain_ids= protein.chains_dict[chain_id].neighbors_list
        designable_chain_ids= protein.design_seq.chains_to_design
        chain_ids= list(set(neighbor_chain_ids) & set(designable_chain_ids)) # find the interaction of the two lists
        chain_ids.sort()

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
        CA_coords_df= pd.DataFrame(CA_coords_df) # each column is a chain, each row is a tied_position, and each element is a CA position
        logger.debug(f'_spatially_coupled_sampling() returned the following CA_coords_df:\n{sep}\n{CA_coords_df}\n{sep}\n')
        
        if len(chain_ids) == 1:
            # if the chosen chain has no neighbors other than itself, then compute pairwise dist within the chain
            min_all_pairwise_dist_mat= distance_matrix(*[np.vstack(CA_coords_df[chain_id])]*2, p= 2)
            logger.debug(f'_spatially_coupled_sampling() returned the following min_all_pairwise_dist_mat:\n{sep}\n{min_all_pairwise_dist_mat}\n{sep}\n')
        else:
            all_pairwise_dist_mat= []
            for chain_pair in itertools.combinations_with_replacement(chain_ids, 2):
                chain_A, chain_B= chain_pair
                dist_mat= distance_matrix(np.vstack(CA_coords_df[chain_A]), np.vstack(CA_coords_df[chain_B]), p= 2)
                min_dist_mat= np.nanmin([dist_mat, dist_mat.T], axis= 0)
                all_pairwise_dist_mat.append(min_dist_mat)
            min_all_pairwise_dist_mat= np.nanmin(all_pairwise_dist_mat, axis= 0) # find the shortest possible distance between each tied_position
            logger.debug(f'_spatially_coupled_sampling() returned the following min_all_pairwise_dist_mat:\n{sep}\n{min_all_pairwise_dist_mat}\n{sep}\n')
            if not np.allclose(min_all_pairwise_dist_mat, min_all_pairwise_dist_mat.T):
                raise ValueError(f'_spatially_coupled_sampling() returned a non-symmetric min_all_pairwise_dist_mat.')

        min_dist_kernel_df= pd.DataFrame(np.exp(-min_all_pairwise_dist_mat**2/sigma**2), columns= allowed_pos_list, index= allowed_pos_list)
        logger.debug(f'_spatially_coupled_sampling() returned the following min_dist_kernel_df:\n{sep}\n{min_dist_kernel_df}\n{sep}\n')

        chain_hotspot_allowed_des_pos_ranked_list= protein.candidate_to_chain_des_pos(
            candidate_des_pos_list= hotspot_allowed_des_pos_ranked_list,
            chain_id= chain_id,
            drop_terminal_missing_res= True,
            drop_internal_missing_res= False
        )
        hotspot_dict= dict(zip(hotspot_allowed_des_pos_ranked_list, chain_hotspot_allowed_des_pos_ranked_list))
        
        n_mutations= max(1, self.rng.binomial(len(allowed_pos_list), mutation_rate)) # at least pick one position for mutation
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
                    new_des_pos= self.rng.choice(allowed_pos_list, size= n_mutations_to_gen, replace= False, p= probs).tolist()
                    des_pos_list+= new_des_pos
                    des_pos_list= [*set(des_pos_list)] # remove duplicates
                    logger.debug(f'_spatially_coupled_sampling() returned the following des_pos_list for pos {des_pos}:\n{sep}\n{new_des_pos}\n{sep}\n')
                else:
                    break
        logger.debug(f'_spatially_coupled_sampling() returned the following combined des_pos_list:\n{sep}\n{des_pos_list}\n{sep}\n')
        return des_pos_list
    
    def _likelihood_esm_score_rank(self, objective_esm, candidate, protein, allowed_pos_list):
        chain_id= objective_esm.chain_id

        esm_scores= np.squeeze(objective_esm.apply([candidate], protein, position_wise= True))
        chain_des_pos= protein.candidate_to_chain_des_pos(allowed_pos_list, chain_id, drop_terminal_missing_res= False, drop_internal_missing_res= False)
        esm_scores_des_pos= []
        for des_pos in chain_des_pos:
            if des_pos is None:
                esm_scores_des_pos.append(np.nan) # give np.nan if the designable position does not map onto a residue in the chain
            else:
                esm_scores_des_pos.append(esm_scores[des_pos])
        # sort in ascending order for the ESM log likelihood scores (i.e., worst to best)
        # np.argsort() will will put the np.nan positions at the end of the list, which is then removed through slicing
        esm_scores_des_pos_argsort= np.argsort(esm_scores_des_pos)
        n_nan= sum(np.isnan(esm_scores_des_pos))
        if n_nan > 0:
            esm_scores_des_pos_argsort= esm_scores_des_pos_argsort[:-n_nan]

        esm_des_pos_rank= np.asarray(allowed_pos_list)[esm_scores_des_pos_argsort]

        logger.debug(f'_likelihood_esm_score_rank() returned the following results:\n{sep}\nchain_id: {chain_id}\nall_scores: {esm_scores}\ndes_pos_scores: {esm_scores_des_pos}\ndes_pos_rank: {esm_des_pos_rank}\n{sep}\n')
        
        # the returned list of scores and indices may be shorter than the length of the allowed_pos_list
        return chain_id, esm_scores_des_pos, esm_des_pos_rank
    
    def _esm_then_cutoff(self, objective_esm, candidate, protein, allowed_pos_list, mutation_rate):
        chain_id, esm_scores, esm_ranks= self._likelihood_esm_score_rank(objective_esm, candidate, protein, allowed_pos_list)
        n_des_pos= len(allowed_pos_list)
        n_mutations= max(1, min(len(esm_ranks), self.rng.binomial(n_des_pos, mutation_rate)))
        des_pos_list= list(esm_ranks[:n_mutations])
        logger.debug(f'_esm_then_cutoff() picked {n_mutations}/{n_des_pos} sites:\n{sep}\n{des_pos_list}\n{sep}\n')
        return des_pos_list

    def _esm_then_spatial(self, objective_esm, candidate, protein, allowed_pos_list, mutation_rate, sigma):
        chain_id, esm_scores, esm_ranks= self._likelihood_esm_score_rank(objective_esm, candidate, protein, allowed_pos_list)
        des_pos_list= self._spatially_coupled_sampling(protein, chain_id, allowed_pos_list, esm_ranks, mutation_rate, sigma)
        logger.debug(f'_esm_then_spatial() returned the following des pos list:\n{sep}\n{des_pos_list}\n{sep}\n')
        return des_pos_list
    
    def _random_resetting(self, candidate, protein, proposed_des_pos_list):
        new_candidate= np.copy(candidate)
        for des_pos in proposed_des_pos_list:
            alphabet= list(protein.design_seq.tied_residues[des_pos].allowed_AA)
            new_candidate[des_pos]= self.rng.choice(alphabet)
        logger.debug(f'_random_resetting() returned the following results:\n{sep}\nold_candidate: {candidate}\ndes_pos_list: {proposed_des_pos_list}\nnew_candidate: {new_candidate}\n{sep}\n')
        return new_candidate

    def _protein_mpnn(self, protein_mpnn, method, candidate, proposed_des_pos_list):
        protein_mpnn_seed= self.rng.integers(10000000000)
        new_candidate= np.squeeze(protein_mpnn.design_and_decode_to_candidates(method, candidate, proposed_des_pos_list, num_seqs= 1, batch_size= 1, seed= protein_mpnn_seed))
        logger.debug(f'_protein_mpnn() returned the following results:\n{sep}\nold_candidate: {candidate}\ndes_pos_list: {proposed_des_pos_list}\nnew_candidate: {new_candidate}\n{sep}\n')
        return new_candidate

    def _random_chain_picker(self, protein):
        new_chain= self.rng.choice(list(protein.design_seq.chains_to_design)) # only draw from the set of designable chains
        logger.debug(f'_random_chain_picker() returned a new chain_id: {new_chain}\n')
        return new_chain

class ProteinMutation(Mutation):
    def __init__(self, method_list, seed, comm= None, **kwargs):
        super().__init__(prob=1.0, prob_var=None, **kwargs)

        self.method_list= method_list
        self.comm= comm
        self.class_seed= 35067127485832228718352093
        
        if self.comm is None:
            self.rng= np.random.default_rng([self.class_seed, seed])
        else:
            self.rng= np.random.default_rng([self.class_seed, self.rank, seed])
            self.rank= self.comm.Get_rank()
            self.size= self.comm.Get_size()

        for method_ind, method in enumerate(self.method_list):
            method.set_rng(np.random.default_rng([self.class_seed, self.rank, method_ind, seed]))

    def _do(self, problem, candidates, **kwargs):
        Xp= []
        if self.comm is None:
            for candidate in candidates:
                chosen_method= self.rng.choice(self.method_list, p= [method.prob for method in self.method_list])
                proposed_candidate= chosen_method.apply(candidate, problem.protein, self.rng)
                Xp.append(proposed_candidate)
                logger.debug(f'ProteinMutation returned the following results:\n{sep}\nold_candidate: {candidate}\nchosen_method: {str(chosen_method)}\nnew_candidate: {proposed_candidate}\n{sep}\n')
            return np.asarray(Xp)
        else:
            candidates_subset= _get_array_chunk(candidates, self.rank, self.size)
            chosen_method= self.rng.choice(self.method_list, p= [method.prob for method in self.method_list])
            for candidate in candidates_subset:
                proposed_candidate= chosen_method.apply(candidate, problem.protein, self.rng)
                Xp.append(proposed_candidate)
                logger.debug(f'ProteinMutation (rank {self.rank}/{self.size}) returned the following results:\n{sep}\nold_candidate: {candidate}\nchosen_method: {str(chosen_method)}\nnew_candidate: {proposed_candidate}\n{sep}\n')
            
            Xp= self.comm.gather(Xp, root= 0)
            Xp= self.comm.bcast(Xp, root= 0)
            logger.debug(f'ProteinMutation (rank {self.rank}/{self.size}) received the following broadcasted Xp:\n{sep}\n{Xp}\n{sep}\n')
            
            return np.asarray(Xp)

class ProteinSampling(Sampling):
    def __init__(self, seed, comm= None):
        super().__init_()

        self.comm= comm
        self.class_seed= 50112996148903046399

        if self.comm is None:
            self.rng= np.random.default_rng([self.class_seed, seed])
        else:
            self.rng= np.random.default_rng([self.class_seed, self.rank, seed])
            self.rank= self.comm.Get_rank()
            self.size= self.comm.Get_size()

    def _do(self, problem, n_samples, **kwargs):
        if problem.comm == None:
            method= MutationMethod(
                choose_pos_method= 'random',
                choose_AA_method= 'random',
                mutation_rate= 0.1,
                prob= 1.0
            )
            method.set_rng(self.rng)
            WT_candidate= problem.protein.get_candidate()
            proposed_candidates= np.array([method.apply(WT_candidate, problem.protein) for _ in range(n_samples)])
            logger.debug(f'ProteinSampling returned the following results:\n{sep}\nold_candidate: {WT_candidate}\nnew_candidates: {proposed_candidates}\n{sep}\n')
            return proposed_candidates
        else:
            if self.rank == 0:
                method= MutationMethod(
                    choose_pos_method= 'random',
                    choose_AA_method= 'random',
                    mutation_rate= 0.1,
                    prob= 1.0
                )
                method.set_rng(self.rng)
                WT_candidate= problem.protein.get_candidate()
                proposed_candidates= np.array([method.apply(WT_candidate, problem.protein) for _ in range(n_samples)])

            proposed_candidates= self.comm.bcast(proposed_candidates, root= 0)
            logger.debug(f'ProteinSampling (rank {self.rank}/{self.size}) returned the following broadcasted results:\n{sep}\nold_candidate: {WT_candidate}\nnew_candidates: {proposed_candidates}\n{sep}\n')
            return proposed_candidates

class MultistateSeqDesignProblem(Problem):
    def __init__(self, protein, protein_mpnn_wrapper, metrics_list, comm= None, **kwargs):
        super().__init__(n_var= protein.design_seq.n_des_res, n_obj= len(self.metrics_list), n_ieq_constr= 0, xl= None, xu= None, **kwargs)

        self.protein= protein
        self.protein_mpnn= protein_mpnn_wrapper
        self.metrics_list= metrics_list
        self.comm= comm

        if self.comm is not None:
            self.rank= self.comm.Get_rank()
            self.size= self.comm.Get_size()

    def _evaluate(self, candidates, out, *args, **kwargs):
        scores= []
        if self.comm is None:
            for metric in self.metrics_list:
                scores.append(metric.apply(candidates, self.protein))
            scores= np.vstack(scores).T
            out['F'] = scores
        else:
            candidates_subset= _get_array_chunk(candidates, self.rank, self.size)
            for metric in self.metrics_list:
                scores.append(metric.apply(candidates_subset, self.protein))
            scores= np.vstack(scores).T # reshape scores from (n_metric, n_candidate) to (n_candidate, n_metric)

            scores= self.comm.gather(scores, root= 0)
            scores= np.vstack(scores)
            scores= self.comm.bcast(scores, root= 0)
            logger.debug(f'MultistateSeqDesignProblem (rank {self.rank}/{self.size}) received the following broadcasted scores:\n{sep}\n{scores}\n{sep}\n')

            out['F'] = scores

class SavePop(Callback):

    def __init__(self, metric_list) -> None:
        super().__init__()
        self.data['pop'] = []
        self.metric_name_list= [str(metric) for metric in metric_list]

    def notify(self, algorithm):
        metrics= algorithm.pop.get('F')
        candidates= algorithm.pop.get('X')

        pop_df= pd.DataFrame(metrics, columns= self.metric_name_list)
        pop_df['candidate']= [''.join(candidate) for candidate in candidates]

        self.data['pop'].append(pop_df)

        logger.debug(f'SavePop returned the following population:\n{sep}\n{pop_df}\n{sep}\n')
