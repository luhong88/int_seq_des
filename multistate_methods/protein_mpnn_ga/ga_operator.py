import logging, numpy as np
from multistate_methods.protein_mpnn_ga.wrapper import ObjectiveESM
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
from pymoo.core.problem import Problem

logger= logging.getLogger(__name__)
c_handler= logging.StreamHandler()
c_handler.setLevel(logging.DEBUG)
c_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(c_handler)
sep= '-'*50

# TODO: get RNG to work with MPI properly
rng= np.random.default_rng()

class MutationMethod(object):
    # all private methods are written to handle a single candidate at each call
    def __init__(
            self, choose_pos_method, choose_AA_method, prob,
            mutation_rate, sigma= None,
            protein_mpnn= None, protein_mpnn_seed= None,
            esm_model= None, esm_device= None):
        self.prob= prob

        if choose_pos_method == 'random':
            self.choose_pos= lambda candidate, protein, allowed_pos_list: self._random_sampling(allowed_pos_list, mutation_rate)
        elif choose_pos_method == 'spatial_coupling':
            self.choose_pos= lambda candidate, protein, allowed_pos_list: self._spatially_coupled_sampling(
                protein,
                self._random_chain_picker(protein),
                allowed_pos_list,
                rng.permuted(allowed_pos_list),
                mutation_rate,
                sigma)
        elif choose_pos_method == 'likelihood_ESM':
            self.choose_pos= lambda candidate, protein, allowed_pos_list: self._esm_then_cutoff(
                ObjectiveESM(
                    self._random_chain_picker(protein),
                    model_name= esm_model if esm_model is not None else 'esm1v',
                    device= esm_device if esm_device is not None else 'cpu'
                ),
                candidate,
                protein,
                allowed_pos_list,
                mutation_rate
            )
        elif choose_pos_method == 'ESM+spatial':
            self.choose_pos= lambda candidate, protein, allowed_pos_list: self._esm_then_spatial(
                ObjectiveESM(
                    self._random_chain_picker(protein),
                    model_name= self.esm_model if esm_model is not None else 'esm1v',
                    device= self.esm_device if esm_device is not None else 'cpu'
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
                self._protein_mpnn(protein_mpnn, choose_AA_method, protein_mpnn_seed, [candidate], proposed_des_pos_list)
        else:
            raise ValueError('Unknown choose_AA_method')

        self.name= f'choose_pos_method: {choose_pos_method}, choose_AA_method: {choose_AA_method}, prob: {prob}'

    def __str__(self):
        return self.name

    def apply(self, candidate, protein, allowed_pos_list= None):
        if allowed_pos_list is None:
            allowed_pos_list= np.arange(protein.design_seq.n_des_res) # by default, allow all designable positions to be redesigned
        return self.choose_AA(candidate, protein, self.choose_pos(candidate, protein, allowed_pos_list))

    def _random_sampling(self, allowed_pos_list, mutation_rate):
        des_pos_list= []
        for pos in allowed_pos_list:
            if rng.random() < mutation_rate:
                des_pos_list.append(pos)
        logger.debug(f'_random_sampling() returned the following des_pos_list:\n{sep}\n{des_pos_list}\n{sep}\n')
        return des_pos_list
    
    def _spatially_coupled_sampling(self, protein, chain_id, allowed_pos_list, hotspot_allowed_des_pos_ranked_list, mutation_rate, sigma):
        des_pos_list= []
        n_des_pos= len(allowed_pos_list)
        n_mutations= rng.binomial(n_des_pos, mutation_rate)
        dist_matrix= protein.get_CA_dist_matrices(chain_id)
        gaussian_kernel= np.exp(-dist_matrix**2/sigma**2) # assuming that the allowed_pos_list is a subset of the indices represented in the dist matrix
        for des_pos in hotspot_allowed_des_pos_ranked_list:
            n_mutations_remaining= n_mutations - len(des_pos_list)
            if n_mutations_remaining > 0:
                probs= gaussian_kernel[des_pos][allowed_pos_list]
                probs= probs/np.sum(probs)
                n_mutations_to_gen= min(len(probs), n_mutations_remaining)
                new_des_pos= rng.choice(allowed_pos_list, size= n_mutations_to_gen, replace= False, p= probs).tolist()
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
        des_pos= np.asarray(protein.design_seq.chain_des_pos_dict[chain_id]) - 1 # - 1 since we always score the full sequence with ESM, so only need to convert from 1-index to 0-index
        esm_scores_des_pos= esm_scores[des_pos[allowed_pos_list]]
        esm_scores_des_pos_argsort= np.argsort(esm_scores_des_pos) # sort in ascending order for the negative ESM log likelihood scores (i.e., worst to best)

        esm_des_pos_rank= np.asarray(allowed_pos_list)[esm_scores_des_pos_argsort]

        logger.debug(f'_likelihood_esm_score_rank() returned the following results:\n{sep}\nchain_id: {chain_id}\nall_scores: {esm_scores}\ndes_pos_scores: {esm_scores_des_pos}\ndes_pos_rank: {esm_des_pos_rank}\n{sep}\n')
        return chain_id, esm_scores_des_pos, esm_des_pos_rank
    
    def _esm_then_cutoff(self, objective_esm, candidate, protein, allowed_pos_list, mutation_rate):
        chain_id, esm_scores, esm_ranks= ProteinMutation._likelihood_esm_score_rank(objective_esm, candidate, protein, allowed_pos_list)
        n_des_pos= len(allowed_pos_list)
        n_mutations= rng.binomial(n_des_pos, mutation_rate)
        des_pos_list= list(esm_ranks[:n_mutations])
        logger.debug(f'_esm_then_cutoff() picked {n_mutations}/{n_des_pos} sites:\n{sep}\n{des_pos_list}\n{sep}\n')
        return des_pos_list

    def _esm_then_spatial(self, objective_esm, candidate, protein, allowed_pos_list, mutation_rate, sigma):
        chain_id, esm_scores, esm_ranks= ProteinMutation._likelihood_esm_score_rank(objective_esm, candidate, protein, allowed_pos_list)
        des_pos_list= ProteinMutation._spatially_coupled_sampling(protein, chain_id, allowed_pos_list, esm_ranks, mutation_rate, sigma)
        logger.debug(f'_esm_then_spatial() returned the following des pos list:\n{sep}\n{des_pos_list}\n{sep}\n')
        return des_pos_list
    
    def _random_resetting(self, candidate, protein, proposed_des_pos_list):
        new_candidate= np.copy(candidate)
        for des_pos in proposed_des_pos_list:
            alphabet= list(protein.design_seq.tied_residues[des_pos].allowed_AA)
            new_candidate[des_pos]= rng.choice(alphabet)
        logger.debug(f'_random_resetting() returned the following results:\n{sep}\nold_candidate: {candidate}\ndes_pos_list: {proposed_des_pos_list}\nnew_candidate: {new_candidate}\n{sep}\n')
        return new_candidate

    def _protein_mpnn(self, protein_mpnn, method, seed, candidate, proposed_des_pos_list):
        new_candidate= np.squeeze(protein_mpnn.design_and_decode_to_candidates(method, candidate, proposed_des_pos_list, num_seqs= 1, batch_size= 1, seed= seed))
        logger.debug(f'_protein_mpnn() returned the following results:\n{sep}\nold_candidate: {candidate}\ndes_pos_list: {proposed_des_pos_list}\nnew_candidate: {new_candidate}\n{sep}\n')
        return new_candidate

    def _random_chain_picker(self, protein):
        new_chain= rng.choice(list(protein.chains_dict.keys()))
        logger.debug(f'_random_chain_picker() returned a new chain_id: {new_chain}\n')
        return new_chain

class ProteinMutation(Mutation):
    def __init__(self, method_list, **kwargs):
        super().__init__(prob=1.0, prob_var=None, **kwargs)
        self.method_list= method_list

    #TODO: implement MPI
    def _do(self, problem, candidates, **kwargs):
        Xp= []
        for candidate in candidates:
            chosen_method= rng.choice(self.method_list, p= [method.prob for method in self.method_list])
            proposed_candidate= chosen_method.apply(candidate, problem.protein)
            Xp.append(proposed_candidate)
            logger.debug(f'ProteinMutation returned the following results:\n{sep}\nold_candidate: {candidate}\nchosen_method: {str(chosen_method)}\nnew_candidate: {proposed_candidate}\n{sep}\n')
        return np.asarray(Xp)

class ProteinSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        method= MutationMethod(
            choose_pos_method= 'random',
            choose_AA_method= 'random',
            mutation_rate= 0.05,
            prob= 1.0
        )
        WT_candidate= problem.protein.get_candidate()
        proposed_candidates= np.array([method.apply(WT_candidate, problem.protein) for _ in range(n_samples)])
        logger.debug(f'ProteinSampling returned the following results:\n{sep}\nold_candidate: {WT_candidate}\nnew_candidates: {proposed_candidates}\n{sep}\n')
        return proposed_candidates

class MultistateSeqDesignProblem(Problem):
    def __init__(self, protein, protein_mpnn_wrapper, metrics_list, **kwargs):
        self.protein= protein
        self.protein_mpnn= protein_mpnn_wrapper
        self.metrics_list= metrics_list
        super().__init__(n_var= protein.design_seq.n_des_res, n_obj= len(self.metrics_list), n_ieq_constr= 0, xl= None, xu= None, **kwargs)

    def _evaluate(self, candidates, out, *args, **kwargs):
        scores= []
        for metric in self.metrics_list:
            scores.append(metric.apply(candidates, self.protein))
        scores= np.vstack(scores).T
        out["F"] = scores


