import numpy as np
from multistate_methods.protein_mpnn_ga.wrapper import ObjectiveESM
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
from pymoo.core.problem import Problem


# TODO: get RNG to work with MPI properly
rng= np.random.default_rng()

class MutationMethod(object):
    #TODO: make it able to handle multiple candidates at once?
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
        #TODO: likelihood_ESM is not currently correctly implemented; need to take mutation_rate and select a subset, not just rank
        elif choose_pos_method == 'likelihood_ESM':
            self.choose_pos= lambda candidate, protein, allowed_pos_list: self._likelihood_esm_score_rank(
                ObjectiveESM(
                    '_',
                    self._random_chain_picker(protein),
                    model_name= esm_model if esm_model is not None else 'esm1v',
                    device= esm_device if esm_device is not None else 'cpu'
                ),
                [candidate],
                protein,
                allowed_pos_list
            )
        elif choose_pos_method == 'ESM+spatial':
            self.choose_pos= lambda candidate, protein, allowed_pos_list: self._esm_then_spatial(
                ObjectiveESM(
                    '_',
                    self._random_chain_picker(protein),
                    model_name= self.esm_model if esm_model is not None else 'esm1v',
                    device= self.esm_device if esm_device is not None else 'cpu'
                ),
                [candidate],
                protein,
                self._random_chain_picker(protein),
                allowed_pos_list,
                mutation_rate,
                sigma
            )
        else:
            raise ValueError('Unknown choose_pos_method')

        if choose_AA_method == 'random':
            self.choose_AA= lambda candidate, protein, proposed_des_pos_list: self._random_resetting([candidate], protein, proposed_des_pos_list)
        elif choose_AA_method in ['ProteinMPNN-AD', 'ProteinMPNN-PD']:
            self.choose_AA= lambda candidate, protein, proposed_des_pos_list: \
                self._protein_mpnn(protein_mpnn, choose_AA_method, protein_mpnn_seed, [candidate], proposed_des_pos_list)
        else:
            raise ValueError('Unknown choose_AA_method')

    def apply(self, candidate, protein, allowed_pos_list= None):
        if allowed_pos_list is None:
            allowed_pos_list= np.arange(protein.design_seq.n_des_res) # by default, allow all designable positions to be redesigned
        return np.squeeze(self.choose_AA(candidate, protein, self.choose_pos(candidate, protein, allowed_pos_list)))

    def _random_sampling(self, allowed_pos_list, mutation_rate):
        des_pos_list= []
        for pos in allowed_pos_list:
            if rng.random() < mutation_rate:
                des_pos_list.append(pos)
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
                des_pos_list+= rng.choice(allowed_pos_list, size= n_mutations_to_gen, replace= False, p= probs).tolist()
                des_pos_list= [*set(des_pos_list)] # remove duplicates
            else:
                break
        return des_pos_list
    
    def _likelihood_esm_score_rank(self, objective_esm, candidates, protein, allowed_pos_list):
        chain_id= objective_esm.chain_id

        esm_scores= objective_esm.apply(candidates, protein, position_wise= True)
        des_pos= np.asarray(protein.design_seq.chain_des_pos_dict[chain_id]) - protein.chains_dict[chain_id].init_resid
        esm_scores_des_pos= esm_scores[:, des_pos[allowed_pos_list]]
        esm_scores_des_pos_argsort= np.argsort(esm_scores_des_pos) # sort in ascending order for the negative ESM log likelihood scores (i.e., worst to best)

        esm_des_pos_rank= np.array(allowed_pos_list[mask] for mask in esm_scores_des_pos_argsort)

        return esm_scores_des_pos, esm_des_pos_rank
    
    def _esm_then_spatial(self, objective_esm, candidates, protein, chain_id, allowed_pos_list, mutation_rate, sigma):
        esm_scores, esm_ranks= ProteinMutation._likelihood_esm_score_rank(objective_esm, candidates, protein, allowed_pos_list)
        des_pos_list= ProteinMutation._spatially_coupled_sampling(protein, chain_id, allowed_pos_list, esm_ranks, mutation_rate, sigma)
        return des_pos_list
    
    def _random_resetting(self, candidates, protein, proposed_des_pos_list):
        Xp= np.copy(candidates)

        for candidate in Xp:
            for des_pos in proposed_des_pos_list:
                alphabet= list(protein.design_seq.tied_residues[des_pos].allowed_AA)
                candidate[des_pos]= rng.choice(alphabet)
        return Xp

    def _protein_mpnn(self, protein_mpnn, method, seed, candidates, proposed_des_pos_list):
        new_candidates= []
        for candidate in candidates:
            # take the 0th element since we only designed one new sequence
            new_candidate= np.squeeze(protein_mpnn.design_and_decode_to_candidates(method, candidate, proposed_des_pos_list, num_seqs= 1, batch_size= 1, seed= seed))
            new_candidates.append(new_candidate)

    def _random_chain_picker(self, protein):
        return rng.choice(list(protein.chains_dict.keys()))

class ProteinMutation(Mutation):
    def __init__(self, method_list, **kwargs):
        super().__init__(prob=1.0, prob_var=None, **kwargs)
        self.method_list= method_list

    #TODO: check that everytime you get a different des_ind
    #TODO: implement MPI
    def _do(self, problem, candidates, **kwargs):
        Xp= []
        for candidate in candidates:
            chosen_method= rng.choice(self.method_list, p= [method.prob for method in self.method_list])
            proposed_candidate= chosen_method.apply(candidate, problem.protein)
            Xp.append(proposed_candidate)

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


