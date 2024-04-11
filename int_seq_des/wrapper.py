import textwrap, os, io, sys, subprocess, tempfile, time, importlib

import numpy as np, pandas as pd
from Bio import SeqIO

from int_seq_des.protein import (
    DesignedProtein, SingleStateProtein, Residue, TiedResidue
)
from int_seq_des.utils import (
    sort_order, npz_to_dict, get_logger, sep, Device
)
from int_seq_des import pmpnn

logger= get_logger(__name__)

class ObjectiveAF2Rank(object):
    '''
    A wrapper for the AF2Rank method via ColabDesign. Sequences for protein 
    complexes will be concatenated.
    '''
    def __init__(
        self, 
        chain_ids, 
        template_file_loc, 
        tmscore_exec, 
        params_dir, 
        score_term= 'composite', 
        device= 'cpu', 
        sign_flip= True, 
        use_surrogate_tied_residues= False,
        persistent= False,
        haiku_parameters_dict= None
    ):
        '''
        Input
        -----
        chain_ids (list): a list of chain ID str corresponding to the chains in
        the template PDB file.

        template_file_loc (str): path to the tempate PDB file.

        tmscore_exec (str): path to the TMscore binary.

        params_dir (str): path to the folder containing AlphaFold2 parameter files.

        score_term (str): a score term from the AF2Rank output to be used as the
        metric/objective function; allowed score terms include 'plddt', 'pae',
        'rmsd_io', 'tm_io', or 'composite' (default).

        device (str): where to perform AF2Rank calculations. Set to 'cpu' to force
        calculations on the CPUs, otherwise the argument has no effect.

        sign_flip (bool): whether to multiply the score by -1. By default set to
        True so that the metric can be used in a minimization problem.

        use_surrogate_tied_residues: set to True for single-state scoring; False
        by default.

        persistent (bool): set to True to initialize the AF2 model in __init__(),
        otherwise the model will be initialized in apply(). Useful if running
        the simulation in serial and needs to reuse the same objective function
        object multiple times.

        haiku_parameters_dict (dict): a dictionary where the keys are AF2 model
        parameter names and the values are the corresponding model parameters.
        Useful for pre-loading AF2 model parameters if the the same objective
        function object will be reused multiple times.
        '''
        multimer= True if len(chain_ids) > 1 else False
        # note that the multimer params version might change in the future,
        # depending on alphafold-multimer and colabfold developments.
        model_name= 'model_1_multimer_v3' if multimer else 'model_1_ptm'

        self.chain_ids= chain_ids
        self.chain_ids.sort(key= sort_order)
        self.use_surrogate_tied_residues= use_surrogate_tied_residues
        self.score_term= score_term
        self.settings= {
            'rm_seq': True, # mask_sequence
            'rm_sc': True, # mask_sidechains
            'rm_ic': False, # mask_interchain
            'recycles': 1, 
            'iterations': 1, 
            'model_name': model_name
        }
        
        self.sign_flip= sign_flip
        self.name= ('neg_' if sign_flip else '') + \
            f'af2rank_{score_term}_chain_{"".join(self.chain_ids)}_{model_name}'
        self.device= device
        self.template_file_loc= template_file_loc
        self.tmscore_exec= tmscore_exec
        self.params_dir= params_dir
        self.persistent= persistent
        self.haiku_parameters_dict= haiku_parameters_dict

        if self.persistent:
            with Device(self.device):
                from int_seq_des.af2rank import af2rank
                self.model= af2rank(
                    pdb= self.template_file_loc,
                    chain= ','.join(self.chain_ids),
                    model_name= self.settings['model_name'],
                    tmscore_exec= self.tmscore_exec,
                    params_dir= self.params_dir,
                    haiku_parameters_dict= self.haiku_parameters_dict,
                    model_names= [self.settings['model_name']]
                )
        
    def __str__(self):
        return self.name
        
    def apply(self, candidates, protein):
        '''
        Input
        -----
        candidates (list[list[str]]): a list of design candidates. A candidate is 
        a list of residues at the designable positions.
        
        protein (protein.Protein): details of the protein system and design parameters.

        Output
        -----
        neg_output (np.ndarray[float]): a (N,) array containing the scores for the N 
        input candidates.
        '''
        full_seqs= []
        for candidate in candidates:
            full_seq= ''
            for chain_id in self.chain_ids:
                # by default, all missing residues are ignored, 
                # and multiple chains are concatenated as if there were only one continuous chain with no breaks
                full_seq+= protein.get_chain_full_seq(
                    chain_id, 
                    candidate, 
                    drop_terminal_missing_res= True, 
                    drop_internal_missing_res= True, 
                    use_surrogate_tied_residues= self.use_surrogate_tied_residues
                )
            full_seqs.append(full_seq)
        logger.debug(
            textwrap.dedent(
                f'''\
                AF2Rank (device: {self.device}, name: {self.name}) called with the sequences:
                {sep}\n{full_seqs}\n{sep}
                '''
            )
        )
        output= []
        with Device(self.device):
            if self.persistent:
                model= self.model
            else:
                from int_seq_des.af2rank import af2rank
                model= af2rank(
                    pdb= self.template_file_loc,
                    chain= ','.join(self.chain_ids),
                    model_name= self.settings['model_name'],
                    tmscore_exec= self.tmscore_exec,
                    params_dir= self.params_dir,
                    haiku_parameters_dict= self.haiku_parameters_dict,
                    model_names= [self.settings['model_name']]
                )
            for seq_ind, seq in enumerate(full_seqs):
                t0= time.time()
                output_dict= model.predict(
                    seq= seq, 
                    **self.settings, 
                    output_pdb= None, 
                    extras= {'id': seq_ind}, 
                    verbose= False
                )
                t1= time.time()
                logger.info(
                    f'AF2Rank (device: {self.device}, name: {self.name}) run time: {t1 - t0} s.'
                )
                logger.debug(
                    textwrap.dedent(
                        f'''\
                        AF2Rank (device: {self.device}, name: {self.name}) output:
                        {sep}\n{output_dict}\n{sep}
                        '''
                    )
                )
                output.append(output_dict[self.score_term])
        output= np.asarray(output)
        # take the negative because the algorithm expects a minimization problem
        neg_output= -output if self.sign_flip else output 

        logger.debug(
            textwrap.dedent(
                f'''\
                AF2Rank (device: {self.device}, name: {self.name}) final output:
                {sep}\n{neg_output}\n{sep}
                '''
            )
        )

        return neg_output

class ObjectiveESM(object):
    '''
    A wrapper for the pgen package that can be used to score sequences with ESM
    models.

    TODO: support multichain protein complexes by concatenating their sequences.
    '''
    def __init__(
        self, 
        chain_id, 
        model_name= 'esm1v', 
        device= 'cpu', 
        sign_flip= True,
        esm_sampler_dict= None
    ):
        '''
        Input
        -----
        chain_id (str): the ID for the chain to be scored.

        script_loc (str, None): path to the 'likelihood_esm.py' file in the 'pgen'
        package. If None, will attempt to search for the module.

        model_name (str): name of an ESM model; accepted values are 'esm1b',
        'esm6', 'esm12', 'esm34', and 'esm1v' (default).

        device (str): where to perform ESM calculations. Set to 'cpu' to force
        calculations on the CPUs, otherwise the argument has no effect.

        sign_flip (bool): whether to multiply the score by -1. By default set to
        True so that the metric can be used in a minimization problem.

        esm_sampler_dict (dict[str, pgen.esm_sampler.ESM_sampler]): a dictionary
        where the keys are ESM model names and the values are the ESM_sampler objects.
        If this is provided, then apply() will no longer initializing new models
        each time it is called.
        '''
        self.chain_id= chain_id
        self.model_name= model_name
        self.esm_sampler_dict= esm_sampler_dict
        self.device= device
        self.sign_flip= sign_flip
        self.name= ('neg_' if sign_flip else '') + f'{model_name}_chain_{chain_id}'

    def __str__(self):
        return self.name
    
    def apply(
        self,
        candidates, 
        protein, 
        position_wise= False,
    ):
        '''
        Input
        -----
        candidates (list[list[str]]): a list of design candidates. A candidate is 
        a list of residues at the designable positions.
        
        protein (protein.Protein): details of the protein system and design parameters.

        position_wise (bool): set to True to return (psueo)likelihood scores for
        each residue position on the input chain; set to False (default) to return 
        a single score for the input chain, which is calculated as the mean of 
        the per-position scores.

        Output
        -----
        neg_output_arr (np.ndarray[float]): a (N, L) array if 'position_wise' is True,
        or a (N,) array if 'position_wise' is False; here, N is the number of input
        candidates containing and L is the length of the target chain.
        '''
        from pgen.esm_sampler import ESM_sampler
        from pgen.likelihood_esm import model_map

        input_seqs= []
        for candidate in candidates:
            full_seq= protein.get_chain_full_seq(
                self.chain_id, 
                candidate, 
                drop_terminal_missing_res= False, 
                drop_internal_missing_res= False
            )
            input_seqs.append(full_seq)

        t0= time.time()        
        if self.esm_sampler_dict is not None:
            sampler= self.esm_sampler_dict[self.model_name]
        else:
            model= model_map[self.model_name]()
            sampler= ESM_sampler(model, device= self.device)
        scores_iter= sampler.log_likelihood_batch(
            input_seqs, 
            with_masking= False, 
            mask_distance= float("inf"), 
            batch_size= 1
        )

        output_scores= []
        output_positional_scores= []
        for score, positional_scores in scores_iter:
            output_scores.append(score)
            output_positional_scores.append(positional_scores)

        t1= time.time()
        logger.info(
            f'ESM (device: {self.device}, name= {self.name}, position_wise) run time: {t1 - t0} s.\n'
        )

        if position_wise:
            output_arr= np.asarray(output_positional_scores)
        else:
            output_arr= np.asarray(output_scores)
        # take the negative because the algorithm expects a minimization problem
        neg_output_arr= -output_arr if self.sign_flip else output_arr 

        logger.debug(
            textwrap.dedent(
                f'''\
                ESM (device: {self.device}, name= {self.name}) apply() returned the following results:
                {sep}\n{neg_output_arr}\n{sep}
                '''
            )
        )

        return neg_output_arr

class ObjectiveDebug(object):
    '''
    A sham objective function class that "scores" candidiates with random numbers.
    Useful for debugging.
    '''
    def __init__(self):
        self.name= 'debug'

    def __str__(self):
        return self.name

    def apply(self, candidates, protein):
        results= np.random.rand(len(candidates))
        logger.debug(
            textwrap.dedent(
                f'''\
                {str(self)} objective returned the following results:
                {sep}\n{results}\n{sep}
                '''
            )
        )
        return results
    
class ObjectiveCombine(object):
    '''
    An objective function that can combine other metrics/objective functions
    using a user-specified function.

    objective functions used to define an ObjectiveCombine objects will be called
    sequentially and parallelization of this proces is currently not implemented.
    '''
    def __init__(self, objectives, combine_fxn):
        '''
        Input
        -----
        objectives (list): a list of metric/objective function objects; these objects
        need to have an apply() method that takes in 'candidates' and 'protein'
        as input arguments.

        combine_fxn (callable): a function that take in an (N,) array-like and
        returns a single value, where N is the number of objectives.
        '''
        self.objectives= objectives
        self.combine_fxn= combine_fxn
        self.name= f'combine_{"+".join([str(objective) for objective in objectives])}_using_{combine_fxn.__name__}'

    def __str__(self):
        return self.name

    def apply(self, candidates, protein):
        '''
        Input
        -----
        candidates (list[list[str]]): a list of design candidates. A candidate is 
        a list of residues at the designable positions.
        
        protein (protein.Protein): details of the protein system and design parameters.

        Output
        -----
        results (np.ndarray[float]): a (N,) array containing the scores for the N 
        input candidates.
        '''
        logger.debug(
            textwrap.dedent(
                f'''\
                {str(self)} objective called with the candidates:
                {sep}\n{candidates}\n{sep}
                '''
            )
        )
        obj_res= np.array(
            [
                objective.apply(candidates, protein)
                for objective in self.objectives
            ]
        ).T
        results= np.array([self.combine_fxn(res) for res in obj_res])
        logger.debug(
            textwrap.dedent(
                f'''\
                {str(self)} objective received the following objectives and returned the following results:
                {sep}\n{results}\n{sep}
                '''
            )
        )
        return results
    
class ProteinMPNNWrapper(object):
    '''
    A wrapper for ProteinMPNN.

    Note that the auxiliary json input files are handled by the protein.Protein
    class rather than this class.
    '''
    def __init__(
        self, 
        protein, 
        temp,
        model_weights_loc,
        device= 'cpu', 
    ):
        '''
        Input
        -----
        protein (protein.Protein): details of the protein system and design parameters.

        temp (float): ProteinMPNN sampling temperature; equivalent to the 'sampling_temp'
        parameter in ProteinMPNN. Inputting a list of temperatures is not supported.

        model_weights_loc (str): path to the folder containing the desired ProteinMPNN
        weight parameter files.

        device (str): where to perform ProteinMPNN calculations. Set to 'cpu' to 
        force calculations on the CPUs, otherwise the argument has no effect.
        '''
        self.protein= protein
        self.path_to_model_weights= model_weights_loc
        self.sampling_temp= str(temp)
        self.device= device

    def design(
        self, 
        method, 
        base_candidate, 
        proposed_des_pos_list, 
        num_seqs, 
        batch_size, 
        seed= None
    ):
        '''
        Perform sequence design with ProteinMPNN.

        Input
        -----
        method (str): how to perform multistate design: 'ProteinMPNN-AD' to perform
        average decoding for each tied positions, or 'ProteinMPNN-PD' to perform uniform
        sampling conditioned on the Pareto front at each tied position. 'ProteinMPNN-AD'
        is the method described in the original ProteinMPNN paper, and should be used
        for single-state designs.

        base_candidate (list[str]): the WT/parental sequence represented as a 
        candidate. A candidate is a list of residues at the designable positions.

        proposed_des_pos_list (list[int]): a list containing a subset of the
        designable positions; the elements correspond to the 0-indices of elements
        in a candidate.

        num_seqs (int): how many sequences to design; equivalent to the
        'num_seq_per_target' argument in ProteinMPNN.

        batch_size (int): ProteinMPNN decoding batch size; same as the 'batch_size' 
        option in ProteinMPNN.

        seed (int, None): the random seed for ProteinMPNN; equivalent to the 'seed'
        argument in ProteinMPNN.

        Output
        -----
        des_seq_list (list[str]): a list of designed sequences as strings.

        chains_to_design (np.ndarray[str]): an array containing the chains that
        are redesigned; i.e., the chains must contain residues listed in the
        'proposed_des_pos_list'. Note that this array is not necessarily the
        same as the list of all designable chains specified in the self.protein
        object, because it is possible to select a subset of design positions
        that do not map onto some of the designable chains.
        '''
        model_info_dict= pmpnn.load_model(
            seed= seed if seed is not None else 0,
            path_to_model_weights= self.path_to_model_weights,
            force_cpu= self.device == 'cpu'
        )

        if method not in ['ProteinMPNN-AD']:
            raise ValueError('Invalid method definition.')
        
        designed_protein= DesignedProtein(
            self.protein, base_candidate, proposed_des_pos_list
        )

        t0= time.time()
        des_seq_list= pmpnn.design_proteins(
            model_info_dict= model_info_dict,
            parsed_pdbs= designed_protein.parsed_pdb_json,
            num_seq_per_target= num_seqs,
            batch_size= batch_size,
            sampling_temp= self.sampling_temp,
            chain_id_dict= designed_protein.parsed_fixed_chains,
            fixed_positions_dict= designed_protein.parsed_fixed_positions,
            tied_positions_dict= designed_protein.parsed_tied_positions
        )
        t1= time.time()

        logger.info(
            f'ProteinMPNN (device: {self.device}) run time: {t1 - t0} s.'
        )

        return des_seq_list, designed_protein.design_seq.chains_to_design
    
    def design_seqs_to_candidates(
        self, 
        des_seq_list, 
        candidate_chains_to_design, 
        base_candidate
    ):
        '''
        convert a list of designed sequences to a list of candidates. A candidate 
        is a list of residues at the designable positions.

        Input
        -----
        fa_records (list[str]): a list of designed sequences as strings.

        candidate_chains_to_design (np.array[str]): an array containing the chains
        that are redesigned.

        base_candidate (list[str]): the WT/parental sequence represented as a 
        candidate.

        Output
        -----
        candidates (np.array([str])): an (N, L) array containing the designed
        sequences as candidates; here, N is the number of designed sequences,
        and L is the length of a candidate.
        '''
        AA_locator= []
        for tied_res in self.protein.design_seq.tied_residues:
            if isinstance(tied_res, TiedResidue):
                # use the first residue in a tied_residue as the representative
                rep_res= tied_res.residues[0]
            elif isinstance(tied_res, Residue):
                rep_res= tied_res
                
            chain_id= rep_res.chain_id
            resid= rep_res.resid
            # the offset here is due to ProteinMPNN not having the missing terminal res
            res_ind= resid - self.protein.chains_dict[chain_id].init_resid 
            AA_locator.append([chain_id, res_ind])

        seq_list= []
        for seq in des_seq_list:
            # ProteinMPNN only output sequences of the designable chains
            seq_dict= dict(zip(candidate_chains_to_design, seq.split('/'))) 
            seq_list.append(seq_dict)
        
        candidates= []
        # skip the first element in seq_list, since ProteinMPNN will always output 
        # the input sequence as the first output
        for seq in seq_list[1:]:
            candidate= base_candidate.copy()
            for candidate_ind, (chain_id, res_ind) in enumerate(AA_locator):
                if chain_id in candidate_chains_to_design:
                    candidate[candidate_ind]= seq[chain_id][res_ind]
            candidates.append(candidate)
        logger.debug(
            textwrap.dedent(
                f'''\
                ProteinMPNN design_seqs_to_candidates() input:
                {sep}\n{seq_list[1:]}\n{sep}
                output:
                \n{sep}\n{candidates}\n{sep}
                '''
            )
        )
        candidates= np.asarray(candidates)
        
        return candidates
    
    def design_and_decode_to_candidates(
        self, 
        method, 
        base_candidate, 
        proposed_des_pos_list, 
        num_seqs, 
        batch_size, 
        seed= None
    ):
        '''
        A utility function that calls design() and pass the outputs to
        design_seqs_to_candidates(); see the docstrings of these two functions
        for more information.
        '''
        des_seq_list, candidate_chains_to_design= self.design(
            method, 
            base_candidate, 
            proposed_des_pos_list, 
            num_seqs, 
            batch_size, 
            seed
        )
        candidates= self.design_seqs_to_candidates(
            des_seq_list, candidate_chains_to_design, base_candidate
        )
        return candidates

    def score(
        self, 
        scoring_mode, 
        chains_sublist, 
        pdb_file_name, 
        candidates= None, 
        num_seqs= 1, 
        batch_size= 1, 
        seed= None, 
        use_surrogate_tied_residues= False
    ):
        '''
        Use ProteinMPNN to score sequences.

        Note here that the temperature setting has no effect on the output scores.

        Input
        -----
        scoring_mode (str): a ProteinMPNN scoring mode; accepted values are
        'score_only', 'conditional_probs_only', 'conditional_probs_only_backbone', 
        and 'unconditional_probs_only'. See ProteinMPNN for more details. Note
        that only the 'score_only' mode accepts 'candidiates', this is because
        only the 'score_only' mode is configured in ProteinMPNN to take in an
        input FASTA file.

        chains_sublist (list[str]): a list of chain IDs; should match the content
        of 'pdb_file_name'.

        pdb_file_name (str): path to the PDB file containing the structure against
        which the sequences will be scored.

        candidates (list[list[str]], None): a list of candidates to be scored. If
        None (default), then score the sequence(s) in the 'pdb_file_name'.

        num_seqs (int): how many times to score the sequences; default to 1, but
        a higher number reduces stochasticity in the output scores; equivalent
        to the 'num_seq_per_target' argument in ProteinMPNN.

        batch_size (int): ProteinMPNN decoding batch size; same as the 'batch_size' 
        option in ProteinMPNN. Default to 1.

        seed (int, None): the random seed for ProteinMPNN; equivalent to the 'seed'
        argument in ProteinMPNN.

        use_surrogate_tied_residues (bool): set to True for single-state scoring; 
        False by default.

        Output
        -----
        outputs (list[dict]): a (N,) list of dictionaries containing the ProteinMPNN
        .npz score results. If 'candidates' is not None, then N is the number of
        candidates; otherwise N = 1.
        '''
        model_info_dict= pmpnn.load_model(
            seed= seed if seed is not None else 0,
            path_to_model_weights= self.path_to_model_weights,
            force_cpu= self.device == 'cpu'
        )

        ss_protein= SingleStateProtein(
            self.protein, 
            chains_sublist, 
            pdb_file_name, 
            use_surrogate_tied_residues
        )
        
        if scoring_mode not in [
            'score_only', 
            'conditional_probs_only', 
            'conditional_probs_only_backbone', 
            'unconditional_probs_only'
        ]:
            raise ValueError(f'Unrecognized scoring_mode {scoring_mode}')
        if candidates is not None and scoring_mode in [
            'conditional_probs_only', 
            'conditional_probs_only_backbone', 
            'unconditional_probs_only'
        ]:
            raise NotImplementedError()
        
        if scoring_mode == 'score_only' and candidates is not None:
            out_dir= tempfile.TemporaryDirectory()
            input_seqs= ss_protein.candidates_to_full_seqs(candidates)
            input_seqs_f= f'{out_dir.name}/input_seqs.fa'
            with open(input_seqs_f, 'w') as f:
                for seq_ind, seq in enumerate(input_seqs):
                    f.write(f'>des_seq_{seq_ind}\n{seq}\n')
        else:
            input_seqs_f= ''

        t0= time.time()
        outputs= pmpnn.score_protein(
            model_info_dict= model_info_dict,
            parsed_pdbs= ss_protein.parsed_pdb_json,
            score_mode= scoring_mode,
            path_to_fasta= input_seqs_f,
            num_seq_per_target= num_seqs,
            batch_size= batch_size,
            chain_id_dict= ss_protein.parsed_fixed_chains,
            fixed_positions_dict= ss_protein.parsed_fixed_positions,
            tied_positions_dict= ss_protein.parsed_tied_positions
        )
        t1= time.time()
                
        logger.info(
            f'ProteinMPNN (device: {self.device}) run time: {t1 - t0} s.'
        )

        if scoring_mode == 'score_only':
            if candidates is None:
                return outputs['pdb']
            else:
                out_dir.cleanup()
                return outputs['fasta']
        else:
            return outputs
        

class ObjectiveProteinMPNNNegLogProb(object):
    '''
    A class for scoring sequences with ProteinMPNN negative log likelihood scores;
    essentially a wrapper of ProteinMPNNWrapper.score().
    '''
    def __init__(
        self, 
        chain_ids, 
        pdb_file_name, 
        score_type, 
        model_weights_loc, 
        num_seqs= 10, 
        device= 'cpu', 
        sign_flip= False, 
        use_surrogate_tied_residues= False
    ):
        '''
        Input
        -----
        chain_ids (list[str]): a list of chain IDs; should match the content
        of 'pdb_file_name'.

        pdb_file_name (str): path to the PDB file containing the structure against
        which the sequences will be scored.

        score_type (str): set to 'designable_positions' to compute a per-sequence
        score by averaging over scores at the designable positions; set to 
        'all_positions' to average over all positions in the input PDB.

        model_weights_loc (str): path to the folder containing the desired ProteinMPNN
        weight parameter files.


        num_seqs (int): how many times to score the sequences; default to 10; 
        equivalent to the 'num_seq_per_target' argument in ProteinMPNN.

        device (str): where to perform ProteinMPNN calculations. Set to 'cpu' to 
        force calculations on the CPUs, otherwise the argument has no effect.

        sign_flip (bool): whether to multiply the score by -1. By default set to
        False so that the metric can be used in a minimization problem.

        use_surrogate_tied_residues (bool): set to True for single-state scoring; 
        False by default.
        '''
        self.model_weights_loc= model_weights_loc

        self.pdb_file_name= pdb_file_name
        self.chain_ids= chain_ids
        self.chain_ids.sort(key= sort_order)

        self.num_seqs= num_seqs

        assert score_type in ['designable_positions', 'all_positions'], \
            f'The score type {score_type} is not recognized!'
        self.score_type= score_type

        # sign_flip should be false by default, because negative log probability needs to be minimized
        self.sign_flip= sign_flip 

        self.use_surrogate_tied_residues= use_surrogate_tied_residues

        model_name= f'protein_mpnn_{"neg_" if not sign_flip else ""}log_prob_{score_type}'
        self.name= model_name + f'_chain_{"".join(self.chain_ids)}'
        
        self.device= device
    
    def __str__(self):
        return self.name
    
    def apply(self, candidates, protein):
        '''
        Input
        -----
        candidates (list[list[str]]): a list of design candidates. A candidate is 
        a list of residues at the designable positions.
        
        protein (protein.Protein): details of the protein system and design parameters.

        Output
        -----
        neg_mean_scores (np.ndarray[float]): a (N,) array containing the scores for the N 
        input candidates.
        '''
        logger.debug(
            textwrap.dedent(
                f'''\
                ProteinMPNNNegLogProb (device: {self.device}, name= {self.name}) called with the candidates:
                {sep}\n{candidates}\n{sep}
                '''
            )
        )

        protein_mpnn_wrapper= ProteinMPNNWrapper(
            protein= protein,
            temp= 0.1,
            model_weights_loc= self.model_weights_loc,
            device= self.device
        )
        
        protein_mpnn_outputs= protein_mpnn_wrapper.score(
            scoring_mode= 'score_only',
            chains_sublist= self.chain_ids, 
            pdb_file_name= self.pdb_file_name, 
            candidates= candidates, 
            num_seqs= self.num_seqs, 
            batch_size= 1, 
            seed= 1,
            use_surrogate_tied_residues= self.use_surrogate_tied_residues
        )
        
        if self.score_type == 'designable_positions':
            score_term= 'score'
        elif self.score_type == 'all_positions':
            score_term= 'global_score'
        
        mean_scores= np.array(
            [np.mean(output[score_term]) for output in protein_mpnn_outputs]
        )
        neg_mean_scores= -mean_scores if self.sign_flip else mean_scores

        logger.debug(
            textwrap.dedent(
                f'''\
                ProteinMPNNNegLogProb (device: {self.device}, name= {self.name}) apply() returned the following raw outputs:
                {sep}\n{protein_mpnn_outputs}\n{sep}
                and the following processed scores:
                {sep}\n{neg_mean_scores}\n{sep}
                '''
            )
        )

        return neg_mean_scores
        
