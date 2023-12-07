import os, io, sys, subprocess, tempfile, time, numpy as np, pandas as pd
from Bio import SeqIO
from multistate_methods.protein_mpnn_ga.af2rank import af2rank
from multistate_methods.protein_mpnn_ga.protein import DesignedProtein, SingleStateProtein, Residue, TiedResidue
from multistate_methods.protein_mpnn_ga.utils import sort_order, npz_to_dict, get_logger, sep, Device
from inspect import signature

logger= get_logger(__name__)

class ObjectiveAF2Rank(object):
    def __init__(self, chain_ids, template_file_loc, tmscore_exec, params_dir, score_term= 'composite', device= 'cpu', sign_flip= True, use_surrogate_tied_residues= False):
        multimer= True if len(chain_ids) > 1 else False
        # note that the multimer params version might change in the future, depending on alphafold-multimer and colabfold developments.
        model_name= 'model_1_multimer_v3' if multimer else 'model_1_ptm'

        self.chain_ids= chain_ids
        self.chain_ids.sort(key= sort_order)
        self.use_surrogate_tied_residues= use_surrogate_tied_residues
        self.score_term= score_term
        self.settings= {
            'rm_seq': True, #mask_sequence
            'rm_sc': True, #mask_sidechains
            'rm_ic': False, #mask_interchain
            'recycles': 1, 'iterations': 1, 'model_name': model_name
        }
        
        self.sign_flip= sign_flip
        self.name= ('neg_' if sign_flip else '') + f'af2rank_{score_term}_chain_{"".join(self.chain_ids)}_{model_name}'
        self.device= device
        self.template_file_loc= template_file_loc
        self.tmscore_exec= tmscore_exec
        self.params_dir= params_dir
        
    def __str__(self):
        return self.name
        
    def apply(self, candidates, protein):
        '''
        Can handle multiple sequences
        '''
        full_seqs= []
        for candidate in candidates:
            full_seq= ''
            for chain_id in self.chain_ids:
                # by default, all missing residues are ignored, and multiple chains are concatenated as if there were only one continuous chain with no breaks
                full_seq+= protein.get_chain_full_seq(chain_id, candidate, drop_terminal_missing_res= True, drop_internal_missing_res= True, use_surrogate_tied_residues= self.use_surrogate_tied_residues)
            full_seqs.append(full_seq)
        logger.debug(f'AF2Rank (device: {self.device}, name: {self.name}) called with the sequences:\n{sep}\n{full_seqs}\n{sep}\n')
        output= []
        with Device(self.device):
            model= af2rank(
                pdb= self.template_file_loc,
                chain= ','.join(self.chain_ids),
                model_name= self.settings['model_name'],
                tmscore_exec= self.tmscore_exec,
                params_dir= self.params_dir)
            for seq_ind, seq in enumerate(full_seqs):
                t0= time.time()
                output_dict= model.predict(seq= seq, **self.settings, output_pdb= None, extras= {'id': seq_ind}, verbose= False)
                t1= time.time()
                logger.info(f'AF2Rank (device: {self.device}, name: {self.name}) run time: {t1 - t0} s.')
                logger.debug(f'AF2Rank (device: {self.device}, name: {self.name}) output:\n{sep}\n{output_dict}\n{sep}\n')
                output.append(output_dict[self.score_term])
        output= np.asarray(output)
        neg_output= -output if self.sign_flip else output # take the negative because the algorithm expects a minimization problem

        logger.debug(f'AF2Rank (device: {self.device}, name: {self.name}) final output:\n{sep}\n{neg_output}\n{sep}\n')

        return neg_output

class ObjectiveESM(object):
    '''
    This method cannot be used to score multimers
    '''
    def __init__(self, chain_id, script_loc, model_name= 'esm1v', device= 'cpu', sign_flip= True):
        self.chain_id= chain_id
        self.model_name= model_name
        self.device= device
        self.sign_flip= sign_flip
        self.name= ('neg_' if sign_flip else '') + f'{model_name}_chain_{chain_id}'

        # input and output both handled through io streams
        self.exec= [
            sys.executable, script_loc,
            '--device', device,
            '--model', self.model_name,
            '--score_name', self.model_name,
            '--masking_off',
            '--csv'
        ]

    def __str__(self):
        return self.name

    def apply(self, candidates, protein, position_wise= False):
        '''
        Can handle multiple sequences
        '''
        with Device(self.device):
            esm_dir= tempfile.TemporaryDirectory()
            input_fa= ''
            for candidate_ind, candidate in enumerate(candidates):
                full_seq= protein.get_chain_full_seq(self.chain_id, candidate, drop_terminal_missing_res= False, drop_internal_missing_res= False)
                input_fa+= f'>seq_{candidate_ind}\n{full_seq}\n'
            with open(f'{esm_dir.name}/input_seq.fa', 'w') as f:
                f.write(input_fa)
            
            exec_str= self.exec
            exec_str+= ['-i', f'{esm_dir.name}/input_seq.fa']
            out_f= f'{esm_dir.name}/scores.out'
            if position_wise:
                exec_str+= ['--positionwise', out_f]
            else:
                exec_str+= ['-o', out_f]

            t0= time.time()
            proc= subprocess.run(exec_str, stdout= subprocess.PIPE, stderr= subprocess.PIPE, check= False)
            t1= time.time()
            logger.info(f'ESM (device: {self.device}, name= {self.name}, position_wise) run time: {t1 - t0} s.\n')  

            try:
                output_df= pd.read_csv(out_f, sep= ',')
                if position_wise:
                    output_arr= output_df[self.model_name].str.split(pat= ';', expand= True).to_numpy(dtype= float)
                else:
                    output_arr= output_df[self.model_name].to_numpy(dtype= float)
                    
            except:
                # The script uses stderr to print progression info, so only check for error when attempting to read the output file
                logger.exception(f'Command {proc.args} returned non-zero exist status {proc.returncode} with the input\n{sep}\n{input_fa}\n{sep}\nand the stdout\n{sep}\n{proc.stdout.decode()}\n{sep}\nand the stderr\n{sep}\n{proc.stderr.decode()}\n{sep}\n')
                sys.exit(1)

            esm_dir.cleanup()
            
            '''
            if position_wise:
                out= tempfile.NamedTemporaryFile(suffix= str(hash(input_fa)))
                exec_str+= ['--positionwise', out.name]
                
                t0= time.time()
                proc= subprocess.Popen(exec_str, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                proc_output, proc_error= proc.communicate(input= input_fa.encode())
                t1= time.time()
                logger.info(f'ESM (device: {self.device}, name= {self.name}, position_wise) run time: {t1 - t0} s.\n')                    
                try:
                    time.sleep(5) # wait for filesystem I/O
                    output_df= pd.read_csv(out.name, sep= ',')
                except:
                    # The script uses stderr to print progression info, so only check for error when attempting to read the output file
                    logger.exception(f'Command {proc.args} returned non-zero exist status {proc.returncode} with the input\n{sep}\n{input_fa}\n{sep}\n and the stderr\n{sep}\n{proc_error.decode()}n{sep}\n')
                    sys.exit(1)
                output_arr= output_df[self.model_name].str.split(pat= ';', expand= True).to_numpy(dtype= float)
                out.close()
            else:
                t0= time.time()
                proc= subprocess.Popen(exec_str, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                proc_output, proc_error= proc.communicate(input= input_fa.encode())
                t1= time.time()
                logger.info(f'ESM (device: {self.device}, name= {self.name}) run time: {t1 - t0} s.\n')
                try:
                    time.sleep(5)
                    output_df= pd.read_csv(io.StringIO(proc_output.decode()), sep= ',')
                except:
                    logger.exception(f'Command {proc.args} returned non-zero exist status {proc.returncode} with the input\n{sep}\n{input_fa}\n{sep}\n and the stderr\n{sep}\n{proc_error.decode()}\n{sep}\n')
                    sys.exit(1)
                output_arr= output_df[self.model_name].to_numpy(dtype= float)
            '''
            
            logger.debug(f'ESM (device: {self.device}, name= {self.name}) was called with the command:\n{sep}\n{exec_str}\n{sep}\nstdout:\n{sep}\n{proc.stdout.decode()}\n{sep}\nstderr:\n{sep}\n{proc.stderr.decode()}\n{sep}\n')

            neg_output_arr= -output_arr if self.sign_flip else output_arr # take the negative because the algorithm expects a minimization problem
            logger.debug(f'ESM (device: {self.device}, name= {self.name}) apply() returned the following results:\n{sep}\n{neg_output_arr}\n{sep}\n')

            return neg_output_arr

class ObjectiveDebug(object):
    def __init__(self):
        self.name= 'debug'

    def __str__(self):
        return self.name

    def apply(self, candidates, protein):
        results= np.random.rand(len(candidates))
        logger.debug(f'{str(self)} objective returned the following results:\n{sep}\n{results}\n{sep}\n')
        return results
    
class ObjectiveCombine(object):
    def __init__(self, objectives, combine_fxn):
        #assert len(signature(combine_fxn).parameters) == len(objectives), f'The number of arguments accepted by {combine_fxn.__name__} is not equal to the length of the input objective functions list!'

        self.objectives= objectives
        self.combine_fxn= combine_fxn
        self.name= f'combine_{"+".join([str(objective) for objective in objectives])}_using_{combine_fxn.__name__}'

    def __str__(self):
        return self.name

    def apply(self, candidates, protein):
        logger.debug(f'{str(self)} objective called with the candidates:\n{sep}\n{candidates}\n{sep}\n')
        obj_res= np.array([objective.apply(candidates, protein) for objective in self.objectives]).T
        results= np.array([self.combine_fxn(res) for res in obj_res])
        logger.debug(f'{str(self)} objective received the following objectives and returned the following results:\n{sep}\n{results}\n{sep}\n')
        return results
    
class ProteinMPNNWrapper(object):
    def __init__(
            self, protein, temp,
            model_weights_loc,
            detect_degeneracy= False, corr_cutoff= 0.9,
            uniform_sampling= False, geometric_prob= 1.0,
            device= 'cpu', protein_mpnn_run_loc= None
        ):
        self.protein= protein

        if protein_mpnn_run_loc is None:
            protein_mpnn_run_loc= os.path.dirname(os.path.realpath(__file__)) + '/../protein_mpnn_pd/protein_mpnn_run.py'

        self.exec_str= [
            sys.executable, protein_mpnn_run_loc,
            '--path_to_model_weights', model_weights_loc,
            '--out_folder', os.getcwd(),
            '--sampling_temp', str(temp),
            '--corr_cutoff', str(corr_cutoff),
            '--geometric_prob', str(geometric_prob),
            '--write_to_stdout'
        ]
        if detect_degeneracy:
            self.exec_str+= ['--detect_degeneracy']
        if uniform_sampling:
            self.exec_str+=['--uniform_sampling']
        #TODO: enable **kwargs parsing

        self.device= device

    def design(self, method, base_candidate, proposed_des_pos_list, num_seqs, batch_size, seed= None):
        with Device(self.device):
            if method not in ['ProteinMPNN-AD', 'ProteinMPNN-PD']:
                raise ValueError('Invalid method definition.')
            
            designed_protein= DesignedProtein(self.protein, base_candidate, proposed_des_pos_list)

            out_dir, file_loc_exec_str= designed_protein.dump_jsons()

            exec_str= self.exec_str + file_loc_exec_str + ['--num_seq_per_target', str(num_seqs), '--batch_size', str(batch_size)]
            if seed is not None:
                exec_str += ['--seed', str(seed)]
            if method == 'ProteinMPNN-PD':
                exec_str+= ['--pareto']

            t0= time.time()
            proc= subprocess.run(exec_str, stdout= subprocess.PIPE, stderr= subprocess.PIPE, check= False)
            t1= time.time()

            logger.info(f'ProteinMPNN (device: {self.device}) run time: {t1 - t0} s.')
            if proc.stderr:
                raise RuntimeError(f'Command {proc.args} returned non-zero exist status {proc.returncode} with the stderr\n{proc.stderr.decode()}')
            logger.debug(f'ProteinMPNN (device: {self.device}) was called with the command:\n{sep}\n{exec_str}\n{sep}\nstdout:\n{sep}\n{proc.stdout.decode()}\n{sep}\nstderr:\n{sep}\n{proc.stderr.decode()}\n{sep}\n')

            records = SeqIO.parse(io.StringIO(proc.stdout.decode()), "fasta")
            
            out_dir.cleanup()

            return list(records), designed_protein.design_seq.chains_to_design
    
    def design_seqs_to_candidates(self, fa_records, candidate_chains_to_design, base_candidate):
        AA_locator= []
        for tied_res in self.protein.design_seq.tied_residues:
            if isinstance(tied_res, TiedResidue):
                # use the first residue in a tied_residue as the representative
                rep_res= tied_res.residues[0]
            elif isinstance(tied_res, Residue):
                rep_res= tied_res
                
            chain_id= rep_res.chain_id
            resid= rep_res.resid
            res_ind= resid - self.protein.chains_dict[chain_id].init_resid # the offset here is due to ProteinMPNN not having the missing terminal res
            AA_locator.append([chain_id, res_ind])

        seq_list= []
        for fa in fa_records:
            name, seq = fa.id, str(fa.seq)
            seq_dict= dict(zip(candidate_chains_to_design, seq.split('/'))) # ProteinMPNN only output sequences of the designable chains
            seq_list.append(seq_dict)
        
        candidates= []
        # skip the first element in seq_list, since ProteinMPNN will always output the input sequence as the first output
        for seq in seq_list[1:]:
            candidate= base_candidate.copy()
            for candidate_ind, (chain_id, res_ind) in enumerate(AA_locator):
                if chain_id in candidate_chains_to_design:
                    candidate[candidate_ind]= seq[chain_id][res_ind]
            candidates.append(candidate)
        logger.debug(f'ProteinMPNN design_seqs_to_candidates() input:\n{sep}\n{seq_list[1:]}\n{sep}\noutput:\n{sep}\n{candidates}\n{sep}\n')
        candidates= np.asarray(candidates)
        
        return candidates
    
    def design_and_decode_to_candidates(self, method, base_candidate, proposed_des_pos_list, num_seqs, batch_size, seed= None):
        fa_records, candidate_chains_to_design= self.design(method, base_candidate, proposed_des_pos_list, num_seqs, batch_size, seed)
        candidates= self.design_seqs_to_candidates(fa_records, candidate_chains_to_design, base_candidate)
        return candidates

    def score(self, scoring_mode, chains_sublist, pdb_file_name, candidates= None, num_seqs= 1, batch_size= 1, seed= None, use_surrogate_tied_residues= False):
        '''
        Note here that the temperature setting has no effect on the output scores.
        Only the --score_only mode is configured in ProteinMPNN to take in an input FASTA file.
        To score candidates in the other modes is not implemented.
        '''
        ss_protein= SingleStateProtein(self.protein, chains_sublist, pdb_file_name, use_surrogate_tied_residues)
        
        if scoring_mode not in ['score_only', 'conditional_probs_only', 'conditional_probs_only_backbone', 'unconditional_probs_only']:
            raise ValueError(f'Unrecognized scoring_mode {scoring_mode}')
        if candidates is not None and scoring_mode in ['conditional_probs_only', 'conditional_probs_only_backbone', 'unconditional_probs_only']:
            raise NotImplementedError()
        
        score_exec_str= [f'--{scoring_mode}', '1']
        # conditional_probs_only_backbone can only be activated if conditional_probs_only is also turned on
        if scoring_mode == 'conditional_probs_only_backbone':
            score_exec_str+=  ['--conditional_probs_only', '1']
        
        out_dir, file_loc_exec_str= ss_protein.dump_jsons()
        file_loc_exec_str+= ['--out_folder', out_dir.name] # override the out folder setting when the wrapper was init.

        if scoring_mode == 'score_only' and candidates is not None:
            input_seqs= ss_protein.candidates_to_full_seqs(candidates)
            input_seqs_f= f'{out_dir.name}/input_seqs.fa'
            with open(input_seqs_f, 'w') as f:
                for seq_ind, seq in enumerate(input_seqs):
                    f.write(f'>des_seq_{seq_ind}\n{seq}\n')
            score_exec_str+= ['--path_to_fasta', input_seqs_f]

        exec_str= self.exec_str + file_loc_exec_str + score_exec_str + ['--num_seq_per_target', str(num_seqs), '--batch_size', str(batch_size), '--seed', str(seed)]
        if '--write_to_stdout' in exec_str: exec_str.remove('--write_to_stdout')

        with Device(self.device):
            t0= time.time()
            proc= subprocess.run(exec_str, stdout= subprocess.PIPE, stderr= subprocess.PIPE, check= False)
            t1= time.time()
            
            logger.info(f'ProteinMPNN (device: {self.device}) run time: {t1 - t0} s.')
            if proc.stderr:
                raise RuntimeError(f'Command {proc.args} returned non-zero exist status {proc.returncode} with the stderr\n{proc.stderr.decode()}')
            logger.debug(f'ProteinMPNN (device: {self.device}) was called with the command:\n{sep}\n{exec_str}\n{sep}\nstdout:\n{sep}\n{proc.stdout.decode()}\n{sep}\nstderr:\n{sep}\n{proc.stderr.decode()}\n{sep}\n')

        pdb_file_name= 'combined_pdb' # the pdb file name will always be combined_pdb because of the pdb parsing/merging step
        if scoring_mode == 'score_only':
            if candidates is None:
                outputs= [npz_to_dict(np.load(f'{out_dir.name}/{scoring_mode}/{pdb_file_name}_pdb.npz'))]
            else:
                outputs= [npz_to_dict(np.load(f'{out_dir.name}/{scoring_mode}/{pdb_file_name}_fasta_{ind + 1}.npz')) for ind in range(len(candidates))]
        elif scoring_mode in ['conditional_probs_only', 'conditional_probs_only_backbone']:
            outputs= [npz_to_dict(np.load(f'{out_dir.name}/conditional_probs_only/{pdb_file_name}.npz'))]
        elif scoring_mode == 'unconditional_probs_only':
            outputs= [npz_to_dict(np.load(f'{out_dir.name}/{scoring_mode}/{pdb_file_name}.npz'))]
        
        out_dir.cleanup()

        return outputs
        

class ObjectiveProteinMPNNNegLogProb(object):
    def __init__(self, chain_ids, pdb_file_name, score_type, model_weights_loc, protein_mpnn_run_loc= None, num_seqs= 10, device= 'cpu', sign_flip= False, use_surrogate_tied_residues= False):
        self.model_weights_loc= model_weights_loc
        if protein_mpnn_run_loc is None:
            self.protein_mpnn_run_loc= os.path.dirname(os.path.realpath(__file__)) + '/../protein_mpnn_pd/protein_mpnn_run.py'
        else:
            self.protein_mpnn_run_loc= protein_mpnn_run_loc

        self.pdb_file_name= pdb_file_name
        self.chain_ids= chain_ids
        self.chain_ids.sort(key= sort_order)

        self.num_seqs= num_seqs

        assert score_type in ['designable_positions', 'all_positions'], f'The score type {score_type} is not recognized!'
        self.score_type= score_type

        self.sign_flip= sign_flip # sign_flip should be false by default, because negative log probability needs to be minimized

        self.use_surrogate_tied_residues= use_surrogate_tied_residues

        model_name= f'protein_mpnn_{"neg_" if not sign_flip else ""}log_prob_{score_type}'
        self.name= model_name + f'_chain_{"".join(self.chain_ids)}'
        
        self.device= device
    
    def __str__(self):
        return self.name
    
    def apply(self, candidates, protein):
        logger.debug(f'ProteinMPNNNegLogProb (device: {self.device}, name= {self.name}) called with the candidates:\n{sep}\n{candidates}\n{sep}\n')

        protein_mpnn_wrapper= ProteinMPNNWrapper(
            protein= protein,
            temp= 0.1,
            model_weights_loc= self.model_weights_loc,
            detect_degeneracy= False,
            uniform_sampling= False,
            device= self.device,
            protein_mpnn_run_loc= self.protein_mpnn_run_loc
        )
        
        protein_mpnn_outputs= protein_mpnn_wrapper.score(
            scoring_mode= 'score_only',
            chains_sublist= self.chain_ids, 
            pdb_file_name= self.pdb_file_name, 
            candidates= candidates, 
            num_seqs= self.num_seqs, 
            batch_size= 1, 
            seed= 1,
            use_surrogate_tied_residues= self.use_surrogate_tied_residues)
        
        if self.score_type == 'designable_positions':
            score_term= 'score'
        elif self.score_type == 'all_positions':
            score_term= 'global_score'
        
        mean_scores= np.array([np.mean(output[score_term]) for output in protein_mpnn_outputs])
        neg_mean_scores= -mean_scores if self.sign_flip else mean_scores

        logger.debug(f'ProteinMPNNNegLogProb (device: {self.device}, name= {self.name}) apply() returned the following raw outputs:\n{sep}\n{protein_mpnn_outputs}\n{sep}\nand the following processed scores:\n{sep}\n{neg_mean_scores}\n{sep}\n')

        return neg_mean_scores
        
        