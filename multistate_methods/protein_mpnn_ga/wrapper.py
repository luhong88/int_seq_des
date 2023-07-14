import os, io, sys, subprocess, tempfile, time, numpy as np, pandas as pd
from Bio import SeqIO
from multistate_methods.protein_mpnn_ga.af2rank import af2rank
from multistate_methods.protein_mpnn_ga.protein import DesignedProtein
from multistate_methods.protein_mpnn_ga.utils import get_logger, sep, Device

logger= get_logger(__name__)

class ObjectiveAF2Rank(object):
    def __init__(self, chain_ids, template_file_loc, tmscore_exec, params_dir, score_term= 'composite', device= 'cpu', sign_flip= True):
        multimer= True if len(chain_ids) > 1 else False
        # note that the multimer params version might change in the future, depending on alphafold-multimer and colabfold developments.
        model_name= 'model_1_multimer_v3' if multimer else 'model_1_ptm'

        self.chain_ids= chain_ids
        self.chain_ids.sort()
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
                full_seq+= protein.get_chain_full_seq(chain_id, candidate, drop_terminal_missing_res= True, drop_internal_missing_res= True)
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
            input_fa= ''
            for candidate_ind, candidate in enumerate(candidates):
                full_seq= protein.get_chain_full_seq(self.chain_id, candidate, drop_terminal_missing_res= False, drop_internal_missing_res= False)
                input_fa+= f'>seq_{candidate_ind}\n{full_seq}\n'

            exec_str= self.exec
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
                    logger.exception(f'Command {proc.args} returned non-zero exist status {proc.returncode} with the stderr\n{sep}{proc_error.decode()}{sep}\n')
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
                    output_df= pd.read_csv(io.StringIO(proc_output.decode()), sep= ',')
                except:
                    logger.exception(f'Command {proc.args} returned non-zero exist status {proc.returncode} with the stderr\n{sep}{proc_error.decode()}{sep}\n')
                    sys.exit(1)
                output_arr= output_df[self.model_name].to_numpy(dtype= float)
            
            logger.debug(f'ESM (device: {self.device}, name= {self.name}) was called with the command:\n{sep}\n{exec_str}\n{sep}\nstdout:\n{sep}\n{proc_output.decode()}\n{sep}\nstderr:\n{sep}\n{proc_error.decode()}\n{sep}\n')

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
            # use the first residue in a tied_residue as the representative
            rep_res= tied_res.residues[0]
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

    def score(self):
        raise NotImplementedError()