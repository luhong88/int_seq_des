import os, io, sys, subprocess, tempfile, numpy as np, pandas as pd
from Bio import SeqIO
from multistate_methods.protein_mpnn_ga.af2rank import af2rank
from multistate_methods.protein_mpnn_ga.protein import DesignedProtein

# a way to foce cpu computation
class Device(object):
    def __init__(self, device):
        self.device= device
    def __enter__(self):
        if self.device == 'cpu':
            os.environ['CUDA_VISIBLE_DEVICES']= ''
    def __exit__(self, type, value, traceback):
        if self.device == 'cpu':
            del os.environ['CUDA_VISIBLE_DEVICES']

class ObjectiveAF2Rank(object):
    #TODO: make sure that this method can handle multiple chains at once (also need to support undesigned chains)
    #TODO: implement cpu device
    def __init__(self, chain_id, template_file_loc, tmscore_exec, params_dir, model_name= 'model_1_ptm', score_term= 'composite'):
        self.chain_id= chain_id
        self.model= af2rank(
            pdb= template_file_loc,
            chain= chain_id,
            model_name= model_name,
            tmscore_exec= tmscore_exec,
            params_dir= params_dir)
        self.score_term= score_term
        self.settings= {
            'rm_seq': True, #mask_sequence
            'rm_sc': True, #mask_sidechains
            'rm_ic': False, #mask_interchain
            'recycles': 1, 'iterations': 1, 'model_name': model_name
        }
        
    def apply(self, candidates, protein):
        '''
        Can handle multiple sequences
        '''
        full_seqs= []
        for candidate in candidates:
            full_seq= protein.get_chain_full_seq(self.chain_id, candidate, drop_terminal_missing_res= True, drop_internal_missing_res= True)
            full_seqs.append(full_seq)

        output= []
        for seq_ind, seq in enumerate(full_seqs):
            output_dict= self.model.predict(seq= seq, **self.settings, output_pdb= None, extras= {'id': seq_ind}, verbose= False)
            output.append(output_dict[self.score_term])
        output= np.asarray(output)

        neg_output= -output # take the negative because the algorithm expects a minimization problem

        return neg_output


class ObjectiveESM(object):
    #TODO: make sure that this method can handle multiple chains at once
    def __init__(self, chain_id, script_loc, model_name= 'esm1v', device= 'cpu'):
        self.chain_id= chain_id
        self.model_name= model_name
        self.device= device

        # input and output both handled through io streams
        self.exec= [
            sys.executable, script_loc,
            '--device', device,
            '--model', self.model_name,
            '--score_name', self.model_name,
            '--masking_off',
            '--csv'
        ]

    def apply(self, candidates, protein, position_wise= False):
        '''
        Can handle multiple sequences
        '''
        with Device(self.device):
            input_fa= ''
            for candidate_ind, candidate in enumerate(candidates):
                full_seq= protein.get_chain_full_seq(self.chain_id, candidate, drop_terminal_missing_res= False, drop_internal_missing_res= False)
                input_fa+= f'>seq_{candidate_ind}\n{full_seq}\n'

            if position_wise:
                out= tempfile.NamedTemporaryFile()
                exec_str= self.exec + ['--positionwise', out.name]

                proc= subprocess.Popen(exec_str, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                proc.communicate(input= input_fa.encode())
                output_df= pd.read_csv(out.name, sep= ',')
                output_arr= output_df[self.model_name].str.split(pat= ';', expand= True).to_numpy()
                out.close()
            else:
                proc= subprocess.Popen(self.exec, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                output, err= proc.communicate(input= input_fa.encode())
                output_df= pd.read_csv(io.StringIO(output.decode()), sep= ',')
                output_arr= output_df[self.model_name].to_numpy()
            
            neg_output_arr= -output_arr # take the negative because the algorithm expects a minimization problem
            return neg_output_arr
    
class ProteinMPNNWrapper(object):
    def __init__(
            self, protein, temp,
            model_weights_loc,
            uniform_sampling= 0, geometric_prob= 1.0,
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
            '--write_to_stdout'
        ]
        #TODO: enable **kwargs parsing

        self.uniform_sampling= uniform_sampling
        self.geometric_prob= geometric_prob

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
                exec_str+= [
                    '--pareto',
                    '--uniform_sampling', str(self.uniform_sampling),
                    '--geometric_prob', str(self.geometric_prob)
                ]
            #proc= subprocess.run(exec_str, stdout= subprocess.PIPE, stderr= subprocess.PIPE, check= True)
            proc= subprocess.run(exec_str, stdout= subprocess.PIPE, stderr= subprocess.PIPE, check= False)
            if len(proc.stderr) > 0:
                print(exec_str)
                print(proc.stderr)
                sys.exit(1)

            records = SeqIO.parse(io.StringIO(proc.stdout.decode()), "fasta")

            out_dir.cleanup()

            return records
    
    def design_seqs_to_candidates(self, fa_records):
        AA_locator= []
        for tied_res in self.protein.design_seq.tied_residues:
            # use the first residue in a tied_residue as the representative
            rep_res= tied_res.residues[0]
            chain_id= rep_res.chain_id
            resid= rep_res.resid
            res_ind= resid - self.protein.chains_dict[chain_id].init_resid
            AA_locator.append([chain_id, res_ind])

        chain_ids= [chain.chain_id for chain in self.protein.chains_list]
        seq_list= []
        for fa in fa_records:
            name, seq = fa.id, str(fa.seq)
            seq_dict= dict(zip(chain_ids, seq.split('/')))
            seq_list.append(seq_dict)
        
        candidates= []
        # skip the first element in seq_list, since ProteinMPNN will always output the input sequence as the first output
        for seq in seq_list[1:]:
            candidates.append([seq[chain_id][res_ind] for chain_id, res_ind in AA_locator])
        
        return np.asarray(candidates)
    
    def design_and_decode_to_candidates(self, method, base_candidate, proposed_des_pos_list, num_seqs, batch_size, seed= None):
        fa_records= self.design(method, base_candidate, proposed_des_pos_list, num_seqs, batch_size, seed)
        candidates= self.design_seqs_to_candidates(fa_records)
        return candidates

    def score(self):
        raise NotImplementedError()