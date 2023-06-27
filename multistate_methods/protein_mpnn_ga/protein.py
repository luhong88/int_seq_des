import glob, sys, subprocess, tempfile, json, numpy as np
from multistate_methods.protein_mpnn_ga.utils import get_logger, sep, alphabet, merge_pdb_files

logger= get_logger(__name__)

class Residue(object):
    def __init__(self, chain_id, resid, weight):
        if not isinstance(chain_id, str):
            raise TypeError(f'chain_id {chain_id} is not a str.')
        if not (isinstance(resid, int) and resid > 0):
            raise TypeError(f'resid {resid} is not a positive integer (resid should be 1-indexed).')
        if not isinstance(weight, float):
            raise TypeError(f'weight {weight} is not a float.')
        
        self.chain_id= chain_id
        self.resid= resid
        self.weight= weight

class TiedResidue(object):
    def __init__(self, *residues, omit_AA= None):
        self.residues= residues
        self.allowed_AA= alphabet

        chain_ids= [res.chain_id for res in residues]
        self.chain_ids_unique= [*set(chain_ids)]

        if len(self.chain_ids_unique) != len(chain_ids):
            raise NotImplementedError(f'You are trying to tie together residues in the same chain; this feature is not implemented.')

        if omit_AA is not None:
            if not isinstance(omit_AA, str):
                raise TypeError(f'omit_AA {omit_AA} is not a str.')
            for AA in list(omit_AA):
                self.allowed_AA= self.allowed_AA.replace(AA, '')
    
    def __iter__(self):
        return iter(self.residues)
    
    def __str__(self):
        return '[' + ', '.join([f'{res.chain_id}.{res.resid}({res.weight})' for res in self]) + ']'

class DesignSequence(object):
    '''
    We do not allow for the presence of un-tied (i.e., single-state design) positions
    '''
    def __init__(self, *tied_residues):
        self.tied_residues= tied_residues

        self.n_des_res= len(self.tied_residues)

        self.chains_to_design= np.array([[residue.chain_id for residue in tied_residue] for tied_residue in self.tied_residues])
        self.chains_to_design= np.unique(self.chains_to_design.flatten()) # will return in alphabetical order; upper case before lower case

        self.chain_des_pos_dict= {chain: [] for chain in self.chains_to_design}
        for tied_residue in self.tied_residues:
            for residue in tied_residue:
                self.chain_des_pos_dict[residue.chain_id].append(residue.resid)
        
        logger.debug(f'DesignSequence init definition:\n{sep}\ntied_residues: {str(self)}\nchains_to_design: {self.chains_to_design}\nchain_des_pos_dict: {self.chain_des_pos_dict}\n{sep}\n')

    def __iter__(self):
        return iter(self.tied_residues)
    
    def __str__(self):
        return '[' + ', '.join([str(tied_res) for tied_res in self]) + ']'

class Chain(object):
    # designable chains are mandatory, non-designable chains are mandatory if scored by AF2Rank
    def __init__(self, chain_id, init_resid, fin_resid, internal_missing_res_list, full_seq):
        if not isinstance(chain_id, str):
            raise TypeError(f'chain_id {chain_id} is not a str.')
        if not (isinstance(init_resid, int) and init_resid > 0):
            raise TypeError(f'init_resid {init_resid} is not a positive integer (resid is 1-indexed).')
        if not (isinstance(fin_resid, int) and fin_resid > 0):
            raise TypeError(f'fin_resid {fin_resid} is not a positive integer (resid is 1-indexed).')
        if not isinstance(internal_missing_res_list, list):
            raise TypeError(f'internal_missing_res_list {internal_missing_res_list} is not a list.')
        if not isinstance(full_seq, str):
            raise TypeError(f'full_seq {full_seq} is not a str.')
        
        self.chain_id= chain_id

        self.init_resid= init_resid
        self.fin_resid= fin_resid
        self.resid_range= [init_resid, fin_resid]

        self.internal_missing_res_list= internal_missing_res_list

        self.full_seq= full_seq # should include all missing residues; do not need to agree with template seqs

        logger.debug(f'Chain init definition:\n{sep}\nchain_id: {chain_id}\ninit_resid: {init_resid}, fin_resid: {fin_resid}\ninternal_missing_res_list: {internal_missing_res_list}\nfull_seq: {full_seq}\n{sep}\n')

class Protein(object):
    def __init__(self, design_seq, chains_list, chains_neighbors_list, pdb_files_dir, protein_mpnn_helper_scripts_dir):
        '''
        Input structures must have the correct chain id
        It is okay to have gaps in the chains; resid 1-indexed
        
        '''
        self.design_seq= design_seq
        self.chains_neighbors_list= chains_neighbors_list
        
        updated_chains_list= []
        for chain in chains_list:
            # check if the chain is designable
            if chain.chain_id in self.design_seq.chains_to_design:
                chain.is_designable= True
            else:
                chain.is_designable= False
            logger.debug(f'chain {chain.chain_id} is marked with is_designable={chain.is_designable}')
            # update the chains with neighbors list
            neighbors= []
            for neighbors_list in self.chains_neighbors_list:
                if chain.chain_id in neighbors_list:
                    neighbors+= neighbors_list
            neighbors= [*set(neighbors)]
            neighbors.sort()
            chain.neighbors_list= neighbors
            updated_chains_list.append(chain)
            logger.debug(f'chain {chain.chain_id} updated with the following neighbors_list: {chain.neighbors_list}')

        # chains_list should only include designable chains
        # ensure that the chains are listed in alphabetical order
        # need python >=3.7 to ensure the dict remembers insertion order
        chain_id_list= [chain.chain_id for chain in updated_chains_list]
        chain_order= np.argsort(chain_id_list)
        self.chains_list= [updated_chains_list[ind] for ind in chain_order]
        self.chains_dict= {chain.chain_id: chain for chain in self.chains_list}

        self.pdb_files_dir= pdb_files_dir
        self.helper_scripts_dir= protein_mpnn_helper_scripts_dir
        
        self.parsed_pdb_json, parsed_pdb_handle= self.parse_pdbs()
        self.parsed_fixed_chains= self.parse_fixed_chains(parsed_pdb_handle)
        self.parsed_fixed_positions= self.parse_fixed_positions(parsed_pdb_handle)
        self.parsed_tied_positions= self.parse_tied_positions()
        #TODO: add additional parsing options

        parsed_pdb_handle.close()
    
    def get_candidate(self):
        candidate= []
        for tied_res in self.design_seq.tied_residues:
            # use the first residue in a tied_residue as the representative
            rep_res= tied_res.residues[0]
            chain_id= rep_res.chain_id
            resid= rep_res.resid
            candidate.append(self.chains_dict[chain_id].full_seq[resid - 1]) # for full seq with no missing residue, only need to convert from 1-index to 0-index
        logger.debug(f'get_candidate() returned: {candidate}\n')
        return candidate

    def get_chain_full_seq(
            self, 
            chain_id, 
            candidate, 
            drop_terminal_missing_res, 
            drop_internal_missing_res, 
            replace_missing_residues_with= None
        ):
        
        full_seq= np.array(list(self.chains_dict[chain_id].full_seq), dtype= object)
        init_resid, fin_resid= self.chains_dict[chain_id].resid_range

        if candidate is not None:
            for tied_res, candidate_AA in zip(self.design_seq.tied_residues, candidate):
                for res in tied_res.residues:
                    if res.chain_id == chain_id:
                        #full_seq[res.resid - init_resid]= candidate_AA
                        full_seq[res.resid - 1]= candidate_AA
        
        resid_arr= np.arange(len(full_seq)) + 1
        terminal_missing_res_mask= (resid_arr < init_resid) | (resid_arr > fin_resid)
        internal_missing_res_mask= np.isin(resid_arr, self.chains_dict[chain_id].internal_missing_res_list)

        if replace_missing_residues_with is not None:
            full_seq[terminal_missing_res_mask | internal_missing_res_mask]= replace_missing_residues_with
        if drop_terminal_missing_res:
            full_seq[terminal_missing_res_mask]= None
        if drop_internal_missing_res:
            full_seq[internal_missing_res_mask]= None

        output_seq= ''.join(full_seq[full_seq != None])
        logger.debug(f'get_chain_full_seq() returned the folllowing results:\n{sep}\nchain_id: {chain_id}\ncandidate: {candidate}\ndrop_terminal: {drop_terminal_missing_res}\ndrop_internal: {drop_internal_missing_res}\nreplace_missing: {replace_missing_residues_with}\noutput_seq: {output_seq}\n{sep}\n')
        return output_seq
    
    def candidate_to_chain_des_pos(self, candidate_des_pos_list, chain_id, drop_terminal_missing_res= False, drop_internal_missing_res= False):
        '''
        input candidate_des_pos_list is a 0-indexed list of indices for the candidate array elements
        output chain_des_pos_list is a 0-indexed, list of indices for the corresponding residues in a given chain, conditioned on the inclusion/exclusion of missing residues
        if a des_pos does not map onto the chain, the returned index is None
        the function depends on the assumption that residues on the same chain cannot be tied together
        '''
        chain_des_pos_list= []
        for des_pos in candidate_des_pos_list:
            tied_res= self.design_seq.tied_residues[des_pos]
            if chain_id in tied_res.chain_ids_unique:
                for res in tied_res:
                    if res.chain_id == chain_id:
                        resid= res.resid
                        if drop_internal_missing_res:
                            missing_list= self.chains_dict[chain_id].internal_missing_res_list
                            if resid in missing_list:
                                raise ValueError(f'Designable residue {resid} in chain {chain_id} is on the internal_missing_res_list of that chain, which should not be possible.')
                            offset= sum(missing_resid < resid for missing_resid in missing_list)
                            resid-= offset
                        if drop_terminal_missing_res:
                            init_resid= self.chains_dict[chain_id].init_resid
                            resid-= init_resid # this automatically make resid 0-indexed
                        else:
                            resid-= 1 # make resid 0-indexed
                        chain_des_pos_list.append(resid)
            else:
                chain_des_pos_list.append(None)
        
        if all(des_pos is None for des_pos in chain_des_pos_list):
            raise RuntimeError(f'chain_des_pos_list returns None only for chain {chain_id}. Are you sure the chain has any designable positions?')

        #chain_des_pos_list.sort() # not sure why I wrote this? also won't work if there's any None in the list

        logger.debug(f'candidate_to_chain_des_pos() returned the following results:\n{sep}\ncandidate_des_pos_list: {candidate_des_pos_list}\nchain_id: {chain_id}; drop_terminal_missing_res: {drop_internal_missing_res}; drop_internal_missing_res: {drop_internal_missing_res}\nchain_des_pos_list: {chain_des_pos_list}\n{sep}\n')
        return chain_des_pos_list

    def parse_pdbs(self):
        combined_pdb_file_dir= tempfile.TemporaryDirectory()
        pdbs_list= glob.glob(f'{self.pdb_files_dir}/*.pdb')
        merge_pdb_files(pdbs_list, f'{combined_pdb_file_dir.name}/combined_pdb.pdb')

        out= tempfile.NamedTemporaryFile()
        exec_str= [
            sys.executable, f'{self.helper_scripts_dir}/parse_multiple_chains.py',
            f'--input_path={combined_pdb_file_dir.name}',
            f'--output_path={out.name}'
        ]
        proc= subprocess.run(exec_str, stdout= subprocess.PIPE, stderr= subprocess.PIPE, check= False)
        if proc.stderr:
                raise RuntimeError(f'Command {proc.args} returned non-zero exist status {proc.returncode} with the stderr\n{proc.stderr.decode()}')
        logger.debug(f'parse_multiple_chains.py was called with the following command:\n{sep}\n{exec_str}\nstdout:{proc.stdout.decode()}\nstderr: {proc.stderr.decode()}\n{sep}\n')

        combined_pdb_file_dir.cleanup()

        with open(out.name, 'r') as f:
            parsed_pdb_json= json.load(f)

        # correct any discrepancy between the sequence given by the chain objects vs. the sequence read in from the pdb files
        # this will update both the designable and non-designed chains that are provided
        for chain_id in self.chains_dict.keys():
            old_full_seq= parsed_pdb_json[f'seq_chain_{chain_id}']
            new_full_seq= self.get_chain_full_seq(
                chain_id,
                candidate= None, 
                drop_terminal_missing_res= True, 
                drop_internal_missing_res= False, 
                replace_missing_residues_with= '-'
            )

            if old_full_seq != new_full_seq:
                if len(old_full_seq) != len(new_full_seq):
                    raise IndexError(f'The chain {chain_id} sequence parsed from the pdb file does not have the same length as that parsed from the inputs (after removing terminal missing residues):\n\{sep}\npdb_seq: {old_full_seq}\nparsed_seq: {new_full_seq})\n{sep}\n')
                logger.warning(f'The chain {chain_id} sequence parsed from the pdb file ({old_full_seq} is not the same as that parsed from the inputs ({new_full_seq}))')
            logger.debug(f'parse_pdbs() chain {chain_id} sequence is updated:\n{sep}\nold_seq: {old_full_seq}\nnew_seq: {new_full_seq}\n{sep}\n')
            
            parsed_pdb_json[f'seq_chain_{chain_id}']= new_full_seq
        # update the full concatenated seq
        cumulative_seq= ''
        for key in parsed_pdb_json.keys():
            if 'seq_chain_' in key:
                cumulative_seq+= parsed_pdb_json[key]
        parsed_pdb_json['seq']= cumulative_seq
        logger.debug(f'parsed_pdbs() cumulative seq is updated:\n{sep}\nold_seq: {parsed_pdb_json["seq"]}\nnew_seq: {cumulative_seq}\n{sep}\n')

        with open(out.name, 'w') as f:
            json.dump(parsed_pdb_json, f)
            f.seek(0)

        return {'json': parsed_pdb_json, 'exec_str': 'jsonl_path'}, out
    
    def parse_fixed_chains(self, parsed_pdb_handle):
        chains_to_design= self.design_seq.chains_to_design
        if self.parsed_pdb_json['json']['num_of_chains'] > len(chains_to_design):
            chains_to_design_str= ' '.join(chains_to_design)
            out= tempfile.NamedTemporaryFile()
            exec_str= [
                sys.executable, f'{self.helper_scripts_dir}/assign_fixed_chains.py',
                f'--input_path={parsed_pdb_handle.name}',
                f'--output_path={out.name}',
                f'--chain_list={chains_to_design_str}'
            ]
            proc= subprocess.run(exec_str, stdout= subprocess.PIPE, stderr= subprocess.PIPE, check= False)
            if proc.stderr:
                raise RuntimeError(f'Command {proc.args} returned non-zero exist status {proc.returncode} with the stderr\n{proc.stderr.decode()}')
            logger.debug(f'assign_fixed_chains.py was called with the following command:\n{sep}\n{exec_str}\nstdout:{proc.stdout.decode()}\nstderr: {proc.stderr.decode()}\n{sep}\n')

            with open(out.name, 'r') as f:
                parsed_fixed_chains= json.load(f)
            out.close()

            return {'json': parsed_fixed_chains, 'exec_str': 'chain_id_jsonl'}
        else:
            return None
        
    def parse_fixed_positions(self, parsed_pdb_handle):
        fixed_pos_str= []
        chains_str= []
        for chain_id, des_pos in self.design_seq.chain_des_pos_dict.items():
            init_resid, fin_resid= self.chains_dict[chain_id].resid_range
            whole_chain= np.arange(init_resid, fin_resid + 1)
            fixed_pos= np.setdiff1d(whole_chain, np.asarray(des_pos, dtype= int))
            if fixed_pos.size == 0:
                pass
            else:
                fixed_pos_offset= fixed_pos - (init_resid - 1) # - 1 because we want the result to be in 1-index
                fixed_pos_str.append(' '.join(map(str, fixed_pos_offset)))
                chains_str.append(chain_id)
        
        if len(fixed_pos_str) == 0:
            logger.debug(f'make_fixed_positions_dict.py was not called because there are no fixed positions in the designable chains.')
            return None
        else:
            fixed_pos_str= ', '.join(fixed_pos_str)
            chains_str= ' '.join(chains_str)

            out= tempfile.NamedTemporaryFile()
            exec_str= [
                sys.executable, f'{self.helper_scripts_dir}/make_fixed_positions_dict.py',
                f'--input_path={parsed_pdb_handle.name}',
                f'--output_path={out.name}',
                f'--chain_list={chains_str}',
                f'--position_list={fixed_pos_str}'
            ]
            proc= subprocess.run(exec_str, stdout= subprocess.PIPE, stderr= subprocess.PIPE, check= False)
            if proc.stderr:
                raise RuntimeError(f'Command {proc.args} returned non-zero exist status {proc.returncode} with the stderr\n{proc.stderr.decode()}')
            logger.debug(f'make_fixed_positions_dict.py was called with the following command:\n{sep}\n{exec_str}\nstdout:{proc.stdout.decode()}\nstderr: {proc.stderr.decode()}\n{sep}\n')

            with open(out.name, 'r') as f:
                parsed_fixed_positions= json.load(f)
            out.close()

            return {'json': parsed_fixed_positions, 'exec_str': 'fixed_positions_jsonl'}
    
    def parse_tied_positions(self):
        tied_lists= []
        for tied_residue in self.design_seq:
            tied_list= tied_list= '{' + ', '.join([f'''"{residue.chain_id}": [[{residue.resid - (self.chains_dict[residue.chain_id].init_resid - 1)}], [{residue.weight}]]''' for residue in tied_residue]) + '}'
            tied_lists.append(tied_list)

        combined_tied_list= '{"' + 'combined_pdb' + '": [' + ', '.join(tied_lists) + ']}'
        logger.debug(f'parse_tied_positions() returned the following results:\n{sep}\n{combined_tied_list}\n{sep}\n')
        parsed_tied_positions= json.loads(combined_tied_list)

        return {'json': parsed_tied_positions, 'exec_str': 'tied_positions_jsonl'}
    
    def parse_omit_AA(self):
        raise NotImplementedError()

    def get_CA_coords(self, chain_id):
        CA_coords= np.asarray(self.parsed_pdb_json['json'][f'coords_chain_{chain_id}'][f'CA_chain_{chain_id}'])
        logger.debug(f'get_CA_coords() returned the following results:\n{sep}\nchain_id: {chain_id}\nCA_coords: {CA_coords}\n{sep}\n')
        return CA_coords

    def dump_jsons(self):
        out_dir= tempfile.TemporaryDirectory()
        outputs= [
            self.parsed_pdb_json,
            self.parsed_fixed_chains,
            self.parsed_fixed_positions,
            self.parsed_tied_positions,
        ]
        exec_str= []

        for output in outputs:
            if output is not None:
                output_loc= f'{out_dir.name}/{output["exec_str"]}.json'
                with open(output_loc, 'w') as f:
                    json.dump(output['json'], f)
                    exec_str+= ['--' + output['exec_str'], output_loc]
                #debug
                #output_loc= f'./{output["exec_str"]}.json'
                #with open(output_loc, 'w') as f:
                #    json.dump(output['json'], f)
        
        logger.debug(f'dump_jsons() returned the following exec_str:\n{sep}\n{exec_str}\n{sep}\n')
        return out_dir, exec_str

class DesignedProtein(Protein):
    def __init__(self, wt_protein, base_candidate, proposed_des_pos_list):
        if len(proposed_des_pos_list) == 0:
            raise RuntimeError(f'The proposed_des_pos_list for DesignedProtein is empty!')

        # update chain full_seq based on candidate
        if base_candidate is None:
            new_chains_list= wt_protein.chains_list.copy()
        else:
            new_chains_list= []
            for old_chain in wt_protein.chains_list:
                new_full_seq= wt_protein.get_chain_full_seq(
                    old_chain.chain_id,
                    candidate= base_candidate, 
                    drop_terminal_missing_res= False, 
                    drop_internal_missing_res= False, 
                    replace_missing_residues_with= None
                )
                new_chain= Chain(old_chain.chain_id, old_chain.init_resid, old_chain.fin_resid, old_chain.internal_missing_res_list, new_full_seq)
                new_chains_list.append(new_chain)
        
        # update design_seq based on proposed_des_pos_list
        new_design_seq= DesignSequence(*[wt_protein.design_seq.tied_residues[des_pos] for des_pos in proposed_des_pos_list])
        super().__init__(
            new_design_seq,
            new_chains_list,
            wt_protein.chains_neighbors_list,
            wt_protein.pdb_files_dir,
            wt_protein.helper_scripts_dir
        )
    
