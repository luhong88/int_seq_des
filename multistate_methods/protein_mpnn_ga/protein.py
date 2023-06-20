import glob, sys, logging, subprocess, tempfile, json, numpy as np
from scipy.spatial import distance_matrix
from Bio.PDB import PDBParser, PDBIO

logger= logging.getLogger(__name__)
c_handler= logging.StreamHandler()
c_handler.setLevel(logging.DEBUG)
c_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(c_handler)
sep= '-'*50

alphabet = 'ACDEFGHIKLMNPQRSTVWY'

def _equidistant_points(n_pts, mol_radius, min_dist):
    '''
    Create a list of equidistant points on a circle in the xy-plane
    The minimum distance between spheres of radius mol_radius centered at these points is 2*mol_radius+min_dist
    '''
    if n_pts > 1:
        theta= 2*np.pi/n_pts
        radius= (2*mol_radius + min_dist)/(2*np.sin(theta/2.))
        pos_list= np.asarray([[radius*np.cos(n*theta), radius*np.sin(n*theta), 0.] for n in range(n_pts)])
        return pos_list
    else:
        return np.array([[0., 0., 0.]])

def _merge_pdb_files(input_files, output_file, min_dist= 100):
        '''
        min_dist in Angstrom
        '''
        parser = PDBParser()

        structures= [parser.get_structure(file, file) for file in input_files]

        CA_coords_list= []
        for structure in structures:
            if len(structure) > 1:
                logger.warning(f'More than one models detected in {structure.id}; only the first model will be read and used!')
            CA_coords= []
            for chain in structure[0]:
                for residue in chain:
                        CA_coords.append(residue['CA'].get_coord())
            CA_coords_list.append(np.asarray(CA_coords))
        
        old_COM_list= [np.mean(CA_coords, axis= 0) for CA_coords in CA_coords_list]
        mol_radius_list= [np.max(np.linalg.norm(CA_coords - COM, axis= 1)) for CA_coords, COM in zip(CA_coords_list, old_COM_list)]
        new_COM_list= _equidistant_points(len(structures), np.max(mol_radius_list), min_dist)
        
        for structure, old_COM, new_COM in zip(structures, old_COM_list, new_COM_list):
            for chain in structure[0]:
                for residue in chain:
                    for atom in residue:
                        atom.transform(np.eye(3), new_COM - old_COM)
        
        merged_structure= structures[0].copy()
        if len(structures) > 1:
            for structure in structures[1:]:
                for chain in structure[0]:
                    merged_structure[0].add(chain)
        
        # Write the merged structure to the output file
        pdb_io= PDBIO()
        pdb_io.set_structure(merged_structure)
        pdb_io.save(output_file)

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
        chain_ids_unique= [*set(chain_ids)]
        if len(chain_ids_unique) != len(chain_ids):
            logger.warning(f'You are trying to tie together residues in the same chain; this has not been tested and may lead to unexpected behavior.')

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

        logger.debug(f'Chain init definition:\n{sep}\nchain_id: {chain_id}\ninit_resid: {init_resid}, fin_resid: {fin_resid}\ninternal_missing_res_list: {internal_missing_res_list}\nfull_seq: {full_seq}\m{sep}\n')

class Protein(object):
    def __init__(self, design_seq, chains_list, pdb_files_dir, protein_mpnn_helper_scripts_dir):
        '''
        Input structures must have the correct chain id
        It is okay to have gaps in the chains; resid 1-indexed
        
        '''
        self.design_seq= design_seq

        # chains_list should only include designable chains
        # ensure that the chains are listed in alphabetical order
        # need python >=3.7 to ensure the dict remembers insertion order
        chain_id_list= [chain.chain_id for chain in chains_list]
        chain_order= np.argsort(chain_id_list)
        self.chains_list= [chains_list[ind] for ind in chain_order]
        self.chains_dict= {chain.chain_id: chain for chain in self.chains_list}

        self.pdb_files_dir= pdb_files_dir
        self.helper_scripts_dir= protein_mpnn_helper_scripts_dir
        
        self.parsed_pdb_json, self.parsed_pdb_handle= self.parse_pdbs()
        self.parsed_fixed_chains= self.parse_fixed_chains()
        self.parsed_fixed_positions= self.parse_fixed_positions()
        self.parsed_tied_positions= self.parse_tied_positions()
        #TODO: add additional parsing options

        self.parsed_pdb_handle.close()
    
    def get_candidate(self):
        candidate= []
        for tied_res in self.design_seq.tied_residues:
            # use the first residue in a tied_residue as the representative
            rep_res= tied_res.residues[0]
            chain_id= rep_res.chain_id
            resid= rep_res.resid
            res_ind= resid - self.chains_dict[chain_id].init_resid
            candidate.append(self.chains_dict[chain_id].full_seq[res_ind])
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
                        full_seq[res.resid - init_resid]= candidate_AA
        
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

    def parse_pdbs(self):
        combined_pdb_file_dir= tempfile.TemporaryDirectory()
        pdbs_list= glob.glob(f'{self.pdb_files_dir}/*.pdb')
        _merge_pdb_files(pdbs_list, f'{combined_pdb_file_dir.name}/combined_pdb.pdb')

        out= tempfile.NamedTemporaryFile()
        exec_str= [
            sys.executable, f'{self.helper_scripts_dir}/parse_multiple_chains.py',
            f'--input_path={combined_pdb_file_dir.name}',
            f'--output_path={out.name}'
        ]
        proc= subprocess.run(exec_str, stdout= subprocess.PIPE, stderr= subprocess.PIPE, check= True)
        logger.debug(f'parse_multiple_chains.py was called with the following command:\n{sep}\n{exec_str}\nstdout:{proc.stdout}\nstderr: {proc.stderr}\n{sep}\n')

        combined_pdb_file_dir.cleanup()

        with open(out.name, 'r') as f:
            parsed_pdb_json= json.load(f)

        # correct any discrepancy between the sequence given by the chain objects vs. the sequence read in from the pdb files
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
                logger.warning(f'The chain {chain_id} sequence parsed from the pdb file ({old_full_seq} is not the same as that parsed from the inputs (new_full_seq))')
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
    
    def parse_fixed_chains(self):
        if self.parsed_pdb_json['json']['num_of_chains'] > len(self.chains_list):
            chains_to_design_str= ' '.join(self.design_seq.chains_to_design)
            out= tempfile.NamedTemporaryFile()
            exec_str= [
                sys.executable, f'{self.helper_scripts_dir}/assign_fixed_chains.py',
                f'--input_path={self.parsed_pdb_handle.name}',
                f'--output_path={out.name}',
                f'--chain_list={chains_to_design_str}'
            ]
            proc= subprocess.run(exec_str, stdout= subprocess.PIPE, stderr= subprocess.PIPE, check= True)
            logger.debug(f'assign_fixed_chains.py was called with the following command:\n{sep}\n{exec_str}\nstdout:{proc.stdout}\nstderr: {proc.stderr}\n{sep}\n')

            with open(out.name, 'r') as f:
                parsed_fixed_chains= json.load(f)
            out.close()

            return {'json': parsed_fixed_chains, 'exec_str': 'chain_id_jsonl'}
        else:
            return None
        
    def parse_fixed_positions(self):
        fixed_pos_str= []
        chains_str= []
        for chain_id, des_pos in self.design_seq.chain_des_pos_dict.items():
            init_resid, fin_resid= self.chains_dict[chain_id].resid_range
            whole_chain= np.arange(init_resid, fin_resid + 1)
            fixed_pos= np.setdiff1d(whole_chain, np.asarray(des_pos, dtype= int))
            if fixed_pos.size == 0:
                pass
            else:
                fixed_pos_offset= fixed_pos - (init_resid - 1)
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
                f'--input_path={self.parsed_pdb_handle.name}',
                f'--output_path={out.name}',
                f'--chain_list={chains_str}',
                f'--position_list={fixed_pos_str}'
            ]
            proc= subprocess.run(exec_str, stdout= subprocess.PIPE, stderr= subprocess.PIPE, check= True)
            logger.debug(f'make_fixed_positions_dict.py was called with the following command:\n{sep}\n{exec_str}\nstdout:{proc.stdout}\nstderr: {proc.stderr}\n{sep}\n')

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

    def get_CA_dist_matrices(self, chain_id):
        chain= self.chains_dict[chain_id]
        
        init_resid= chain.init_resid
        des_pos= self.design_seq.chain_des_pos_dict[chain_id]
        des_pos= np.sort(des_pos)

        ca_coords= np.asarray(self.parsed_pdb_json['json'][f'coords_chain_{chain_id}'][f'CA_chain_{chain_id}'])
        des_pos_arr= np.asarray(des_pos, dtype= int) - init_resid
        ca_coords_des= ca_coords[des_pos_arr]
        dist_mat= distance_matrix(ca_coords_des, ca_coords_des, p= 2)
        logger.debug(f'get_CA_dist_matrices() returned the following results:\n{sep}\n{dist_mat}\n{sep}\n')

        return dist_mat

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
                output_loc= f'./{output["exec_str"]}.json'
                with open(output_loc, 'w') as f:
                    json.dump(output['json'], f)
        
        logger.debug(f'dump_jsons() returned the following exec_str:\n{sep}\n{exec_str}\n{sep}\n')
        return out_dir, exec_str

class DesignedProtein(Protein):
    def __init__(self, wt_protein, base_candidate, proposed_des_pos_list):
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

        super().__init__(new_design_seq, new_chains_list, wt_protein.pdb_files_dir, wt_protein.helper_scripts_dir)
    
