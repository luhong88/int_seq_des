import textwrap, glob, sys, subprocess, tempfile, json

import numpy as np

from int_seq_des.utils import (
    sort_order, argsort, get_logger, sep, alphabet, merge_pdb_files
)

logger= get_logger(__name__)

class Residue(object):
    '''
    A class for storing information about designable residue positions.

    Attributes
    -----
    chain_id (str): the residue chain ID.

    resid (int): the residue ID; need to match the residue ID in the input PDB file.
    Note that this is NOT the 0-index or 1-index of the residue in the PDB file.

    weight (float): state weight for multistate design.

    alowed_AA (str): a list of residues (one-letter) allowed for design; by
    default all standard residue types are allowed. If Residue is used as part of 
    TiedResidue, this attribute will not be used. Currently, only the random 
    resetting operator utilizes this attribute.
    '''
    def __init__(self, chain_id, resid, weight, omit_AA= None):
        if not isinstance(chain_id, str):
            raise TypeError(f'chain_id {chain_id} is not a str.')
        if not (isinstance(resid, int) and resid > 0):
            raise TypeError(f'resid {resid} is not a positive integer (resid should be 1-indexed).')
        if not isinstance(weight, float):
            raise TypeError(f'weight {weight} is not a float.')
        
        self.chain_id= chain_id
        self.resid= resid
        self.weight= weight

        self.allowed_AA= alphabet
        if omit_AA is not None:
            if not isinstance(omit_AA, str):
                raise TypeError(f'omit_AA {omit_AA} is not a str.')
            for AA in list(omit_AA):
                self.allowed_AA= self.allowed_AA.replace(AA, '')

class TiedResidue(object):
    '''
    A class for tied residues. Tied residues are the basic unit of redesign in
    a multistate design problem.

    Attributes
    -----
    residues (list[Residue]): a variable number of Residue objects that will be
    tied for multistate design.

    alowed_AA (str): a list of residues (one-letter) allowed for design; by
    default all standard residue types are allowed. Will overwrite the alow_AA 
    attribute of the constituent Residue objects.

    chain_ids_unique (list[str]): a list of chain that contain the constituent
    Residue objects.
    '''
    def __init__(self, *residues, omit_AA= None):
        self.residues= residues
        self.allowed_AA= alphabet

        chain_ids= [res.chain_id for res in residues]
        self.chain_ids_unique= [*set(chain_ids)]

        if len(self.chain_ids_unique) != len(chain_ids):
            raise NotImplementedError(
                f'You are trying to tie together residues in the same chain; this feature is not implemented.'
            )

        if omit_AA is not None:
            if not isinstance(omit_AA, str):
                raise TypeError(f'omit_AA {omit_AA} is not a str.')
            for AA in list(omit_AA):
                self.allowed_AA= self.allowed_AA.replace(AA, '')
    
    def __iter__(self):
        return iter(self.residues)
    
    def __str__(self):
        return (
            '[' + \
            ', '.join(
                [f'{res.chain_id}.{res.resid}({res.weight})' for res in self]
            ) + \
            ']'
        )

class DesignSequence(object):
    '''
    A class that defines the set of designable positions.

    Attributes
    -----
    tied_residues (list[Residue | TiedResidue]): a list of residues and/or
    tied residues that will be allowed for redesign. Note that mixing Residue and
    TiedResidue objects is not well tested.

    has_no_tied_residues (bool): whether any of the designable positions are tied.

    n_des_res (int): the number of designable (tied) residues.

    chains_to_design (list[str]): a list of sorted chain IDs that contain all
    designable (tied) residues.

    chain_des_pos_dict (dict[str, list[str]]): a dictionary where the keys are the
    chain IDs and the corresponding values are lists of residue IDs for the (tied)
    residues in the chain.
    '''
    def __init__(self, *residues):
        self.tied_residues= residues

        self.has_no_tied_residues= True
        for tied_residue in self.tied_residues:
            if isinstance(tied_residue, TiedResidue):
                self.has_no_tied_residues= False

        self.n_des_res= len(self.tied_residues)

        chains_to_design= []
        for tied_residue in self.tied_residues:
            if isinstance(tied_residue, Residue):
                chains_to_design.append(tied_residue.chain_id)
            elif isinstance(tied_residue, TiedResidue):
                chains_to_design+= [residue.chain_id for residue in tied_residue]
            else:
                raise TypeError('Invalid input residue object!')
        chains_to_design= np.array(chains_to_design)
        
        chains_to_design= np.unique(chains_to_design.flatten())
        self.chains_to_design= sorted(list(chains_to_design), key= sort_order)

        self.chain_des_pos_dict= {chain: [] for chain in self.chains_to_design}
        for tied_residue in self.tied_residues:
            if isinstance(tied_residue, Residue):
                self.chain_des_pos_dict[tied_residue.chain_id].append(tied_residue.resid)
            elif isinstance(tied_residue, TiedResidue):
                for residue in tied_residue:
                    self.chain_des_pos_dict[residue.chain_id].append(residue.resid)
        
        logger.debug(
            textwrap.dedent(
                f'''\
                DesignSequence init definition:
                {sep}
                tied_residues: {str(self)}
                chains_to_design: {self.chains_to_design}
                chain_des_pos_dict: {self.chain_des_pos_dict}
                {sep}
                '''
            )
        )

    def __iter__(self):
        return iter(self.tied_residues)
    
    def __str__(self):
        return '[' + ', '.join([str(tied_res) for tied_res in self]) + ']'

class Chain(object):
    '''
    A class that defines a protein chain.

    Attributes
    -----
    chain_id (str)
    
    init_resid (int): the residue ID of the first residue in the chain; need to 
    match the input PDB.

    fin_resid (int): the residue ID of the last residue in the chain; need to
    match the input PDB.

    resid_range (list[int]): a list containing the init_resid and the fin_resid.

    internal_missing_res_list (list[int]): a list of residue IDs that are missing
    in the input PDB; note that missing residues before init_resid or after
    fin_resid do not count.

    full_seq (str): the full sequence of the chain, including all terminal and
    internal missing residues. Note that this does not need to agree with the
    sequence in the input PDB; the full_seq will override the PDB sequence during
    ProteinMPNN PDB parsing.

    is_designable (bool): whether the chain contains designable (tied) residues.
    This attribute is added when the Chain object is passed to a Protein object.

    neighbors_list (list[str]): a list of chain IDs with which the chain forms
    a protein complex. This attribute is added when the Chain object is passed
    to a Protein object.
    '''
    def __init__(
        self, 
        chain_id, 
        init_resid, 
        fin_resid, 
        internal_missing_res_list, 
        full_seq
    ):
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

        self.full_seq= full_seq 

        logger.debug(
            textwrap.dedent(
                f'''\
                Chain init definition:
                {sep}
                chain_id: {chain_id}
                init_resid: {init_resid}, fin_resid: {fin_resid}
                internal_missing_res_list: {internal_missing_res_list}
                full_seq: {full_seq}
                {sep}
                '''
            )
        )

class Protein(object):
    '''
    A class that defines the protein for redesign. In addition, handles generation
    and parsing of input json files for ProteinMPNN, as well as the conversion
    of different representations of protein sequences.

    Note that not all ProteinMPNN functionalities are reproduced in this class;
    e.g., omitting AA and adding a PSSM are currently not implemented.

    Attributes
    -----
    design_seq (DesignSequence): definition of the designable (tied) residues.

    chain_neighbors_list (list[list[str]]): a list of chain "neighbor" definitions;
    a set of chains are considered "neighbors" if they are part of a protein
    complex. For example, [['A', 'B'], ['C']] indicates that three chains need
    to be considered for the design problem; chains A and B form a complex with
    each other, but not with chain C.

    chains_list (list[Chain]): a list of chains in the input PDB files.
    When setting up a design problem, only designable chains are mandatory.
    Non-designable chains are mandatory if scored by AF2Rank. The chains_list
    is updated by __init__() in the sense that the list will be sorted by chain ID,
    and each chain will be annotated with the 'is_designable' and 'neighbors_list' 
    attributes.

    chains_dict (dict[str, Chain]): same as chains_list, but as a dictionary.

    surrogate_tied_residues_list (list[TiedResidue], None): a list of tied residues.
    In a multistate design problem, one might be interested in how sequences generated
    from single-state designs will be scored against an alternative state. In this
    scenario, Protein.design_seq should be defined by Residue objects to perform
    single-state design, but Protein.surrogate_tied_residues_list should contain
    TiedResidue objects for scoring so that the algorithm understands how to
    "translate" sequence changes made to one state to another state.

    pdb_files_dir (str): path to the directory containing all input PDB files.
    The files will be combined into a single PDB file and passed to ProteinMPNN.

    helper_scripts_dir (str): path to the ProteinMPNN 'helper_scripts' directory.
    This attribute belongs to Protein rather than ProteinMPNNWrapper because the
    Protein class handles the generation and parsing of all input json files.
    
    parsed_pdb_json (dict): the PDB parsing output; see parse_pdbs() for more details.

    parsed_fixed_chains (dict, None): the fixed chains parsing output; see
    parse_fixed_chains() for more details.

    parsed_fixed_positions (dict, None): the fixed positions parsing output;
    see parse_fixed_positions() for more details.

    parsed_tied_positions (dict, None): the tied positions parsing output;
    see parse_tied_positions() for more details.
    '''
    def __init__(
        self, 
        design_seq, 
        chains_list, 
        chains_neighbors_list, 
        pdb_files_dir, 
        protein_mpnn_helper_scripts_dir, 
        surrogate_tied_residues_list= None
    ):
        self.design_seq= design_seq
        self.chains_neighbors_list= chains_neighbors_list
        self.surrogate_tied_residues_list= surrogate_tied_residues_list

        updated_chains_list= []
        for chain in chains_list:
            # check if the chain is designable
            if chain.chain_id in self.design_seq.chains_to_design:
                chain.is_designable= True
            else:
                chain.is_designable= False
            logger.debug(
                f'chain {chain.chain_id} is marked with is_designable={chain.is_designable}'
            )
            # update the chains with neighbors list
            neighbors= []
            for neighbors_list in self.chains_neighbors_list:
                if chain.chain_id in neighbors_list:
                    neighbors+= neighbors_list
            neighbors= [*set(neighbors)]
            neighbors.sort(key= sort_order)
            chain.neighbors_list= neighbors
            updated_chains_list.append(chain)
            logger.debug(
                f'chain {chain.chain_id} updated with the following neighbors_list: {chain.neighbors_list}'
            )
        
        # check that the designable positions for each chain are present in the chain
        for chain in updated_chains_list:
            if chain.is_designable:
                chain_des_pos= self.design_seq.chain_des_pos_dict[chain.chain_id]
                for des_pos in chain_des_pos:
                    if not chain.init_resid <= des_pos <= chain.fin_resid:
                        raise IndexError(
                            f'Design position {des_pos} in chain {chain.chain_id} ' + \
                            f'({chain.init_resid}-{chain.fin_resid}) is out of bounds.'
                        )

        # ensure that the chains are listed in alphabetical order
        # need python >=3.7 to ensure the dict remembers insertion order
        chain_id_list= [chain.chain_id for chain in updated_chains_list]
        if len(chain_id_list) != len([*set(chain_id_list)]):
            raise ValueError(
                f'Duplicate chain definitions detected! (parsed chain_id_list: {chain_id_list})'
            )
        chain_order= argsort(chain_id_list)
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
        '''
        Translate self into a candidate.

        Output
        -----
        candidate (list[str]): a list of residues at the designable positions
        specified by self.design_seq. Note that the residue identities are not
        parsed from the full_seq attribute of Chain objects, rather than from
        input PDB files.
        '''
        candidate= []
        for tied_res in self.design_seq.tied_residues:
            if isinstance(tied_res, TiedResidue):
                # use the first residue in a tied_residue as the representative
                rep_res= tied_res.residues[0]
            elif isinstance(tied_res, Residue):
                rep_res= tied_res

            chain_id= rep_res.chain_id
            resid= rep_res.resid
            # for full seq with no missing residue, only need to convert from 1-index to 0-index
            candidate.append(self.chains_dict[chain_id].full_seq[resid - 1]) 
        logger.debug(f'get_candidate() returned: {candidate}\n')
        return candidate

    def get_chain_full_seq(
        self, 
        chain_id, 
        candidate, 
        drop_terminal_missing_res, 
        drop_internal_missing_res, 
        replace_missing_residues_with= None,
        use_surrogate_tied_residues= False # useful for single state designs
    ):
        '''
        Translate a candidate into the sequence of a specific chain.

        Input
        -----
        chain_id (str)

        candidte (list[str]): a design candidate represented as a list of residues
        at the designable positions.

        drop_terminal_missing_res (bool): whether to include terminal missing
        residues in the output sequence.

        drop_internal_missing_res (bool): whether to include internal missing
        residues in the output sequence.

        replace_missing_residues_with (str, None): for any missing residues that
        are not dropped, replace their residue types with this token; typically
        either '-' or 'X', depending on the model in question. If set to None
        (default), then do not perform replacement.

        use_surrogate_tied_residues (bool): set to True for single-state scoring;
        False by default.

        Output
        -----
        output_seq (str): sequence of 'chain_id' that has been updated based on
        'candidate'.
        '''  
        full_seq= np.array(
            list(self.chains_dict[chain_id].full_seq),
            dtype= object
        )
        init_resid, fin_resid= self.chains_dict[chain_id].resid_range

        if candidate is not None:
            if use_surrogate_tied_residues:
                tied_res_list= self.surrogate_tied_residues_list
            else:
                tied_res_list= self.design_seq.tied_residues
                
            for tied_res, candidate_AA in zip(tied_res_list, candidate):
                if isinstance(tied_res, TiedResidue):
                    for res in tied_res.residues:
                        if res.chain_id == chain_id:
                            #full_seq[res.resid - init_resid]= candidate_AA
                            full_seq[res.resid - 1]= candidate_AA
                elif isinstance(tied_res, Residue):
                    if tied_res.chain_id == chain_id:
                        full_seq[tied_res.resid - 1]= candidate_AA
        
        resid_arr= np.arange(len(full_seq)) + 1
        terminal_missing_res_mask= (resid_arr < init_resid) | (resid_arr > fin_resid)
        internal_missing_res_mask= np.isin(
            resid_arr, 
            self.chains_dict[chain_id].internal_missing_res_list
        )

        if replace_missing_residues_with is not None:
            full_seq[terminal_missing_res_mask | internal_missing_res_mask]= replace_missing_residues_with
        if drop_terminal_missing_res:
            full_seq[terminal_missing_res_mask]= None
        if drop_internal_missing_res:
            full_seq[internal_missing_res_mask]= None

        output_seq= ''.join(full_seq[full_seq != None])
        logger.debug(
            textwrap.dedent(
                f'''\
                get_chain_full_seq() returned the folllowing results:
                {sep}
                chain_id: {chain_id}
                candidate: {candidate}
                drop_terminal: {drop_terminal_missing_res}
                drop_internal: {drop_internal_missing_res}
                replace_missing: {replace_missing_residues_with}
                output_seq: {output_seq}
                {sep}
                '''
            )
        )
        return output_seq
    
    def candidate_to_chain_des_pos(
        self, 
        candidate_des_pos_list, 
        chain_id, 
        drop_terminal_missing_res= False, 
        drop_internal_missing_res= False
    ):
        '''
        Convert 0-index of a candidate (or, equivalently, self.design_seq.tied_residues)
        to 0-index of residues in a chain. The function depends on the assumption
        that residues on the same chain cannot be tied together. Useful in cases
        where one might want to select a subset of designable (tied) residues
        in a candidate and check chain-specific properties at the corresponding
        residue positions.

        Input
        -----
        candidate_des_pos_list (list[int]): a list of 0-indices, which map onto
        elements of self.design_seq.tied_residues.

        chain_id (str)

        drop_terminal_missing_res (bool): whether to count the terminal
        missing residues when converting to residue 0-indicies.

        drop_internal_missing_res (bool): whether to count the internal
        missing residues when converting to residue 0-indicies.

        Output
        -----
        chain_des_pos_list (list[int], None): a list of 0-indices, which map onto
        residues in 'chain_id', conditioned on the inclusion/exclusion of missing
        residues. If an element of candidate_des_pos_list cannot be mapped onto
        any residues in a chain, then the corresponding element in chain_des_pos_list
        is None.
        '''
        chain_des_pos_list= []
        for des_pos in candidate_des_pos_list:
            tied_res= self.design_seq.tied_residues[des_pos]
            if isinstance(tied_res, TiedResidue):
                if chain_id in tied_res.chain_ids_unique:
                    for res in tied_res:
                        if res.chain_id == chain_id:
                            resid= res.resid
                            if drop_internal_missing_res:
                                missing_list= self.chains_dict[chain_id].internal_missing_res_list
                                if resid in missing_list:
                                    raise ValueError(
                                        f'Designable residue {resid} in chain {chain_id} is ' + \
                                        'on the internal_missing_res_list of that chain, which should not be possible.'
                                    )
                                offset= sum(
                                    missing_resid < resid 
                                    for missing_resid in missing_list
                                )
                                resid-= offset
                            if drop_terminal_missing_res:
                                init_resid= self.chains_dict[chain_id].init_resid
                                # this automatically make resid 0-indexed
                                resid-= init_resid 
                            else:
                                resid-= 1 # make resid 0-indexed
                            chain_des_pos_list.append(resid)
                else:
                    chain_des_pos_list.append(None)

            elif isinstance(tied_res, Residue):
                if chain_id == tied_res.chain_id:
                    resid= tied_res.resid
                    if drop_internal_missing_res:
                        missing_list= self.chains_dict[chain_id].internal_missing_res_list
                        if resid in missing_list:
                            raise ValueError(
                                f'Designable residue {resid} in chain {chain_id} is ' + \
                                'on the internal_missing_res_list of that chain, which should not be possible.'
                            )
                        offset= sum(
                            missing_resid < resid
                            for missing_resid in missing_list
                        )
                        resid-= offset
                    if drop_terminal_missing_res:
                        init_resid= self.chains_dict[chain_id].init_resid
                        # this automatically make resid 0-indexed
                        resid-= init_resid 
                    else:
                        resid-= 1 # make resid 0-indexed
                    chain_des_pos_list.append(resid)
                else:
                    chain_des_pos_list.append(None)
        
        if all(des_pos is None for des_pos in chain_des_pos_list):
            raise RuntimeError(
                f'chain_des_pos_list returns None only for chain {chain_id}. ' + \
                'Are you sure the chain has any designable positions?'
            )

        #chain_des_pos_list.sort() # not sure why I wrote this? also won't work if there's any None in the list

        logger.debug(
            textwrap.dedent(
                f'''\
                candidate_to_chain_des_pos() returned the following results:
                {sep}
                candidate_des_pos_list: {candidate_des_pos_list}
                chain_id: {chain_id}
                drop_terminal_missing_res: {drop_internal_missing_res}
                drop_internal_missing_res: {drop_internal_missing_res}
                chain_des_pos_list: {chain_des_pos_list}
                {sep}
                '''
            )
        )
        return chain_des_pos_list

    def parse_pdbs(self, pdbs_list= None):
        combined_pdb_file_dir= tempfile.TemporaryDirectory()
        if pdbs_list is None:
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
                raise RuntimeError(
                    textwrap.dedent(
                        f'''\
                        Command {proc.args} returned non-zero exist status {proc.returncode} with the stderr
                        {sep}\n{proc.stderr.decode()}\n{sep}
                        '''
                    )
                )
        logger.debug(
            textwrap.dedent(
                f'''\
                parse_multiple_chains.py was called with the following command:
                {sep}
                {exec_str}
                stdout: {proc.stdout.decode()}
                stderr: {proc.stderr.decode()}
                {sep}
                '''
            )
        )

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
                    raise IndexError(
                        textwrap.dedent(
                            f'''\
                            The chain {chain_id} sequence parsed from the pdb file does not have the same length as that parsed from the inputs (after removing terminal missing residues):
                            {sep}
                            pdb_seq: {old_full_seq}
                            parsed_seq: {new_full_seq}
                            {sep}
                            '''
                        )
                    )
                logger.info(
                    f'The chain {chain_id} sequence parsed from the pdb file ({old_full_seq} ' + \
                    f'is not the same as that parsed from the inputs ({new_full_seq}))'
                )
            logger.debug(
                textwrap.dedent(
                    f'''\
                    parse_pdbs() chain {chain_id} sequence is updated:
                    {sep}
                    old_seq: {old_full_seq}
                    new_seq: {new_full_seq}
                    {sep}
                    '''
                )
            )
            
            parsed_pdb_json[f'seq_chain_{chain_id}']= new_full_seq
        # update the full concatenated seq
        cumulative_seq= ''
        for key in parsed_pdb_json.keys():
            if 'seq_chain_' in key:
                cumulative_seq+= parsed_pdb_json[key]
        parsed_pdb_json['seq']= cumulative_seq
        logger.debug(
            textwrap.dedent(
                f'''\
                parsed_pdbs() cumulative seq is updated:
                {sep}
                old_seq: {parsed_pdb_json["seq"]}
                new_seq: {cumulative_seq}
                {sep}
                '''
            )
        )

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
            proc= subprocess.run(
                exec_str, 
                stdout= subprocess.PIPE, 
                stderr= subprocess.PIPE, 
                check= False
            )
            if proc.stderr:
                raise RuntimeError(
                    textwrap.dedent(
                        f'''\
                        Command {proc.args} returned non-zero exist status {proc.returncode} with the stderr
                        {sep}\n{proc.stderr.decode()}\n{sep}
                        '''
                    )
                )
            logger.debug(
                textwrap.dedent(
                    f'''\
                    assign_fixed_chains.py was called with the following command:
                    {sep}
                    {exec_str}
                    stdout:{proc.stdout.decode()}
                    stderr: {proc.stderr.decode()}
                    {sep}
                    '''
                )
            )

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
                # - 1 because we want the result to be in 1-index
                fixed_pos_offset= fixed_pos - (init_resid - 1) 
                fixed_pos_str.append(' '.join(map(str, fixed_pos_offset)))
                chains_str.append(chain_id)
        
        if len(fixed_pos_str) == 0:
            logger.debug(
                f'make_fixed_positions_dict.py was not called because there are no fixed positions in the designable chains.'
            )
            return None
        else:
            fixed_pos_str= ', '.join(fixed_pos_str)
            chains_str= ' '.join(chains_str)

            out= tempfile.NamedTemporaryFile()
            exec_str= [
                sys.executable, 
                f'{self.helper_scripts_dir}/make_fixed_positions_dict.py',
                f'--input_path={parsed_pdb_handle.name}',
                f'--output_path={out.name}',
                f'--chain_list={chains_str}',
                f'--position_list={fixed_pos_str}'
            ]
            proc= subprocess.run(
                exec_str, 
                stdout= subprocess.PIPE, 
                stderr= subprocess.PIPE, 
                check= False
            )
            if proc.stderr:
                raise RuntimeError(
                    textwrap.dedent(
                        f'''\
                        Command {proc.args} returned non-zero exist status {proc.returncode} with the stderr
                        {sep}\n{proc.stderr.decode()}\n{sep}
                        '''
                    )
                )
            logger.debug(
                textwrap.dedent(
                    f'''\
                    make_fixed_positions_dict.py was called with the following command:
                    {sep}
                    {exec_str}
                    stdout:{proc.stdout.decode()}
                    stderr: {proc.stderr.decode()}
                    {sep}
                    '''
                )
            )

            with open(out.name, 'r') as f:
                parsed_fixed_positions= json.load(f)
            out.close()

            return {
                'json': parsed_fixed_positions,
                'exec_str': 'fixed_positions_jsonl'
            }
    
    def parse_tied_positions(self):
        if self.design_seq.has_no_tied_residues:
            return None
        else:
            tied_lists= []
            for tied_residue in self.design_seq:
                if isinstance(tied_residue, TiedResidue):
                    tied_list= (
                        '{' + \
                        ', '.join(
                            [
                                f'''"{residue.chain_id}": [[{residue.resid - (self.chains_dict[residue.chain_id].init_resid - 1)}], [{residue.weight}]]'''
                                for residue in tied_residue
                            ]
                        ) + \
                        '}'
                    )
                    tied_lists.append(tied_list)

            combined_tied_list= (
                '{"' + \
                'combined_pdb' + \
                '": [' + \
                ', '.join(tied_lists) + \
                ']}'
            )
            logger.debug(
                textwrap.dedent(
                    f'''\
                    parse_tied_positions() returned the following results:
                    {sep}\n{combined_tied_list}\n{sep}
                    '''
                )
            )
            parsed_tied_positions= json.loads(combined_tied_list)

            return {
                'json': parsed_tied_positions, 
                'exec_str': 'tied_positions_jsonl'
            }
    
    def parse_omit_AA(self):
        raise NotImplementedError()

    def get_CA_coords(self, chain_id):
        CA_coords= np.asarray(
            self.parsed_pdb_json['json'][f'coords_chain_{chain_id}'][f'CA_chain_{chain_id}']
        )
        logger.debug(
            textwrap.dedent(
                f'''\
                get_CA_coords() returned the following results:
                {sep}
                chain_id: {chain_id}
                CA_coords: {CA_coords}
                {sep}
                '''
            )
        )
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
        
        logger.debug(
            textwrap.dedent(
                f'''\
                dump_jsons() returned the following exec_str:
                {sep}\n{exec_str}\n{sep}
                '''
            )
        )
        return out_dir, exec_str

class DesignedProtein(Protein):
    def __init__(self, wt_protein, base_candidate, proposed_des_pos_list):
        if len(proposed_des_pos_list) == 0:
            raise RuntimeError(
                f'The proposed_des_pos_list for DesignedProtein is empty!'
            )

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
                new_chain= Chain(
                    old_chain.chain_id, 
                    old_chain.init_resid, 
                    old_chain.fin_resid, 
                    old_chain.internal_missing_res_list,
                    new_full_seq
                )
                new_chains_list.append(new_chain)
        
        # update design_seq based on proposed_des_pos_list
        new_design_seq= DesignSequence(
            *[
                wt_protein.design_seq.tied_residues[des_pos]
                for des_pos in proposed_des_pos_list
            ]
        )
        super().__init__(
            new_design_seq,
            new_chains_list,
            wt_protein.chains_neighbors_list,
            wt_protein.pdb_files_dir,
            wt_protein.helper_scripts_dir
        )
    
class SingleStateProtein(Protein):
    '''
    For ProteinMPNN scoring only
    '''
    def __init__(
        self, 
        multistate_protein, 
        chains_sublist, 
        pdb_file_name, 
        use_surrogate_tied_residues= False
    ):

        '''
        The chains in the chains_sublist and pdb file should match
        '''
        self.surrogate_tied_residues_list= multistate_protein.surrogate_tied_residues_list
        self.use_surrogate_tied_residues= use_surrogate_tied_residues

        chains_sublist.sort(key= sort_order)
        self.chains_sublist= chains_sublist
        
        if self.use_surrogate_tied_residues:
            tied_res_list= self.surrogate_tied_residues_list
        else:
            tied_res_list= multistate_protein.design_seq.tied_residues
            
        tied_res_sublist= []
        for tied_res in tied_res_list:
            if isinstance(tied_res, TiedResidue):
                res_sublist= [
                    res 
                    for res in tied_res 
                    if res.chain_id in chains_sublist
                ]
                new_tied_res= TiedResidue(*res_sublist)
                tied_res_sublist.append(new_tied_res)
            elif isinstance(tied_res, Residue):
                if tied_res.chain_id in chains_sublist:
                    tied_res_sublist.append(tied_res)
        self.design_seq= DesignSequence(*tied_res_sublist)

        self.chains_neighbors_list= [
            lst 
            for lst in multistate_protein.chains_neighbors_list 
            if set(chains_sublist) & set(lst)
        ]

        self.chains_list= [
            chain 
            for chain in multistate_protein.chains_list 
            if chain.chain_id in chains_sublist
        ]
        self.chains_dict= {chain.chain_id: chain for chain in self.chains_list}

        self.helper_scripts_dir= multistate_protein.helper_scripts_dir
        
        self.parsed_pdb_json, parsed_pdb_handle= self.parse_pdbs(
            pdbs_list= [f'{multistate_protein.pdb_files_dir}/{pdb_file_name}.pdb']
        )
        self.parsed_fixed_chains= self.parse_fixed_chains(parsed_pdb_handle)
        self.parsed_fixed_positions= self.parse_fixed_positions(parsed_pdb_handle)
        self.parsed_tied_positions= None

        parsed_pdb_handle.close()

        if len(chains_sublist) != self.parsed_pdb_json['json']['num_of_chains']:
            raise ValueError(
                f'The number of chains found in the input pdb file ({pdb_file_name}) ' + \
                f'does not match the number of chains provided in the chains_sublist ({chains_sublist})'
            )

        logger.debug(
            f'A SingleStateProtein object is created with chains {chains_sublist} and pdb file {pdb_file_name}'
        )

    def candidate_to_chain_des_pos(self, *args, **kwargs):
        raise AttributeError()
    
    def parse_tied_positions(self):
        raise AttributeError()
    
    def candidates_to_full_seqs(self, candidates):
        logger.debug(
            textwrap.dedent(
                f'''\
                candidates_to_full_seqs (SingleStateProtein) is called with the following candidates:
                {sep}\n{candidates}\n{sep}
                '''
            )
        )

        full_seqs= []
        for candidate in candidates:
            parsed_seqs= []
            for chain_id in self.chains_sublist:
                parsed_seq= self.get_chain_full_seq(
                    chain_id= chain_id, 
                    candidate= candidate, 
                    drop_terminal_missing_res= True, 
                    drop_internal_missing_res= False, 
                    replace_missing_residues_with= 'X', # ProteinMPNN uses X to represent gap
                    use_surrogate_tied_residues= self.use_surrogate_tied_residues
                )

                parsed_seqs.append(parsed_seq)

            full_seqs.append('/'.join(parsed_seqs))

        logger.debug(
            textwrap.dedent(
                f'''\
                candidates_to_full_seqs (SingleStateProtein) returned with the following sequences:
                {sep}\n{full_seqs}\n{sep}
                '''
            )
        )

        return full_seqs