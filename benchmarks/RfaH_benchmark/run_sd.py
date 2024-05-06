import sys
from pathlib import Path

from int_seq_des.protein import Protein
from int_seq_des.wrapper import ProteinMPNNWrapper
from int_seq_des import run_single_pass

import config

# Perform single-state RfaH design with in ProteinMPNN

# read in commandline arguments
batch_ind= int(sys.argv[1])
design_chain= sys.argv[2]
alt_chain= 'B' if design_chain == 'A' else 'A'

# define protein and design problem
protein= Protein(
    design_seq= config.chain_A_design_seq if design_chain == 'A' else config.chain_B_design_seq,
    chains_list= config.chains_list,
    chains_neighbors_list= [[design_chain], [alt_chain]],
    pdb_files_dir= config.pdb_files_dir,
    protein_mpnn_helper_scripts_dir= config.protein_mpnn_helper_scripts_dir,
    surrogate_tied_residues_list= config.tied_res_list
)

protein_mpnn= ProteinMPNNWrapper(
    protein= protein,
    temp= 0.3,
    model_weights_loc= config.protein_mpnn_weights_dir
)

# define method parameters & run simulation
out_folder= Path('output')
out_folder.mkdir(parents=True, exist_ok=True)

run_single_pass(
    protein= protein, 
    protein_mpnn= protein_mpnn, 
    design_mode= 'ProteinMPNN-AD', 
    metrics_list= [
        config.af2rank_chainA(use_surrogate_tied_residues= False if design_chain == 'A' else True), 
        config.af2rank_chainB(use_surrogate_tied_residues= False if design_chain == 'B' else True), 
        config.protein_mpnn_chainA(use_surrogate_tied_residues= False if design_chain == 'A' else True), 
        config.protein_mpnn_chainB(use_surrogate_tied_residues= False if design_chain == 'B' else True), 
        config.esm_1v(design_chain)
    ], 
    num_seqs= 1, 
    protein_mpnn_batch_size= 1, 
    root_seed= [batch_ind, 1270336], 
    out_file_name= str(out_folder / f'sd_chain_{design_chain}_batch_{batch_ind}'), 
    comm= None
)