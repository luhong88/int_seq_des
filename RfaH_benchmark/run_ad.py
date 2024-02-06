import sys
from pathlib import Path

from int_seq_des.protein import Protein
from int_seq_des.wrapper import ProteinMPNNWrapper
from int_seq_des import run_single_pass

import config

# Perform multistate RfaH design with tied decoding in ProteinMPNN

# define protein and design problem
protein= Protein(
    design_seq= config.tied_design_seq,
    chains_list= config.chains_list,
    chains_neighbors_list= [['A'], ['B']],
    pdb_files_dir= config.pdb_files_dir,
    protein_mpnn_helper_scripts_dir= config.protein_mpnn_helper_scripts_dir
)

protein_mpnn= ProteinMPNNWrapper(
    protein= protein,
    temp= 0.3,
    model_weights_loc= config.protein_mpnn_weights_dir
)

# define method parameters & run simulation
batch_ind= int(sys.argv[1])

out_folder= Path('output')
out_folder.mkdir(parents=True, exist_ok=True)

run_single_pass(
    protein=protein, 
    protein_mpnn=protein_mpnn, 
    design_mode= 'ProteinMPNN-AD', 
    metrics_list= [
        config.af2rank_chainA(), 
        config.af2rank_chainB(), 
        config.protein_mpnn_chainA(), 
        config.protein_mpnn_chainB(), 
        config.esm_1v()
    ], 
    num_seqs= 1, 
    protein_mpnn_batch_size= 1, 
    root_seed= [batch_ind, 81729], 
    out_file_name= str(out_folder / f'ad_batch_{batch_ind}'),
    comm= None,
    score_wt_only= False # set to True to score the WT sequence
)