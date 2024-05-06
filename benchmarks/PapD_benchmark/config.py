import os
from pgen.esm_sampler import ESM_sampler
from pgen.likelihood_esm import model_map

from int_seq_des.protein import Residue, TiedResidue, DesignSequence, Chain
from int_seq_des.wrapper import ObjectiveAF2Rank, ObjectiveESM, ObjectiveProteinMPNNNegLogProb
from int_seq_des.af_model import get_af_model_parameters


# specify absolute paths; update according to local setups
tmscore_exec= '/wynton/home/kortemme/lhong/software/TMscore/TMscore'
af2_params_dir= '/wynton/home/kortemme/lhong/software/af2rank/params'
protein_mpnn_helper_scripts_dir= '/wynton/home/kortemme/lhong/ProteinMPNN-main/helper_scripts'
protein_mpnn_weights_dir= '/wynton/home/kortemme/lhong/ProteinMPNN-main/vanilla_model_weights'
pdb_files_dir= f'{os.getcwd()}/pdb_files'
temp_dir= '/wynton/scratch/'

# define PapD complexes & design parameters
weights= [1., 1., 1., 1.]

# The first 21 residues in UniProt are the signal peptide and are not indexed in the PDB file
WT_seq= 'AVSLDRTRAVFDGSEKSMTLDISNDNKQLPYLAQAWIENENQEKIITGPVIATPPVQRLEPGAKSMVRLSTTPDISKLPQDRESLFYFNLREIPPRSEKANVLQIALQTKIKLFYRPAAIKTRPNEVWQDQLILNKVSGGYRIENPTPYYVTVIGLGGSEKQAEEGEFETVMLSPRSEQTVKSANYNTPYLSYINDYGGRPVLSFICNGSRCSVKKEK'

des_resids= [1, 3, 4, 5, 6, 7, 8, 31, 91, 104, 105, 106, 107, 108, 109, 110, 112, 152, 154, 163, 164, 166, 170, 194, 200]

tied_res_list= [
    TiedResidue(*[
        Residue(chain_id, resid, weight)
        for chain_id, weight in zip(['A', 'C', 'E', 'F'], weights)
    ])
    for resid in des_resids
]

tied_design_seq= DesignSequence(*tied_res_list)

chain_A_design_seq= DesignSequence(
    *[Residue('A', resid, 1.0) for resid in des_resids]
)
chain_C_design_seq= DesignSequence(
    *[Residue('C', resid, 1.0) for resid in des_resids]
)
chain_E_design_seq= DesignSequence(
    *[Residue('E', resid, 1.0) for resid in des_resids]
)
chain_F_design_seq= DesignSequence(
    *[Residue('F', resid, 1.0) for resid in des_resids]
)

chains_list= [
    # 1N0L
    # PapD: chain A
    # PapE: chain B (the first 24 res in UniProt are the signal peptide that is not indexed in the PDB file); in addition, res 1 is covalently attached to res 13
    Chain(
        chain_id= 'A', init_resid= 1, fin_resid= 215,
        internal_missing_res_list= [], full_seq= WT_seq
    ),
    Chain(
        chain_id= 'B', init_resid= 1, fin_resid= 149,
        internal_missing_res_list= [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 29, 30, 31, 32, 33, 34, 73, 74, 75, 76, 77, 128, 129, 130, 131, 132, 133, 134, 135, 136], 
        full_seq= 'VDNLTFRGKLIIPACTVSNTTVDWQDVEIQTLSQNGNHEKEFTVNMRCPYNLGTMKVTITATNTYNNAILVQNTSNTSSDGLLVYLYNSNAGNIGTAITLGTPFTPGKITGNNADKTISLHAKLGYKGNMQNLIAGPFSATATLVASYS'
    ),
    # 1PDK
    # PapD: chain A -> C
    # PapK: chain B -> D (the first 21 res in UniProt are the signal peptide that is not indexed in the PDB file)
    Chain(
        chain_id= 'C', init_resid= 1, fin_resid= 215,
        internal_missing_res_list= [], full_seq= WT_seq
    ),
    Chain(
        chain_id= 'D', init_resid= 9, fin_resid= 157,
        internal_missing_res_list= [],
        full_seq= 'SDVAFRGNLLDRPCHVSGDSLNKHVVFKTRASRDFWYPPGRSPTESFVIRLENCHATAVGKIVTLTFKGTEEAALPGHLKVTGVNAGRLGIALLDTDGSSLLKPGTSHNKGQGEKVTGNSLELPFGAYVVATPEALRTKSVVPGDYEATATFELTYR'
    ),
    # 1QPP
    # PapD: chain A -> E
    # PapD: chain B -> F
    Chain(
        chain_id= 'E', init_resid= 1, fin_resid= 214,
        internal_missing_res_list= [], full_seq= WT_seq
    ),
    Chain(
        chain_id= 'F', init_resid= 1, fin_resid= 214,
        internal_missing_res_list= [], full_seq= WT_seq
    ),
]

# define objective functions
chain_AB_pdb_name= '1n0l_chainAB_MET_719'
chain_CD_pdb_name= '1pdk_cleaned_203'
chain_EF_pdb_name= '1qpp_chainB_missing_res_197_asym_1000'

chain_AB_pdb_path= f'{pdb_files_dir}/{chain_AB_pdb_name}.pdb'
chain_CD_pdb_path= f'{pdb_files_dir}/{chain_CD_pdb_name}.pdb'
chain_EF_pdb_path= f'{pdb_files_dir}/{chain_EF_pdb_name}.pdb'

# load AF2 model parameters
haiku_parameters_dict= get_af_model_parameters(
    ['model_1_multimer_v3'], 
    af2_params_dir
)

def af2rank_chainAB(use_surrogate_tied_residues= False):
    return ObjectiveAF2Rank(
        chain_ids= ['A', 'B'],
        template_file_loc= chain_AB_pdb_path,
        tmscore_exec= tmscore_exec,
        params_dir= af2_params_dir,
        device= 'gpu',
        sign_flip= True,
        use_surrogate_tied_residues= use_surrogate_tied_residues,
        persistent= True,
        haiku_parameters_dict= haiku_parameters_dict
    )

def af2rank_chainCD(use_surrogate_tied_residues= False):
    return ObjectiveAF2Rank(
        chain_ids= ['C', 'D'],
        template_file_loc= chain_CD_pdb_path,
        tmscore_exec= tmscore_exec,
        params_dir= af2_params_dir,
        device= 'gpu',
        sign_flip= True,
        use_surrogate_tied_residues= use_surrogate_tied_residues,
        persistent= True,
        haiku_parameters_dict= haiku_parameters_dict
    )

def af2rank_chainEF(use_surrogate_tied_residues= False):
    return ObjectiveAF2Rank(
        chain_ids= ['E', 'F'],
        template_file_loc= chain_EF_pdb_path,
        tmscore_exec= tmscore_exec,
        params_dir= af2_params_dir,
        device= 'gpu',
        sign_flip= True,
        use_surrogate_tied_residues= use_surrogate_tied_residues,
        persistent= True,
        haiku_parameters_dict= haiku_parameters_dict
    )

def protein_mpnn_chainAB(use_surrogate_tied_residues= False):
    return ObjectiveProteinMPNNNegLogProb(
        chain_ids= ['A', 'B'],
        pdb_file_name= chain_AB_pdb_name,
        score_type='all_positions',
        num_seqs= 5,
        model_weights_loc= protein_mpnn_weights_dir,
        device= 'gpu',
        sign_flip= False,
        use_surrogate_tied_residues= use_surrogate_tied_residues
    )

def protein_mpnn_chainCD(use_surrogate_tied_residues= False):
    return ObjectiveProteinMPNNNegLogProb(
        chain_ids= ['C', 'D'],
        pdb_file_name= chain_CD_pdb_name,
        score_type='all_positions',
        num_seqs= 5,
        model_weights_loc= protein_mpnn_weights_dir,
        device= 'gpu',
        sign_flip= False,
        use_surrogate_tied_residues= use_surrogate_tied_residues
    )

def protein_mpnn_chainEF(use_surrogate_tied_residues= False):
    return ObjectiveProteinMPNNNegLogProb(
        chain_ids= ['E', 'F'],
        pdb_file_name= chain_EF_pdb_name,
        score_type='all_positions',
        num_seqs= 5,
        model_weights_loc= protein_mpnn_weights_dir,
        device= 'gpu',
        sign_flip= False,
        use_surrogate_tied_residues= use_surrogate_tied_residues
    )

esm_sampler_dict= {
    'esm1v': ESM_sampler(model_map['esm1v'](), device= 'cpu')
}

def esm_1v(design_chain= 'A'):
    return ObjectiveESM(
        chain_id= design_chain,
        model_name= 'esm1v',
        device= 'cpu',
        sign_flip= True,
        esm_sampler_dict= esm_sampler_dict
    )
