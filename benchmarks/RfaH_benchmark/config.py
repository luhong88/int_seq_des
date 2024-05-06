import os
from int_seq_des.protein import Residue, TiedResidue, DesignSequence, Chain
from int_seq_des.wrapper import ObjectiveAF2Rank, ObjectiveESM, ObjectiveProteinMPNNNegLogProb

device= 'cpu'

# specify absolute paths; update according to local setups
tmscore_exec= '/wynton/home/kortemme/lhong/software/TMscore/TMscore'
af2_params_dir= '/wynton/home/kortemme/lhong/software/af2rank/params'
esm_script= '/wynton/home/kortemme/lhong/protein_gibbs_sampler/src/pgen/likelihood_esm.py'
protein_mpnn_helper_scripts_dir= '/wynton/home/kortemme/lhong/ProteinMPNN-main/helper_scripts'
protein_mpnn_weights_dir= '/wynton/home/kortemme/lhong/ProteinMPNN-main/vanilla_model_weights'
pdb_files_dir= f'{os.getcwd()}/pdb_files'
temp_dir= '/wynton/scratch/'

# define RfaH & design parameters
weights= [1., 1.]

WT_seq= 'MQSWYLLYCKRGQLQRAQEHLERQAVNCLAPMITLEKIVRGKRTAVSEPLFPNYLFVEFDPEVIHTTTINATRGVSHFVRFGASPAIVPSAVIHQLSVYKPKDIVDPATPYPGDKVIITEGAFEGFQAIFTEPDGEARSMLLLNLINKEIKHSVKNTEFRKL'

des_resids= list(range(119, 155))

tied_res_list= [
    TiedResidue(*[
        Residue(chain_id, resid, weight)
        for chain_id, weight in zip(['A', 'B'], weights)
    ])
    for resid in des_resids
]

tied_design_seq= DesignSequence(*tied_res_list)

chain_A_design_seq= DesignSequence(
    *[Residue('A', resid, 1.0) for resid in des_resids]
)
chain_B_design_seq= DesignSequence(
    *[Residue('B', resid, 1.0) for resid in des_resids]
)

chains_list= [
    Chain(
        chain_id= 'A', init_resid= 1, fin_resid= 155,
        internal_missing_res_list= list(range(98, 118)), full_seq= WT_seq
    ),
    Chain(
        chain_id= 'B', init_resid= 108, fin_resid= 162,
        internal_missing_res_list= [], full_seq= WT_seq
    ),
]

# define objective functions
chain_A_pdb_name= '5ond_chainA_cleaned_293'
chain_B_pdb_name= '2lcl_truncated_frame0_cleaned_160'

chain_A_pdb_path= f'{pdb_files_dir}/{chain_A_pdb_name}.pdb'
chain_B_pdb_path= f'{pdb_files_dir}/{chain_B_pdb_name}.pdb'

def af2rank_chainA(use_surrogate_tied_residues= False):
    return ObjectiveAF2Rank(
        chain_ids= ['A'],
        template_file_loc= chain_A_pdb_path,
        tmscore_exec= tmscore_exec,
        params_dir= af2_params_dir,
        device= device,
        sign_flip= True,
        use_surrogate_tied_residues= use_surrogate_tied_residues
    )

def af2rank_chainB(use_surrogate_tied_residues= False):
    return ObjectiveAF2Rank(
        chain_ids= ['B'],
        template_file_loc= chain_B_pdb_path,
        tmscore_exec= tmscore_exec,
        params_dir= af2_params_dir,
        device= device,
        sign_flip= True,
        use_surrogate_tied_residues= use_surrogate_tied_residues
    )

def protein_mpnn_chainA(use_surrogate_tied_residues= False):
    return ObjectiveProteinMPNNNegLogProb(
        chain_ids= ['A'],
        pdb_file_name= chain_A_pdb_name,
        score_type='all_positions',
        num_seqs= 5,
        model_weights_loc= protein_mpnn_weights_dir,
        device= device,
        sign_flip= False,
        use_surrogate_tied_residues= use_surrogate_tied_residues
    )

def protein_mpnn_chainB(use_surrogate_tied_residues= False):
    return ObjectiveProteinMPNNNegLogProb(
        chain_ids= ['B'],
        pdb_file_name= chain_B_pdb_name,
        score_type='all_positions',
        num_seqs= 5,
        model_weights_loc= protein_mpnn_weights_dir,
        device= device,
        sign_flip= False,
        use_surrogate_tied_residues= use_surrogate_tied_residues
    )

def esm_1v(design_chain= 'A'):
    return ObjectiveESM(
        chain_id= design_chain,
        script_loc= esm_script,
        model_name= 'esm1v',
        device= 'cpu',
        sign_flip= True,
    )
