import os, os.path, copy, json
from typing import Literal

import numpy as np
import torch

from int_seq_des.pmpnn_utils import _S_to_seq, tied_featurize, parse_fasta, _scores
from int_seq_des.pmpnn_utils import StructureDataset, ProteinMPNN


hidden_dim = 128
num_layers = 3 

def read_parsed_pdbs(
    jsonl_path,
    max_length= 200000
):
    return StructureDataset(
        jsonl_path, 
        truncate=None, 
        max_length=max_length, 
        verbose=False
    )

def load_model(
    seed= 0,
    model_name= 'v_48_020',
    path_to_model_weights= '',
    use_soluble_model= False,
    ca_only= False,
    backbone_noise= 0.00,
    force_cpu= True
):
    if path_to_model_weights:
        model_folder_path = path_to_model_weights
        if model_folder_path[-1] != '/':
            model_folder_path = model_folder_path + '/'
    else: 
        file_path = os.path.realpath(__file__)
        k = file_path.rfind("/")
        if ca_only:
            model_folder_path = file_path[:k] + '/ca_model_weights/'
        else:
            if use_soluble_model:
                model_folder_path = file_path[:k] + '/soluble_model_weights/'
            else:
                model_folder_path = file_path[:k] + '/vanilla_model_weights/'

    checkpoint_path = model_folder_path + f'{model_name}.pt'

    if force_cpu:
        device= 'cpu'
    else:
        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    rng= torch.Generator(device= device)
    rng.manual_seed(int(seed))

    checkpoint = torch.load(checkpoint_path, map_location=device) 
    model = ProteinMPNN(
        ca_only=ca_only, 
        num_letters=21, 
        node_features=hidden_dim, 
        edge_features=hidden_dim, 
        hidden_dim=hidden_dim, 
        num_encoder_layers=num_layers, 
        num_decoder_layers=num_layers, 
        augment_eps=backbone_noise, 
        k_neighbors=checkpoint['num_edges'],
        rng= rng
    )
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    alphabet_dict = dict(zip(alphabet, range(21)))   

    model_info_dict= {
        'device': device,
        'ca_only': ca_only,
        'model': model,
        'rng': rng,
        'alphabet': alphabet,
        'alphabet_dict': alphabet_dict
    }
    
    return model_info_dict

def design_proteins(
    model_info_dict,
    parsed_pdbs,
    num_seq_per_target= 1,
    batch_size= 1,
    sampling_temp= '0.1',
    chain_id_dict= None,
    fixed_positions_dict= None,
    omit_AAs= 'X',
    bias_AA_dict= None,
    bias_by_res_dict= None,
    omit_AA_dict= None,
    pssm_dict= None,
    pssm_multi= 0.0,
    pssm_threshold= 0.0,
    pssm_log_odds_flag= False,
    pssm_bias_flag= False,
    tied_positions_dict= None, 
): 
    model= model_info_dict['model']
    alphabet = model_info_dict['alphabet']

    dataset_valid= parsed_pdbs

    NUM_BATCHES = num_seq_per_target//batch_size
    BATCH_COPIES = batch_size
    temperatures = [float(item) for item in sampling_temp.split()]
    omit_AAs_list = omit_AAs
    omit_AAs_np = np.array([AA in omit_AAs_list for AA in alphabet]).astype(np.float32)

    bias_AAs_np = np.zeros(len(alphabet))
    if bias_AA_dict:
        for n, AA in enumerate(alphabet):
                if AA in list(bias_AA_dict.keys()):
                        bias_AAs_np[n] = bias_AA_dict[AA]

    with torch.no_grad():
        output_seqs= []
        for protein in dataset_valid:
            batch_clones = [copy.deepcopy(protein) for i in range(BATCH_COPIES)]
            (
                X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, 
                visible_list_list, masked_list_list, masked_chain_length_list_list, 
                chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, 
                tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, 
                bias_by_res_all, tied_beta
            ) = tied_featurize(
                batch_clones, model_info_dict['device'], chain_id_dict, 
                fixed_positions_dict, omit_AA_dict, tied_positions_dict, 
                pssm_dict, bias_by_res_dict, ca_only=model_info_dict['ca_only']
            )
            pssm_log_odds_mask = (pssm_log_odds_all > pssm_threshold).float() #1.0 for true, 0.0 for false
            
            # Generate some sequences
            for temp in temperatures:
                for j in range(NUM_BATCHES):
                    randn_2 = torch.randn(
                        chain_M.shape, 
                        device=X.device, 
                        generator= model_info_dict['rng']
                    )
                    if tied_positions_dict == None:
                        sample_dict = model.sample(
                            X, randn_2, S, chain_M, chain_encoding_all, 
                            residue_idx, mask=mask, temperature=temp, 
                            omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, 
                            chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, 
                            pssm_coef=pssm_coef, pssm_bias=pssm_bias, 
                            pssm_multi=pssm_multi, 
                            pssm_log_odds_flag=bool(pssm_log_odds_flag), 
                            pssm_log_odds_mask=pssm_log_odds_mask, 
                            pssm_bias_flag=bool(pssm_bias_flag), 
                            bias_by_res=bias_by_res_all,
                            rng= model_info_dict['rng']
                        )
                    else:
                        sample_dict = model.tied_sample(
                            X, randn_2, S, chain_M, chain_encoding_all, 
                            residue_idx, mask=mask, temperature=temp, 
                            omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, 
                            chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, 
                            pssm_coef=pssm_coef, pssm_bias=pssm_bias, 
                            pssm_multi=pssm_multi, 
                            pssm_log_odds_flag=bool(pssm_log_odds_flag), 
                            pssm_log_odds_mask=pssm_log_odds_mask, 
                            pssm_bias_flag=bool(pssm_bias_flag), 
                            tied_pos=tied_pos_list_of_lists_list[0], 
                            tied_beta=tied_beta, bias_by_res=bias_by_res_all,
                            rng= model_info_dict['rng']
                        )
                    S_sample = sample_dict["S"]
                    for b_ix in range(BATCH_COPIES):
                        masked_chain_length_list = masked_chain_length_list_list[b_ix]
                        masked_list = masked_list_list[b_ix]
                        seq = _S_to_seq(S_sample[b_ix], chain_M[b_ix])
                        native_seq = _S_to_seq(S[b_ix], chain_M[b_ix])
                        if b_ix == 0 and j==0 and temp==temperatures[0]:
                            start = 0
                            end = 0
                            list_of_AAs = []
                            for mask_l in masked_chain_length_list:
                                end += mask_l
                                list_of_AAs.append(native_seq[start:end])
                                start = end
                            native_seq = "".join(list(np.array(list_of_AAs)[np.argsort(masked_list)]))
                            l0 = 0
                            for mc_length in list(np.array(masked_chain_length_list)[np.argsort(masked_list)])[:-1]:
                                l0 += mc_length
                                native_seq = native_seq[:l0] + '/' + native_seq[l0:]
                                l0 += 1
                            output_seqs.append(native_seq)
                        start = 0
                        end = 0
                        list_of_AAs = []
                        for mask_l in masked_chain_length_list:
                            end += mask_l
                            list_of_AAs.append(seq[start:end])
                            start = end

                        seq = "".join(list(np.array(list_of_AAs)[np.argsort(masked_list)]))
                        l0 = 0
                        for mc_length in list(np.array(masked_chain_length_list)[np.argsort(masked_list)])[:-1]:
                            l0 += mc_length
                            seq = seq[:l0] + '/' + seq[l0:]
                            l0 += 1
                        output_seqs.append(seq)
    
    return output_seqs


def score_protein(
    model_info_dict,
    parsed_pdbs,
    score_mode: Literal['score_only', 'conditional_probs_only', 'conditional_probs_only_backbone', 'unconditional_probs_only'],
    path_to_fasta= '',
    num_seq_per_target= 1,
    batch_size= 1,
    chain_id_dict= None,
    fixed_positions_dict= None,
    bias_by_res_dict= None,
    omit_AA_dict= None,
    pssm_dict= None,
    tied_positions_dict= None, 
):
    model= model_info_dict['model']
    alphabet_dict= model_info_dict['alphabet_dict']

    dataset_valid= parsed_pdbs
  
    NUM_BATCHES = num_seq_per_target//batch_size
    BATCH_COPIES = batch_size 

    with torch.no_grad():
        # assume that we are given only one parsed protein
        assert len(dataset_valid) == 1
        protein= dataset_valid[0]
    
        batch_clones = [copy.deepcopy(protein) for i in range(BATCH_COPIES)]
        (
            X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, 
            visible_list_list, masked_list_list, masked_chain_length_list_list, 
            chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, 
            tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, 
            bias_by_res_all, tied_beta
        ) = tied_featurize(
            batch_clones, model_info_dict['device'], chain_id_dict, fixed_positions_dict, 
            omit_AA_dict, tied_positions_dict, pssm_dict, bias_by_res_dict, 
            ca_only=model_info_dict['ca_only']
        )
        
        if score_mode == 'score_only':
            loop_c = 0 
            if path_to_fasta:
                fasta_names, fasta_seqs = parse_fasta(path_to_fasta, omit=["/"])
                loop_c = len(fasta_seqs)
            
            outputs= {
                'pdb': [],
                'fasta': []
            }
            for fc in range(1+loop_c):
                native_score_list = []
                global_native_score_list = []
                if fc > 0:
                    input_seq_length = len(fasta_seqs[fc-1])
                    S_input = torch.tensor([alphabet_dict[AA] for AA in fasta_seqs[fc-1]], device=model_info_dict['device'])[None,:].repeat(X.shape[0], 1)
                    S[:,:input_seq_length] = S_input #assumes that S and S_input are alphabetically sorted for masked_chains
                for j in range(NUM_BATCHES):
                    randn_1 = torch.randn(chain_M.shape, device=X.device, generator= model_info_dict['rng'])
                    log_probs = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)
                    mask_for_loss = mask*chain_M*chain_M_pos
                    scores = _scores(S, log_probs, mask_for_loss)
                    native_score = scores.cpu().data.numpy()
                    native_score_list.append(native_score)
                    global_scores = _scores(S, log_probs, mask)
                    global_native_score = global_scores.cpu().data.numpy()
                    global_native_score_list.append(global_native_score)
                native_score = np.concatenate(native_score_list, 0)
                global_native_score = np.concatenate(global_native_score_list, 0)

                seq_str = _S_to_seq(S[0,], chain_M[0,])

                outputs['pdb' if fc == 0 else 'fasta'].append(
                    {
                        'score': native_score,
                        'global_score': global_native_score,
                        'S': S[0,].cpu().numpy(),
                        'seq_str': seq_str
                    }
                )
                
            return outputs

        elif score_mode in ['conditional_probs_only', 'conditional_probs_only_backbone']:
            if score_mode == 'conditional_probs_only_backbone':
                conditional_probs_only_backbone= True
            else:
                conditional_probs_only_backbone= False

            log_conditional_probs_list = []
            for j in range(NUM_BATCHES):
                randn_1 = torch.randn(chain_M.shape, device=X.device, generator= model_info_dict['rng'])
                log_conditional_probs = model.conditional_probs(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1, conditional_probs_only_backbone)
                log_conditional_probs_list.append(log_conditional_probs.cpu().numpy())
            concat_log_p = np.concatenate(log_conditional_probs_list, 0) #[B, L, 21]
            mask_out = (chain_M*chain_M_pos*mask)[0,].cpu().numpy()
            return [
                {
                    'log_p': concat_log_p,
                    'S': S[0,].cpu().numpy(),
                    'mask': mask[0,].cpu().numpy(),
                    'design_mask': mask_out
                }
            ]
        
        elif score_mode == 'unconditional_probs_only':
            log_unconditional_probs_list = []
            for j in range(NUM_BATCHES):
                log_unconditional_probs = model.unconditional_probs(X, mask, residue_idx, chain_encoding_all)
                log_unconditional_probs_list.append(log_unconditional_probs.cpu().numpy())
            concat_log_p = np.concatenate(log_unconditional_probs_list, 0) #[B, L, 21]
            mask_out = (chain_M*chain_M_pos*mask)[0,].cpu().numpy()
            return [
                {
                    'log_p': concat_log_p,
                    'S': S[0,].cpu().numpy(),
                    'mask': mask[0,].cpu().numpy(),
                    'design_mask': mask_out
                }
            ]
        
        else:
            raise ValueError()