import sys
from pathlib import Path

from pymoo.operators.crossover.pntx import PointCrossover
from int_seq_des.ga_operator import MutationMethod, ProteinMutation
from int_seq_des.protein import Protein
from int_seq_des.wrapper import ProteinMPNNWrapper
from int_seq_des import run_nsga

import config

# Perform multistate CaM design with NSGA-III

# define benchmark configurations
batch_settings_dict= {}
batch_settings_ind= 1

for choose_pos_method, choose_AA_method, objective_type in zip(
    *[
        ['likelihood_ESM', 'likelihood_ESM'],
        ['ProteinMPNN-AD', 'ProteinMPNN-AD'],
        ['protein_mpnn',   'af2rank']
    ]
):
    for mutation_rate in [0.1, 0.3, 0.5]:
        batch_settings_dict[batch_settings_ind]= {}
        batch_settings_dict[batch_settings_ind]['choose_pos_method']= choose_pos_method
        batch_settings_dict[batch_settings_ind]['choose_AA_method']= choose_AA_method
        batch_settings_dict[batch_settings_ind]['objective_type']= objective_type
        batch_settings_dict[batch_settings_ind]['mutation_rate']= mutation_rate
        batch_settings_ind+= 1

batch_ind= int(sys.argv[1])

out_folder= Path('output')
out_folder.mkdir(parents=True, exist_ok=True)

out_file_name= f'output/ga_' + \
    '_'.join(
        map(
            str,
            [
                batch_settings_dict[batch_ind]['choose_pos_method'],
                batch_settings_dict[batch_ind]['choose_AA_method'],
                batch_settings_dict[batch_ind]['objective_type'],
                batch_settings_dict[batch_ind]['mutation_rate'],
                f'batch_{batch_ind}'
            ]
        )
    )# no suffix

# define protein and design problem
protein= Protein(
    design_seq= config.tied_design_seq,
    chains_list= config.chains_list,
    chains_neighbors_list= config.chains_neighbors_list,
    pdb_files_dir= config.pdb_files_dir,
    protein_mpnn_helper_scripts_dir= config.protein_mpnn_helper_scripts_dir
)

protein_mpnn= ProteinMPNNWrapper(
    protein= protein,
    temp= 0.3,
    model_weights_loc= config.protein_mpnn_weights_dir
)

# define objectives
if batch_settings_dict[batch_ind]['objective_type'] == 'af2rank':
    metrics_list= config.af2rank_list()
    observer_metrics_list= config.protein_mpnn_list() + [config.esm_1v()]
elif batch_settings_dict[batch_ind]['objective_type'] == 'af2rank+esm_1v':
    metrics_list= config.af2rank_list() + [config.esm_1v()]
    observer_metrics_list= config.protein_mpnn_list()
elif batch_settings_dict[batch_ind]['objective_type'] == 'protein_mpnn':
    metrics_list= config.protein_mpnn_list()
    observer_metrics_list= config.af2rank_list() + [config.esm_1v()]
elif batch_settings_dict[batch_ind]['objective_type'] == 'protein_mpnn+esm_1v':
    metrics_list= config.protein_mpnn_list() + [config.esm_1v()]
    observer_metrics_list= config.af2rank_list()
else:
    raise ValueError()

# define method parameters & run simulation
comm= None
root_seed= [batch_ind, 1270336]

pop_size= 100
n_generation= 50

crossover_operator= PointCrossover(n_points= 2)

mutation_method= MutationMethod(
    choose_pos_method= batch_settings_dict[batch_ind]['choose_pos_method'],
    choose_AA_method= batch_settings_dict[batch_ind]['choose_AA_method'], 
    prob= 1.0,
    mutation_rate= batch_settings_dict[batch_ind]['mutation_rate'],
    protein_mpnn= protein_mpnn, 
    esm_sampler_dict= config.esm_sampler_dict
)

mutation_operator= ProteinMutation(
    method_list= [mutation_method], 
    root_seed= root_seed, 
    pop_size= pop_size, 
    comm= comm,
    cluster_parallelization= False, 
    cluster_time_limit_str= '10:00:00', 
    cluster_mem_free_str= '10G',
    temp_dir= config.temp_dir
)

run_nsga(
    protein= protein,
    pop_size=pop_size, 
    n_generation=n_generation,
    mutation_operator=mutation_operator, 
    crossover_operator=crossover_operator, 
    metrics_list=metrics_list,
    root_seed=root_seed, 
    out_file_name=out_file_name, 
    saving_method= 'by_generation',
    observer_metrics_list= observer_metrics_list, 
    nsga_version= 3, # this needs NSGA-III because of the dimensionality
    comm= comm,
    cluster_parallelization= False, 
    cluster_parallelize_metrics= False,
    cluster_time_limit_str= '05:00:00', 
    cluster_mem_free_str= '10G',
    temp_dir= '/wynton/scratch',
    restart= False, 
    init_pop_file= None, 
    init_mutation_rate= 1.0
)
