import logging, pickle, numpy as np, pandas as pd
from Bio.PDB import PDBParser, PDBIO

from pymoo.core.callback import Callback
from pymoo.core.sampling import Sampling
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

from multistate_methods.protein_mpnn_ga.ga_operator import ProteinSampling, MultistateSeqDesignProblem

logger= logging.getLogger(__name__)
logger.propagate= False
logger.setLevel(logging.DEBUG)
c_handler= logging.StreamHandler()
c_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(c_handler)

sep= '-'*50

alphabet = 'ACDEFGHIKLMNPQRSTVWY'

class_seeds= {
    'run_single_pass': 60859570177,
    'ProteinMutation': 350671,
    'ProteinSampling': 501129
}

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

def merge_pdb_files(input_files, output_file, min_dist= 100):
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

def get_array_chunk(arr, rank, size):
    chunk_size= len(arr)/size
    if not chunk_size.is_integer():
        raise ValueError(f'It is not possible to evenly divide an array of length {len(arr)} into {size} processes.')
    else:
        chunk_size= int(chunk_size)
        
    return arr[rank*chunk_size:(rank + 1)*chunk_size]

class SavePop(Callback):
    def __init__(self, metric_list):
        super().__init__()
        self.data['pop'] = []
        self.metric_name_list= [str(metric) for metric in metric_list]

    def notify(self, algorithm):
        metrics= algorithm.pop.get('F')
        candidates= algorithm.pop.get('X')

        pop_df= pd.DataFrame(metrics, columns= self.metric_name_list)
        pop_df['candidate']= [''.join(candidate) for candidate in candidates]

        self.data['pop'].append(pop_df)

        logger.debug(f'SavePop returned the following population:\n{sep}\n{pop_df}\n{sep}\n')

class DumpPop(Callback):
    def __init__(self, metric_list, out_file_name, comm= None):
        super().__init__()
        self.metric_name_list= [str(metric) for metric in metric_list]
        self.out_file_name= out_file_name
        self.iteration= 0
        self.comm= comm

    def notify(self, algorithm):
        if (self.comm is None) or (self.comm.Get_rank() == 0):
            metrics= algorithm.pop.get('F')
            candidates= algorithm.pop.get('X')

            pop_df= pd.DataFrame(metrics, columns= self.metric_name_list)
            pop_df['candidate']= [''.join(candidate) for candidate in candidates]

            pickle.dump(pop_df, open(f'{self.out_file_name}_{self.iteration}.p', 'rb'))
            self.iteration+= 1

            logger.debug(f'SavePop dumped to file the following population:\n{sep}\n{pop_df}\n{sep}\n')

class LoadPop(Sampling):
    def __init__(self, pickle_file_loc):
        super().__init__()

        self.pickle_file_loc= pickle_file_loc

    def _do(self, problem, n_samples, **kwargs):
        existing_pop= pickle.load(open(self.pickle_file_loc, 'rb'))
        if isinstance(existing_pop, list):
            # if the pickle file is a list, assume that the last element is a df containing the final population
            proposed_candidates= np.array([list(candidate) for candidate in existing_pop[-1]['candidates']])
        elif isinstance(existing_pop, pd.core.frame.DataFrame):
            proposed_candidates= np.array([list(candidate) for candidate in existing_pop['candidates']])
        else:
            raise ValueError(f'Unrecognized data format detected from {self.pickle_file_loc}')
        
        logger.debug(f'LoadPop read in the following candidates:\n{sep}\n{proposed_candidates}\n{sep}\n')
        return proposed_candidates

def run_single_pass(
        protein,
        protein_mpnn, design_mode,
        metrics_list,
        num_seqs, protein_mpnn_batch_size,
        root_seed,
        out_file_name,
        comm= None):
    '''
    This is equivalent to calling ProteinMPNN-PD
    '''
    class_seed= class_seeds['run_single_pass']
    outputs= {}
    base_candidate= protein.get_candidate()

    if design_mode not in ['ProteinMPNN-PD', 'ProteinMPNN-AD']:
        raise KeyError(f'Unknown {design_mode} mode.')

    if comm is None:
        rng= np.random.default_rng([class_seed, root_seed])
        rank= None

        design_fa, chains_to_design= protein_mpnn.design(
            method= design_mode,
            base_candidate= base_candidate,
            proposed_des_pos_list= np.arange(protein.design_seq.n_des_res),
            num_seqs= num_seqs,
            batch_size= batch_size,
            seed= rng.integers(10000000000)
        )

        design_candidates= protein_mpnn.design_seqs_to_candidates(design_fa, chains_to_design, base_candidate)

        outputs['seq']= [str(fa.seq) for fa in design_fa[1:]]
        outputs['candidate']= [''.join(candidate) for candidate in design_candidates]

        for metric in metrics_list:
            outputs[str(metric)]= metric.apply(design_candidates, protein)
        
        outputs_df= pd.DataFrame(outputs)
        pickle.dump(outputs_df, open(out_file_name + '.p', 'rb'))
    
    else:
        rank= comm.Get_rank()
        size= comm.Get_size()
        rng= np.random.default_rng([class_seed, rank, root_seed])

        chunk_size= num_seqs/size
        if not chunk_size.is_integer():
            raise ValueError(f'It is not possible to evenly divide {num_seqs} sequences into {size + 1} processes.')
        else:
            chunk_size= int(chunk_size)
        
        batch_size= min(protein_mpnn_batch_size, chunk_size)

        design_fa, chains_to_design= protein_mpnn.design(
            method= 'ProteinMPNN-PD',
            base_candidate= base_candidate,
            proposed_des_pos_list= np.arange(protein.design_seq.n_des_res),
            num_seqs= chunk_size,
            batch_size= batch_size,
            seed= rng.integers(10000000000)
        )

        design_candidates= protein_mpnn.design_seqs_to_candidates(design_fa, chains_to_design, base_candidate)
        outputs['seq']= [str(fa.seq) for fa in design_fa[1:]]
        outputs['candidate']= [''.join(candidate) for candidate in design_candidates]

        for metric in metrics_list:
            outputs[str(metric)]= metric.apply(design_candidates, protein)
        
        outputs_df= pd.DataFrame(outputs)

        outputs_df_list= comm.gather(outputs_df, root= 0)
        if rank == 0:
            outputs_df= pd.concat(outputs_df_list, ignore_index= True)
            pickle.dump(outputs_df, open(out_file_name + '.p', 'rb'))

def run_nsga2(
        protein, protein_mpnn,
        pop_size, n_generation,
        mutation_operator, crossover_operator, metrics_list,
        root_seed, out_file_name, saving_method,
        comm= None,
        restart= False, init_pop_file= None
        ):
    
    if restart:
        pop_initializer= LoadPop(init_pop_file)
    else:
        pop_initializer= ProteinSampling(root_seed= root_seed, comm= comm)

    algorithm= NSGA2(
        pop_size= pop_size,
        sampling= pop_initializer,
        crossover= crossover_operator,
        mutation= mutation_operator,
        eliminate_duplicates= False
    )

    design_problem= MultistateSeqDesignProblem(protein, protein_mpnn, metrics_list, comm)

    if saving_method == 'by_generation':
        results= minimize(
            design_problem,
            algorithm,
            ('n_gen', n_generation),
            seed= root_seed,
            verbose= False,
            callback= DumpPop(metrics_list, out_file_name),
            copy_algorithm= False
        )

    elif saving_method == 'by_termination':
        results= minimize(
            design_problem,
            algorithm,
            ('n_gen', n_generation),
            seed= root_seed,
            verbose= False,
            callback= SavePop(metrics_list),
            copy_algorithm= False
        )
        if (comm is None) or (comm.Get_rank() == 0):
            pickle.dump(results.algorithm.callback.data['pop'], open(out_file_name + '.p', 'wb'))
    else:
        raise KeyError(f'Unknown saving_method {saving_method}')