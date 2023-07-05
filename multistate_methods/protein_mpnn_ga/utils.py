import os, logging, pickle, numpy as np, pandas as pd
from Bio.PDB import PDBParser, PDBIO

from pymoo.core.callback import Callback
from pymoo.core.sampling import Sampling

sep= '-'*50

alphabet = 'ACDEFGHIKLMNPQRSTVWY'

class_seeds= {
    'run_single_pass': 605177,
    'ProteinMutation': 350671,
    'ProteinSampling': 501129,
}

def get_logger(module_name):
    logger= logging.getLogger(module_name)
    logger.propagate= False
    logger.setLevel(logging.WARN)
    c_handler= logging.StreamHandler()
    c_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(c_handler)

    return logger

logger= get_logger(__name__)

# a way to foce cpu computation
class Device(object):
    def __init__(self, device):
        self.device= device
    def __enter__(self):
        if self.device == 'cpu':
            os.environ['CUDA_VISIBLE_DEVICES']= ''
    def __exit__(self, type, value, traceback):
        if self.device == 'cpu':
            del os.environ['CUDA_VISIBLE_DEVICES']

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
        parser = PDBParser(QUIET= True)

        structures= [parser.get_structure(file, file) for file in input_files]

        CA_coords_list= []
        for structure in structures:
            if len(structure) > 1:
                logger.warning(f'More than one models detected in {structure.id}; only the first model will be read and used!')
            CA_coords= []
            for chain in structure[0]:
                for residue in chain:
                        try:
                            CA_coords.append(residue['CA'].get_coord())
                        except KeyError:
                            pass
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

            pickle.dump(pop_df, open(f'{self.out_file_name}_{self.iteration}.p', 'wb'))
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
            proposed_candidates= np.array([list(candidate) for candidate in existing_pop[-1]['candidate']])
        elif isinstance(existing_pop, pd.core.frame.DataFrame):
            proposed_candidates= np.array([list(candidate) for candidate in existing_pop['candidate']])
        else:
            raise ValueError(f'Unrecognized data format detected from {self.pickle_file_loc}')
        
        logger.debug(f'LoadPop read in the following candidates:\n{sep}\n{proposed_candidates}\n{sep}\n')
        return proposed_candidates