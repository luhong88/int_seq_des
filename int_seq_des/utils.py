import textwrap, os, sys, logging, time, multiprocessing, tempfile, pickle
from datetime import datetime

import numpy as np, pandas as pd
from Bio.PDB import PDBParser, PDBIO

from pymoo.core.callback import Callback
from pymoo.core.sampling import Sampling

sep= '-'*50

alphabet = 'ACDEFGHIKLMNPQRSTVWY'

class_seeds= {
    'run_single_pass': 605177,
    'ProteinMutation': 350671,
    'ProteinSampling': 501129
}

# This function can be passed to sort() or the following custom argsort() to
# sort chain id into capital letter, lowercase letter, and then numerics
def sort_order(x): return x.isnumeric(), int(x) if x.isnumeric() else x

def argsort(x):
    x_sorted= sorted(x, key= sort_order)
    arg_list= []
    for elem in x_sorted:
        arg_list.append(x.index(elem))
    return arg_list

def get_logger(module_name):
    logger= logging.getLogger(module_name)
    logger.propagate= False
    logger.setLevel(logging.WARN)
    c_handler= logging.StreamHandler()
    c_handler.setFormatter(
        logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    )
    logger.addHandler(c_handler)

    return logger

logger= get_logger(__name__)

def npz_to_dict(npz_obj):
    keys= npz_obj.files
    new_dict= {k: npz_obj[k] for k in keys}
    return new_dict

class Device(object):
    '''
    A context manager class for controlling GPU visibility. Currently work as a
    way to force CPU computation.
    '''
    def __init__(self, device):
        self.device= device
    def __enter__(self):
        if self.device == 'cpu':
            os.environ['CUDA_VISIBLE_DEVICES']= ''
            # limit pytorch to one process
            os.environ['OMP_NUM_THREADS']= '1'
            os.environ['MKL_NUM_THREADS']= '1'
            # Limit to single-threaded jax/xla operations
            # see https://github.com/google/jax/issues/743
            os.environ['XLA_FLAGS']= (
                "--xla_cpu_multi_thread_eigen=false "
                "intra_op_parallelism_threads=1"
            )
            #os.environ['TF_CPP_MIN_LOG_LEVEL']= '2'
    def __exit__(self, type, value, traceback):
        if self.device == 'cpu':
            del os.environ['CUDA_VISIBLE_DEVICES']
            del os.environ['XLA_FLAGS']

class NativeSeqRecovery(object):
    '''
    A metric/objective function class that computes native sequence recovery.
    '''
    def __init__(self):
        self.name= 'identity'
    
    def __str__(self):
        return self.name

    def apply(self, candidates, protein):
        WT_arr= np.asarray(protein.get_candidate())
        candidates_arr= np.asarray(candidates)
        identities= np.sum(candidates_arr == WT_arr, axis= -1)/len(WT_arr)
        return identities

def _equidistant_points(n_pts, mol_radius, min_dist):
    '''
    Create a list of 'n_pts' equidistant points on a circle in the xy-plane. The 
    minimum distance between spheres of radius 'mol_radius' centered at these points is 
    2*mol_radius + min_dist.
    '''
    if n_pts > 1:
        theta= 2*np.pi/n_pts
        radius= (2*mol_radius + min_dist)/(2*np.sin(theta/2.))
        pos_list= np.asarray(
            [
                [radius*np.cos(n*theta), radius*np.sin(n*theta), 0.]
                for n in range(n_pts)
            ]
        )
        return pos_list
    else:
        return np.array([[0., 0., 0.]])

def _lattice_points(n_pts, mol_radius, min_dist):
    '''
    Create a cubic lattice grid for 'n_pts' such that the length of the unit 
    cell is 2*mol_radius + min_dist.
    '''
    unit_len= 2*mol_radius + min_dist
    lattice_dim= int(np.ceil(np.cbrt(n_pts)))
    grid= np.mgrid[:1:lattice_dim*1j, :1:lattice_dim*1j, :1:lattice_dim*1j].reshape(3, -1).T
    lattice_pts= (lattice_dim - 1)*unit_len*grid - (lattice_dim - 1)*unit_len/2
    lattice_subset= lattice_pts[:n_pts]
    lattice_subset_centered= lattice_subset - np.mean(lattice_subset, axis= 0)
    return lattice_subset_centered

def merge_pdb_files(input_files, output_file, min_dist= 24):
    '''
    Merge a set of PDB structures into a single one. If there are four or fewer
    input PDB files, then use the _equidistant_points() function to set the
    centroid locations for each PDB structure, otherwise use the _equidistant_points()
    function. All distance units in Angstrom.

    Input
    -----
    input_files ([list[str]]): a list of paths to the input PDB files.

    output_file (str): path to an output combined PDB file.

    min_dist (float): min_dist argument for _equidistant_points() and _equidistant_points().
    '''
    parser = PDBParser(QUIET= True)

    structures= [parser.get_structure(file, file) for file in input_files]

    CA_coords_list= []
    for structure in structures:
        if len(structure) > 1:
            logger.warning(
                f'More than one models detected in {structure.id}; only the first model will be read and used!'
            )
        CA_coords= []
        for chain in structure[0]:
            for residue in chain:
                    try:
                        CA_coords.append(residue['CA'].get_coord())
                    except KeyError:
                        pass
        CA_coords_list.append(np.asarray(CA_coords))
    
    old_COM_list= [np.mean(CA_coords, axis= 0) for CA_coords in CA_coords_list]
    mol_radius_list= [
        np.max(np.linalg.norm(CA_coords - COM, axis= 1))
        for CA_coords, COM in zip(CA_coords_list, old_COM_list)
    ]
    if len(structures) <= 4:
        new_COM_list= _equidistant_points(
            len(structures), np.max(mol_radius_list), min_dist
        )
    else:
        new_COM_list= _lattice_points(
            len(structures), np.max(mol_radius_list), min_dist
        )
    
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
    '''
    Split an array 'arr' into chunks of size 'size' and then get the chunk indexed
    by 'rank'.
    '''
    chunk_size= len(arr)/size
    if not chunk_size.is_integer():
        raise ValueError(
            f'It is not possible to evenly divide an array of length {len(arr)} into {size} processes.'
        )
    else:
        chunk_size= int(chunk_size)
        
    return arr[rank*chunk_size:(rank + 1)*chunk_size]

def sge_write_submit_script(
    sge_script_loc, 
    job_name, 
    time_limit_str, 
    mem_free_str, 
    python_str
):
    '''
    Write an SGE job submission script that requests a single core and executes
    a python command. This function needs to be customized based on the specifics
    of the computing environment.

    Input
    -----
    sge_script_loc (str): a path for the output job submission script.
    
    job_name (str): name for the SGE job.

    time_limit_str (str): SGE job time limit.

    mem_free_str (str): SGE job memory limit.

    python_str (str): a string of python code to be executed.
    '''
    job_name= job_name[:100] # truncate long job names
    
    # use the -c tag to execute a string through the python interpreter
    submit_str= textwrap.dedent(
        f'''\
        #!/bin/bash
        #$ -cwd
        #$ -N {job_name}
        #$ -o {job_name}.out
        #$ -e {job_name}.err
        #$ -l h_rt={time_limit_str}
        {"#$ -l x86-64-v=3" if "af2rank" in job_name else ""}
        {"#$ -l mem_free="+str(mem_free_str) if mem_free_str is not None else ""}
        #$ -l h=!qb3-atgpu17&!qb3-atgpu17

        module use $HOME/software/modules
        module load python

        export TF_CPP_MIN_LOG_LEVEL=3
        export TF_ENABLE_ONEDNN_OPTS=0
        export CUDA_VISIBLE_DEVICES=""

        python3 -c '{python_str}'
        '''
    )
    with open(sge_script_loc, 'w') as f:
        f.write(submit_str)

def sge_submit_job(sge_script_loc):
    '''
    Submit an SGE job script and get the job ID.
    
    Input
    -----
    sge_script_loc (str): a path to an SGE job.

    Output
    -----
    job_id (int): the id of the submitted SGE job.
    '''
    current_time= datetime.now()
    rng= np.random.default_rng(
        [
            current_time.year, 
            current_time.month, 
            current_time.hour, 
            current_time.minute, 
            current_time.second, 
            current_time.microsecond
        ]
    )

    time_limit= 60*60 # 1 h
    start_time= time.time()
    
    # keep trying to submit the job until it is accepted by the job scheduler
    while time.time() - start_time < time_limit:
        try:
            time.sleep(rng.integers(10, 60))
            # return: <job_id>.<ini_ind>-<fin_ind>:<step>
            sge_out= os.popen(f'qsub -terse {sge_script_loc}').read() 
            job_id= int(sge_out.split('.')[0])
        except:
            print(f'exceptions detected while attempting to submit {sge_script_loc}')
            continue
        else:
            break
    return job_id

def cluster_manage_job(sge_script_loc, out_file, cluster_time_limit_str):
    '''
    Submit an SGE job script, monitor the status of the job, and return the job
    outputs if/when it finishes successfully.

    Input
    -----
    sge_script_loc (str): path to the SGE job script.

    out_file (str): path to the expected pickle output file.

    cluster_time_limit_str (str): SGE job time limit.

    Output
    -----
    results: output from the SGE job.
    '''
    current_time= datetime.now()
    rng= np.random.default_rng(
        [
            current_time.year, 
            current_time.month, 
            current_time.hour, 
            current_time.minute, 
            current_time.second, 
            current_time.microsecond
        ]
    )

    try_limit= 10 # how many times to try to (re)submit the job before giving up
    job_id= sge_submit_job(sge_script_loc)
    num_tries= 1
    
    submit_time= time.time()

    pt= datetime.strptime(cluster_time_limit_str, '%H:%M:%S')
    total_seconds= pt.second + pt.minute*60 + pt.hour*3600
    # how long to wait before giving up, this includes the job queueing time
    # this number can be adjusted to account for the business of the job queue
    time_limit= 2*total_seconds 
    
    while True:
        time.sleep(rng.integers(60, 180))
        job_is_Eqw= 'Eqw' in os.popen(f'qstat | grep {job_id}').read()
        if job_is_Eqw:
            os.popen(f'qdel -f {job_id}')
            time.sleep(rng.integers(10, 120))
            job_id= sge_submit_job(sge_script_loc)
            submit_time= time.time()

        job_is_running= os.popen(f'qstat -j {job_id} 2>/dev/null').read()
        if job_is_running:
            wall_time= time.time() - submit_time
            if wall_time < time_limit:
                continue
            elif num_tries < try_limit:
                os.popen(f'qdel -f {job_id}')
                time.sleep(60)
                job_id= sge_submit_job(sge_script_loc)
                submit_time= time.time()
                num_tries+= 1
                continue
            else:
                raise RuntimeError(
                    f'output file {out_file} does not exist for the job {sge_script_loc} ' + \
                    f'after {try_limit} attempts due to hanged jobs!'
                )
        else:
            time.sleep(10)
            if os.path.isfile(out_file):
                break
            elif num_tries < try_limit:
                job_id= sge_submit_job(sge_script_loc)
                submit_time= time.time()
                num_tries+= 1
                continue
            else:
                raise RuntimeError(
                    f'output file {out_file} does not exist for the job {sge_script_loc} ' + \
                    f'after {try_limit} attempts due to hanged jobs!'
                )
    
    results= pickle.load(open(out_file, 'rb'))
    return results

def cluster_act_single_candidate(
    actions_list, 
    candidate, 
    protein, 
    cluster_time_limit_str, 
    cluster_mem_free_str, 
    pkg_dir, 
    candidate_ind= None, 
    result_queue= None,
    temp_dir= None
):
    '''
    Score/mutate a candidate in an SGE job.

    Input
    -----
    actions_list (list): a list of "actions" to be performed on 'candidate',
    could be a metric/objective function, or a mutation operator.

    candidate (list[str]): a list of residues at the designable positions.

    protein (protein.Protein): details of the protein system and design parameters.

    cluster_time_limit_str (str): how much time to be allowed for the job on
    the compute cluster.

    cluster_mem_free_str (str): how much memory to be allocated for the job
    on the computer cluster.

    pkg_dir (str): absolute path of the parent directory of this script.

    candidate_ind (int, None): an index associated with the 'candidate'; cannot
    be None if 'result_queue' is not None.

    result_queue (multiprocessing.Queue, None): a multiprocessing queue for storing
    the SGE job output; relevant when SGE job submissions are managed using python
    multiprocessing.

    temp_dir (str, None): path to the folder in which to create a temporary directory
    to store the SGE calculation results. If None, then use the default system setting.

    Output
    -----
    If 'result_queue' is not None, then the function stores a tuple of 'candidate_ind'
    and the SGE job outputs in the 'result_queue'; otherwise, the function returns
    the SGE job outputs.
    '''
    logger.debug(
        f'cluster_act_single_candidate() get the candidate {candidate} from the full candidates set'
    )
    if result_queue is not None:
        assert candidate_ind is not None, \
            'candidate_ind cannot be None if a result_queue is passed to cluster_evaluate_single_candidate()'

    try:
        results= []
        for action in actions_list:
            if temp_dir is not None:
                job_dir= tempfile.TemporaryDirectory(dir= temp_dir)
            else:
                job_dir= tempfile.TemporaryDirectory()
            out_file= f'score.p'
            sge_script_file= 'submit.sh'

            pickle.dump(action, open(f'{job_dir.name}/action.p', 'wb'))
            pickle.dump(candidate, open(f'{job_dir.name}/candidate.p', 'wb'))
            pickle.dump(protein, open(f'{job_dir.name}/protein.p', 'wb'))

            python_exec_str= '; '.join(
                [
                    'import pickle, sys',
                    f'sys.path.insert(0, "{pkg_dir}")',
                    'action= pickle.load(open("action.p", "rb"))',
                    'candidate= pickle.load(open("candidate.p", "rb"))',
                    'protein= pickle.load(open("protein.p", "rb"))',
                    'result= action.apply(candidate, protein)',
                    f'pickle.dump(result, open("{out_file}", "wb"))'
                ]
            )
            
            parent_dir= os.getcwd()
            os.chdir(job_dir.name)
            sge_write_submit_script(
                sge_script_loc= sge_script_file,
                job_name= str(action),
                time_limit_str= cluster_time_limit_str,
                mem_free_str= cluster_mem_free_str,
                python_str= python_exec_str
            )
            result= cluster_manage_job(
                sge_script_file, 
                out_file, 
                cluster_time_limit_str
            )
            results.append(result)
            os.chdir(parent_dir)
            job_dir.cleanup()

        logger.debug(
            textwrap.dedent(
                f'''\
                cluster_act_single_candidate() received the following broadcasted results:
                {sep}\n{results}\n{sep}
                '''
            )
        )
        
        if candidate_ind is None:
            return results
        else:
            result_queue.put((candidate_ind, results))

    except Exception as e:
        if candidate_ind is None:
            raise e
        else:
            result_queue.put(e)

def evaluate_candidates(
    metrics_list,
    candidates,
    protein,
    pkg_dir,
    comm= None,
    cluster_parallelization= False,
    cluster_parallelize_metrics= False,
    cluster_time_limit_str= None, 
    cluster_mem_free_str= None
):
    '''
    Evaluate a list of candidates using a list of metrics/objective functions.

    This function operates similarly to ga_operator.ProteinMutation._do().

    Input
    -----
    metrics_list (list): a list of metrics/objective functions.

    candidates (list[list[str]]): a list of candidates; a list of residues at the 
    designable positions.

    protein (protein.Protein): details of the protein system and design parameters.

    pkg_dir (str): absolute path of the parent directory of this script.

    comm (mpi4py.MPI.Comm, None): a mpi4py communicator object (e.g., 
    mpi4py.MPI.COMM_WORLD). Set to None (default) to disable parallelization 
    with mpi4py.

    cluster_parallelization (bool): whether to parallelize calculations for each
    candidate as a job on a compute cluster. If this is set to True, the comm 
    argument is ignored. Set to False by default.

    cluster_parallelize_metrics (bool): whether to split the calculation of each
    metric for a candidate as separate jobs on a compute cluster. False by default.

    cluster_time_limit_str (str): how much time to be allowed for each job on
    the compute cluster. Required for cluster parallelization, but by default 
    set to None. The str should be formatted in a way that can be parsed by the
    SGE job scheduler.

    cluster_mem_free_str (str): how much memory to be allocated for each job
    on the computer cluster. Required for cluster parallelization, but by default 
    set to None. The str should be formatted in a way that can be parsed by the
    SGE job scheduler.

    Output
    -----
    scores (np.ndarray[float]): a (N, M) array of score outputs; N is the number
    of candidates, and M is the number of metrics/objective functions.
    '''
    if cluster_parallelization == True:
        if comm is not None:
            logger.info(
                'evaluate_candidates() is called with an MPI comm setting, ' + \
                'which will be ignored because cluster_parallelization == True'
            )

        jobs= []
        result_queue= multiprocessing.Queue()
        
        for candidate_ind, candidate in enumerate(candidates):
            if cluster_parallelize_metrics:
                for metric_ind, metric in enumerate(metrics_list):
                    # need to put candidate in [] because all the metrics are designed 
                    # to take in a list of candidates
                    proc= multiprocessing.Process(
                        target= cluster_act_single_candidate, 
                        args= (
                            [metric], 
                            [candidate], 
                            protein, 
                            cluster_time_limit_str, 
                            cluster_mem_free_str, 
                            pkg_dir, 
                            (candidate_ind, metric_ind), 
                            result_queue
                        )
                    )
                    jobs.append(proc)
                    proc.start()
            else:
                proc= multiprocessing.Process(
                    target= cluster_act_single_candidate, 
                    args= (
                        metrics_list, 
                        [candidate], 
                        protein, 
                        cluster_time_limit_str, 
                        cluster_mem_free_str, 
                        pkg_dir, 
                        candidate_ind, 
                        result_queue
                    )
                )
                jobs.append(proc)
                proc.start()
        
        # fetch results from the queue
        results_list= []

        for proc in jobs:
            results= result_queue.get()
            # exceptions are passed back to the main process as the result
            if isinstance(results, Exception):
                sys.exit('%s (found in %s)' %(results, proc))
            results_list.append(results)

        for proc in jobs:
            proc.join()
        
        # unscramble the returned results
        if cluster_parallelize_metrics:
            scores= np.empty((len(candidates), len(metrics_list)))
            scores[:]= np.nan
            for result_ind, result in results_list:
                candidate_ind, metric_ind= result_ind
                scores[candidate_ind, metric_ind]= np.squeeze(result)
            has_missing_values= np.any(np.isnan(scores))
            assert not has_missing_values, \
                'some scores are not returned by multiprocessing!'
            logger.debug(
                f'evaluate_candidates() (cluster, cluster_parallelize_metrics == True) ' + \
                f'received the following broadcasted scores:\n{sep}\n{scores}\n{sep}\n'
            )

        else:
            new_results_order= [result[0] for result in results_list]
            assert sorted(new_results_order) == list(range(len(candidates))), \
                'some scores are not returned by multiprocessing!'
            results_list_sorted= sorted(zip(new_results_order, results_list))
            scores= np.squeeze([result[1][1] for result in results_list_sorted])
            logger.debug(
                f'evaluate_candidates() (cluster, cluster_parallelize_metrics == False) ' + \
                f'received the following broadcasted scores:\n{sep}\n{scores}\n{sep}\n'
            )
            
        return scores
    
    else:
        scores= []
        if comm is None:
            for metric in metrics_list:
                scores.append(metric.apply(candidates, protein))
            scores= np.vstack(scores).T
        else:
            rank= comm.Get_rank()
            size= comm.Get_size()
            
            candidates_subset= get_array_chunk(candidates, rank, size)
            logger.debug(
                f'scores (rank {rank}; observer) get the ' + \
                f'candidates_subset {candidates_subset} from the full candidates {candidates}'
            )
            for metric in metrics_list:
                scores.append(metric.apply(candidates_subset, protein))
            # reshape scores from (n_metric, n_candidate) to (n_candidate, n_metric)
            scores= np.vstack(scores).T 
            scores= comm.gather(scores, root= 0)
            if rank == 0: scores= np.vstack(scores)
            scores= comm.bcast(scores, root= 0)
            logger.debug(
                f'evaluate_candidates() (rank {rank}/{size}) ' + \
                f'received the following broadcasted scores:\n{sep}\n{scores}\n{sep}\n'
            )

        return scores

class SavePop(Callback):
    '''
    A subclass of pymoo.core.callback.Callback. The purpose of this class is to
    record the genetic algorithm simulation trajectory and store it in the final
    algorithm output. It takes the population from each iteration of the genetic 
    algorithm, and packages the candidates and objective function scores in a 
    pd.Dataframe, which is then appended to self.data['pop']. This class is
    provided as an argument to pymoo.optimize.minimize() and the stored data
    is available in the minimize() output.
    '''
    def __init__(
        self, 
        protein, 
        metrics_list, 
        observer_metrics_list, 
        pkg_dir, 
        comm= None, 
        cluster_parallelization= False, 
        cluster_parallelize_metrics= False, 
        cluster_time_limit_str= None, 
        cluster_mem_free_str= None
    ):
        super().__init__()
        self.data['pop'] = []
        self.protein= protein
        self.metric_name_list= [str(metric) for metric in metrics_list]
        self.observer_metrics_list= observer_metrics_list
        self.pkg_dir= pkg_dir
        self.comm= comm
        self.cluster_parallelization= cluster_parallelization
        self.cluster_parallelize_metrics= cluster_parallelize_metrics
        self.cluster_time_limit_str= cluster_time_limit_str
        self.cluster_mem_free_str= cluster_mem_free_str

    def notify(self, algorithm):
        metrics= algorithm.pop.get('F')
        candidates= algorithm.pop.get('X')

        pop_df= pd.DataFrame(metrics, columns= self.metric_name_list)
        pop_df['candidate']= [''.join(candidate) for candidate in candidates]

        if self.observer_metrics_list is not None:
            observer_metrics_scores= evaluate_candidates(
                metrics_list= self.observer_metrics_list,
                candidates= candidates,
                protein= self.protein,
                pkg_dir= self.pkg_dir,
                comm= self.comm,
                cluster_parallelization= self.cluster_parallelization,
                cluster_parallelize_metrics= self.cluster_parallelize_metrics,
                cluster_time_limit_str= self.cluster_time_limit_str,
                cluster_mem_free_str= self.cluster_mem_free_str
            )
            observer_metrics_scores_df= pd.DataFrame(
                observer_metrics_scores, 
                columns= [str(metric) for metric in self.observer_metrics_list]
            )
            pop_df= pd.concat([pop_df, observer_metrics_scores_df], axis= 1)

        self.data['pop'].append(pop_df)

        logger.debug(
            textwrap.dedent(
                f'''\
                SavePop returned the following population:
                {sep}\n{pop_df}\n{sep}
                '''
            )
        )

class DumpPop(Callback):
    '''
    A subclass of pymoo.core.callback.Callback. It functions similarly to the
    SavePop class, but instead of saving intermediate results to the final
    output of pymoo.optimize.minimize(), it dumps the intermediate results to
    a pickle file at the end of each iteration.
    '''
    def __init__(
        self, 
        protein, 
        metrics_list, 
        observer_metrics_list, 
        out_file_name, 
        pkg_dir, 
        comm= None, 
        cluster_parallelization= False, 
        cluster_parallelize_metrics= False, 
        cluster_time_limit_str= None, 
        cluster_mem_free_str= None
    ):
        super().__init__()
        self.protein= protein
        self.metric_name_list= [str(metric) for metric in metrics_list]
        self.observer_metrics_list= observer_metrics_list
        self.out_file_name= out_file_name
        self.iteration= 0
        self.pkg_dir= pkg_dir
        self.comm= comm
        self.cluster_parallelization= cluster_parallelization
        self.cluster_parallelize_metrics= cluster_parallelize_metrics
        self.cluster_time_limit_str= cluster_time_limit_str
        self.cluster_mem_free_str= cluster_mem_free_str

    def notify(self, algorithm):
        metrics= algorithm.pop.get('F')
        candidates= algorithm.pop.get('X')

        pop_df= pd.DataFrame(metrics, columns= self.metric_name_list)
        pop_df['candidate']= [''.join(candidate) for candidate in candidates]

        if self.observer_metrics_list is not None:    
            observer_metrics_scores= evaluate_candidates(
                metrics_list= self.observer_metrics_list,
                candidates= candidates,
                protein= self.protein,
                pkg_dir= self.pkg_dir,
                comm= self.comm,
                cluster_parallelization= self.cluster_parallelization,
                cluster_parallelize_metrics= self.cluster_parallelize_metrics,
                cluster_time_limit_str= self.cluster_time_limit_str,
                cluster_mem_free_str= self.cluster_mem_free_str
            )
            observer_metrics_scores_df= pd.DataFrame(
                observer_metrics_scores, 
                columns= [str(metric) for metric in self.observer_metrics_list]
            )
            pop_df= pd.concat([pop_df, observer_metrics_scores_df], axis= 1)

        if (self.comm is None) or (self.comm.Get_rank() == 0):
            pickle.dump(
                pop_df, 
                open(f'{self.out_file_name}_{self.iteration}.p', 'wb')
            )
            self.iteration+= 1

            logger.debug(
                textwrap.dedent(
                    f'''\
                    SavePop dumped to file the following population:
                    {sep}\n{pop_df}\n{sep}
                    '''
                )
            )

class LoadPop(Sampling):
    '''
    A subclass of pymoo.core.sampling.Sampling, useful for initializing a
    genetic algorithm population, by reading in an existing population.

    See also ga_operator.ProteinSampling.
    '''
    def __init__(self, pickle_file_loc):
        '''
        Input
        -----
        pickle_file_loc (str): path to a pickle file containing the initial population.
        '''
        super().__init__()

        self.pickle_file_loc= pickle_file_loc

    def _do(self, problem, n_samples, **kwargs):
        '''
        Read in the content of 'self.pickle_file_loc'. The pickle file should 
        contain either a pd.DataFrame or a list thereof, and the DataFrame should 
        have a 'candidate' column; if a list of DataFrames is provided, only the 
        last element of the list will be read.

        Output
        -----
        proposed_candidates (list[list[str]]): a list of candidates.
        '''
        existing_pop= pickle.load(open(self.pickle_file_loc, 'rb'))
        if isinstance(existing_pop, list):
            # if the pickle file is a list, assume that the last element is a df 
            # containing the final population
            proposed_candidates= np.array(
                [list(candidate) for candidate in existing_pop[-1]['candidate']]
            )
        elif isinstance(existing_pop, pd.core.frame.DataFrame):
            proposed_candidates= np.array(
                [list(candidate) for candidate in existing_pop['candidate']]
            )
        else:
            raise ValueError(
                f'Unrecognized data format detected from {self.pickle_file_loc}'
            )
        
        logger.debug(
            textwrap.dedent(
                f'''\
                LoadPop read in the following candidates:
                {sep}\n{proposed_candidates}\n{sep}
                '''
            )
        )
        return proposed_candidates
