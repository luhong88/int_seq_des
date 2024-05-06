import subprocess
import numpy as np

cam_pdb_files= [
    '1cfd_cleaned_6871', '1cff_cleaned_apo_frame3_reindexed_truncated_14657',
    '1ckk_cleaned_apo_frame0_reindexed_14373', '1cll_cleaned_apo_9604',
    '1cm1_cleaned_apo_6095', '1g4y_cleaned_apo_renamed_truncated_reindexed_asym_11436',
    '1niw_cleaned_apo_chainAB_MET_12352', '1nwd_apo_truncated_reindexed_5348',
    '2f2p_cleaned_apo_reindexed_asym_9369', '2n8j_frame0_reindexed_2275',
    '2wel_cleaned_apo_truncated_reindexed_9656', '3ewt_cleaned_apo_2202',
    '3ewv_cleaned_apo_reindexed_6808', '4djc_cleaned_apo_reindexed_truncated_2279'
]

tmscore_exec= '<path_to_tmscore_compiled_binary>'

n_pdb= len(cam_pdb_files)
rms_arr= np.zeros((n_pdb, n_pdb))
tms_arr= np.zeros((n_pdb, n_pdb))
gdt_arr= np.zeros((n_pdb, n_pdb))

for idx1, pdb1 in enumerate(cam_pdb_files):
    for idx2, pdb2 in enumerate(cam_pdb_files):
        proc = subprocess.run(
            [tmscore_exec, f'{pdb1}.pdb', f'{pdb2}.pdb'],
            stdout= subprocess.PIPE, 
            stderr= subprocess.PIPE, 
            check= True
        )
        
        output = proc.stdout.decode()
        
        parse_float = lambda x: float(x.split("=")[1].split()[0])
        o = {}
        for line in output.splitlines():
            line = line.rstrip()
            if line.startswith("RMSD"):
                rms_arr[idx1, idx2]= parse_float(line)
            if line.startswith("TM-score"):
                tms_arr[idx1, idx2]= parse_float(line)
            if line.startswith("GDT-TS-score"):
                gdt_arr[idx1, idx2]= parse_float(line)

#np.savetxt('pairwise_rms.txt', rms_arr)
np.savetxt('pairwise_tms.txt', tms_arr)
#np.savetxt('pairwise_gdt.txt', gdt_arr)
