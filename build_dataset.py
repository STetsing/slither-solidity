import numpy as np
import pandas as pd
from slither_sol_helpers import *
import os, json
import time
import swifter
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True, nb_workers=os.cpu_count())



partial_dataset_path = '/home/pippertetsing/sourcify_contract_data/partial_match'
full_dataset_path = '/home/pippertetsing/sourcify_contract_data/full_match'
contracts_dirs_saved = '/home/pippertetsing/sourcify_contract_data/contracts_dirs.pkl'

if os.path.exists(contracts_dirs_saved):
    contracts_dirs = pd.read_pickle(contracts_dirs_saved)
else:
    contracts_dirs_partial = check_folder_structure(partial_dataset_path)
    contracts_dirs_full = check_folder_structure(full_dataset_path)
    contracts_dirs = pd.concat([contracts_dirs_full, contracts_dirs_partial])

contracts_dirs = contracts_dirs[contracts_dirs.has_src_files == True]

def slither_process(df_row):
    result = []
    src_dir = df_row.contracts_dir + '/sources'
    args = construct_mapping_and_args(df_row.contracts_dir + "/metadata.json", True, True)

    if get_all_sol_files(src_dir) is None:
        print(f'no solidity file in {src_dir}')
    else:
        for sol_file_cp in get_all_sol_files(src_dir):
            sol_file = sol_file_cp.replace(src_dir, '.')
            cmd = ['slither', sol_file.replace(src_dir, '.')]
            _ = [cmd.append(x) for x in args]
            p = subprocess.run(cmd,
                cwd=src_dir,
                shell=False,                            
                capture_output = True,
                universal_newlines=True)
            
            if p.stdout == '':
                result.append({'source_dir':src_dir, 'sol_file':sol_file_cp,
                                'contracts_dirs':df_row.contracts_dir,
                                'has_src_files': df_row.has_src_files,
                               'slither_processed':False,
                                'slither':None})
                #print(sol_file, 'process status:', False)
            else:
                output = json.loads(p.stdout)
                #print(sol_file, 'process status:', output['success'])
                result.append({'source_dir':src_dir, 'sol_file':sol_file_cp,
                               'slither_processed':output['success'],
                               'contracts_dirs':df_row.contracts_dir,
                               'has_src_files': df_row.has_src_files,
                               'slither':get_slither_check_from_json(output)})
    return pd.DataFrame(result)

contracts_dirs['slither_res'] = contracts_dirs.parallel_apply(slither_process, axis=1) 


dataset = pd.DataFrame()
for _, row in contracts_dirs.iterrows():
    dataset = pd.concat([dataset, row.slither_res])

dataset.to_pickle('./slither_processed_contracts.pkl')
