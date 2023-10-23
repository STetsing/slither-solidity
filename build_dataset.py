import numpy as np
import pandas as pd
from slither_sol_helpers import *
import os, json
import time
from multiprocessing import  Pool, Manager
from functools import partial
from tqdm import tqdm
import traceback
#from pandarallel import pandarallel
#pandarallel.initialize(progress_bar=True, nb_workers=os.cpu_count())

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
pbar = tqdm(total=len(contracts_dirs)//os.cpu_count())
manager = Manager()
file_hashes = manager.dict()
duplicated_files = manager.list()

def slither_process(df_row):
    try:
        pbar.set_description(str(os.getgid()))
        pbar.update(1)

        result = []
        src_dir = df_row.contracts_dir + '/sources'
        args = construct_mapping_and_args(df_row.contracts_dir + "/metadata.json", True, True)

        if get_all_sol_files(src_dir) is None:
            print(f'no solidity file in {src_dir}')
        else:
            for sol_file_cp in get_all_sol_files(src_dir):
                sol_file = sol_file_cp.replace(src_dir, '.')

                if os.path.isdir(sol_file_cp):
                    print('wrong file. Pass a sol file not a dir!', sol_file_cp)
                    continue
    
                # skip sol file if content has already been processed
                c_sum = get_MD5_checksum(get_sol_data(sol_file_cp, False))
                if file_hashes.get(c_sum) is None:
                    file_hashes[c_sum] = sol_file_cp
                else:
                    print(f'File {sol_file_cp} has already been processed')
                    duplicated_files.append([sol_file, c_sum])
                    continue 

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
    except Exception as ex:
        print('Error while slithing sol:', ex)
        traceback.print_stack()
        return None


def parallelize(data, func, num_of_processes=8):
    data_split = np.array_split(data, num_of_processes)
    pool = Pool(num_of_processes)
    data_split = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data_split

def run_on_subset(func, data_subset):
    data_subset['slither_res'] = data_subset.apply(func, axis=1)
    print(f'Job {os.getpid()} finished')
    return data_subset


def parallelize_on_rows(data, func, num_of_processes=8):
    return parallelize(data, partial(run_on_subset, func), num_of_processes)


contracts_dirs = parallelize_on_rows(contracts_dirs, slither_process, num_of_processes=os.cpu_count())

dataset = pd.DataFrame()
for _, row in contracts_dirs.iterrows():
    dataset = pd.concat([dataset, row.slither_res])

dataset.to_pickle(f'./slither_processed_contracts.pkl')

with open('./files_hashes.json', 'w') as fp:
    json.dump(file_hashes.copy(), fp)

with open('./duplicated_files', 'w') as fp:
    for item in duplicated_files:
        # write each item on a new line
        fp.write(str(item[0]) + ', ' + str(item[1]) + "\n")

print('Processing Done!')