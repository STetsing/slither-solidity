# This file aims to provide helper functions for analysing vunerabilites in solidity files
import os, json
import subprocess
from tqdm import tqdm
import pandas as pd
import glob

detectors_json = './detectors.json'
with open(detectors_json, 'r') as f:
    detectors = json.load(f)

def check_folder_structure(dataset_path):
    correct_folders = 0
    total_folders = 0 
    df = []
    for chainID in tqdm(os.listdir(dataset_path)):
        if not os.path.isdir(os.path.join(dataset_path, chainID)):
            continue
        
        for ctc in  tqdm(os.listdir(os.path.join(dataset_path, chainID)), desc='inner1'):
            if not os.path.isdir(os.path.join(dataset_path, chainID, ctc)):
                continue
            
            got_meta = False
            got_src = False
            total_folders += 1
            for obj in os.listdir(os.path.join(dataset_path, chainID, ctc)):
                if 'metadata.json' in obj:
                    got_meta = True
                if 'sources' in obj:
                    got_src = True

            if got_src and got_meta:
                correct_folders +=1 
                df.append({'contracts_dir':os.path.join(dataset_path, chainID, ctc), 'has_src_files':True})
            else:
                df.append({'contracts_dir':os.path.join(dataset_path, chainID, ctc), 'has_src_files':False})

    print(f'found {correct_folders} out of {total_folders} folders:  {correct_folders*100/total_folders}%.2f ')
    return pd.DataFrame(df)

def detect_sol_version(json_meta):
    try: 
        with open(json_meta, 'r') as f:
            data = json.load(f)
        sol_version = data['compiler']['version'].split('+')[0]

        return sol_version

    except Exception as ex:
        print('Error: Could not detect the solidity version')

def set_sol_version(version):
    try:
        p = subprocess.run(['solc-select', 'install', version],
                            stdout=subprocess.PIPE, 
                            universal_newlines=True)

        p = subprocess.run(['solc-select', 'use', version],
                            stdout=subprocess.PIPE, 
                            universal_newlines=True)
        # print('Info: env successfully set!')
        
    except Exception as ex: 
        print('Error: env not set', ex)

def slither_sol_file(file_name):
    try:
        p = subprocess.run(['slither', file_name],
                            capture_output = True,
                            universal_newlines=True)

        return _clean_slither_result(p.stderr)
    except Exception as ex: 
        print('Error: Did slither the sol file', ex)

def _clean_slither_result(result):
    start = 'INFO:Detectors:'
    end = 'INFO:Slither'
    return result[result.find(start)+len(start):result.rfind(end)]

def construct_mapping_and_args(json_file, exclude_low=False, exclude_med=False):
    with open(json_file, 'r') as f:
        data = json.load(f)
    args = []
    remapings = data['settings']['remappings']
    optimize = data['settings']['optimizer']['enabled']
    optimize_runs = data['settings']['optimizer']['runs']
    viaIR = data['settings']['viaIR'] if 'viaIR' in data['settings'].keys() else False
    
    if len(remapings):
        args.append('--solc-remaps')  
        maps = '\"' + ' '.join(remapings) + '\"'
        args.append(maps)  

    if any([optimize, viaIR]):
        args.append('--solc-args')  
        arg = '--optimize ' if optimize else ''
        arg += '--optimize-runs ' + str(optimize_runs)if optimize and optimize_runs else ''
        arg += ' --via-ir' if viaIR else '' 
        args.append(arg)

        args.append('--exclude-informational')
        args.append('--exclude-dependencies')
        args.append('--exclude-optimization')

    if exclude_low:
        args.append('--exclude-low')

    if exclude_med:
        args.append('--exclude-medium')

    args.append('--solc-solcs-select')
    args.append(str(detect_sol_version(json_file)))

    args.append('--json')
    args.append('-')

    return args


def get_all_sol_files(sources_dir):
    x = glob.glob(sources_dir + '/**/*.sol', recursive=True)
    return x

def get_slither_check_from_json(slither_result):
    if slither_result.get('results').get('detectors') is None:
        return None
    else: 
        return [detectors.get(d.get('check'))['idx'] for d in slither_result.get('results').get('detectors')]