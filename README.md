# Sourcify Data Processing

The solidity contract data are preprocessed here for training a solidity code generator and completion. 

## Data Processing Approach
With the main dataset being ~43 GB large, this provide a solid base for training an LLM. However, to avoid vulnerabilities and further bias, the processing strategy involves first compiling each project or solidity files.
Run
```bash 
$ python build_dataset.py
```
for slithering all the sol files. The slithering process is applied on all single Solidity files in every contract directory. A project wide slither process is not supported yet

## Download Raw Data
You can follow the [instructions in the docs](https://docs.sourcify.dev/docs/repository/#s3-bucket) and contact [Kaan Uzdogan](mailto:kaan.uzdogan@ethereum.org) for the credentials.

## Slither 
Slither is used for detecting vulnerabilities in the solidity source code. See [slither](https://github.com/crytic/slither#api-documentation). Update the detectors' list with respect to up-to-date versions in the ```detectors.json``` file.

## Environment
Consider install all ```solc``` versions as the sources file might need different versions for compilation. The slithering process makes use of any possible solidity version. Latest by now 0.8.20.
```bash
$ pip install solc_select
$ solc-select install all 
```