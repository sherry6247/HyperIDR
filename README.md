<!--
 * @Date: 2026-09-28 17:14:14
 * @LastEditors: Sicen Liu
 * @LastEditTime: 2026-09-28 22:25:55
 * @FilePath: /liusicen/mygithub/HyperIDR/README.md
 * @Description:  
 * @Copyright: © 2025 liusicen@smbu.edu.cn. All rights reserved.
-->
# README.md

## HyperIDR

This repository provides the code for **HyperIDR: a multi-scale semantic hypernetwork for identification of intrinsically disordered regions**.

## Requirements

All experimental environments and dependent packages are listed in `./environment.yml`. You can quickly build the running environment with the following command:

```bash
conda env create -f environment.yml
```

## Datasets

### Training sets (`./datasets`)

| File | Description | # Proteins |
| --- | --- | --- |
| `DM4845-globalNR.fasta` | Unified redundancy-reduced training set: original training proteins sharing >25% sequence identity (BlastClust, `-S 25 -L 0.9 -b T`) with ANY of the six test datasets were removed | 4,146 |
| `Train_723_bc25.fasta` | Training set with redundancy to DISORDER723 removed (BlastClust 25%) | 4,273 |
| `Train_494_bc25.fasta` | Training set with redundancy to MXD494 removed (BlastClust 25%) | 4,556 |
| `Train_329_bc25.fasta` | Training set with redundancy to SL329 removed (BlastClust 25%) | 4,807 |
| `Train_casp_bc25.fasta` | Training set with redundancy to CASP removed (BlastClust 25%) | 4,816 |
| `Train_update_caid3_pdb_bc25.fasta` | Training set with redundancy to CAID3 Disorder-PDB removed (BlastClust 25%) | 4,790 |
| `Train_nox_bc25.fasta` | Training set with redundancy to CAID3 Disorder-NOX removed (BlastClust 25%) | 4,838 |

All training fasta files use the 3-line-per-record format: `>id`, amino-acid sequence, per-residue binary label.

### Test sets

The six independent test datasets (MXD494, SL329, DISORDER723, CASP, CAID3 Disorder-PDB, CAID3 Disorder-NOX) can be downloaded from the official web server:

[http://bliulab.net/HyperIDR/](http://bliulab.net/HyperIDR/)


## Prediction results (`./predicted_results`)

Each test-set folder contains two files in the same format:

- `<Dataset>_model_results.txt` — predictions of the benchmark-specific model
- `model_results.txt` — predictions of the unified model (`saved_models/model.pth`)

File format: one block per protein (`>id:` header) with per-residue rows `amino-acid<TAB>true_label<TAB>prediction_probability`.

## Usage

### Training

The entry file for model training:

```bash
python main.py
```

### Prediction

The entry file for model testing and prediction:

```bash
python predict_main.py
```

## Project Structure

- `main.py`: Training pipeline
- `predict_main.py`: Testing and inference pipeline
- `src/`: Model implementation, trainer, evaluation and metrics
- `utils/`: Argument configuration, data processing and feature encoders
- `datasets/`: Training datasets (see above)
<!-- - `saved_models/`: Pre-trained model parameters  -->
- `predicted_results/`: Per-residue predictions on the six test sets
- `environment.yml`: Conda environment configuration
