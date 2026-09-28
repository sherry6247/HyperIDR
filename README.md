<!--
 * @Date: 2026-09-28 17:14:14
 * @LastEditors: Sicen Liu
 * @LastEditTime: 2026-09-28 17:17:30
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

## Pre-trained models (`./saved_models`)

All model files are bare `state_dict` files. The model is `ConditionalIDRModel` (see `src/model.py`), threshold 0.5.

| File | Training set | lr | hidden_size | top_k | pool_size |
| --- | --- | --- | --- | --- | --- |
| `model.pth` | DM4845-globalNR (4,146) — **unified model** | 1e-4 | 128 | 3 | 16 |
| `SL329_model.pth` | Train_329_bc25 (4,807) | 1e-4 | 128 | 1 | 16 |
| `CASP_model.pth` | Train_casp_bc25 (4,816) | 1e-4 | 128 | 1 | 16 |
| `MXD494_model.pth` | Train_494_bc25 (4,556) | 1e-3 | 256 | 3 | 16 |
| `DISORDER723_model.pth` | Train_723_bc25 (4,273) | 1e-4 | 256 | 1 | 16 |
| `Disorder-PDB_model.pth` | Train_update_caid3_pdb_bc25 (4,790) | 1e-4 | 256 | 3 | 16 |
| `Disorder-NOX_model.pth` | DM4229_training (4,229) | 5e-4 | 128 | 7 | 96 |

Common training settings: AdamW (weight_decay 0.01), BCE loss (masked), batch size 4, prompt length 10, λ_orth 0.001 (0.01 for the NOX model).

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
- `saved_models/`: Pre-trained model parameters 
- `predicted_results/`: Per-residue predictions on the six test sets
- `environment.yml`: Conda environment configuration
