# README.md

## HyperIDR

This repository provides the code for **HyperIDR: a multi-scale semantic hypernetwork for identification of intrinsically disordered regions**.

## Requirements

All experimental environments and dependent packages are listed in `./environment.yml`. You can quickly build the running environment with the following command:

```bash
conda env create -f environment.yml
```

## Datasets

The datasets used in this project can be downloaded from the official web server:

[http://bliulab.net/HyperIDR/](http://bliulab.net/HyperIDR/)

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
- `environment.yml`: Conda environment configuration



