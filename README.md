# TLoRA: Tri-Matrix Low-Rank Adaptation for Large Language Models

This repository implements **TLoRA**, a novel approach for parameter-efficient fine-tuning of large language models using a tri-matrix low-rank adaptation strategy. Our paper details the method and experimental results, and this code repository is linked in the paper.

Paper: https://arxiv.org/abs/2504.18735 

## Overview

TLoRA aims to significantly reduce the number of trainable parameters while retaining high model performance. The repository includes:
- **Data utilities:** Data loading and preprocessing ([data.py](data.py))
- **Model adaptation:** Core implementation of TLoRA ([tlora.py](tlora.py))
- **Experiment scripts:** Running and logging experiments ([experiments.py](experiments.py))
- **Visualization tools:** Plotting training dynamics and model analysis ([plot.py](plot.py), [figures.ipynb](figures.ipynb))
- **Logs and models:** Experiment logs in [logs/](logs/) and checkpoints in [models/](models/)

## Installation

A Python 3.8+ environment is required. We recommend using a virtual environment:

```sh
python -m venv venv
source venv/bin/activate  # On Linux/MacOS
venv\Scripts\activate     # On Windows
```

Install all required packages with:

```sh
pip install -r requirements.txt
```

*If a requirements file is unavailable, please refer to the installation section in the paper.*

## Usage

### Running Experiments

Use the [experiments.py](experiments.py) script to launch experiments. Output logs will be saved in the [logs/](logs/) directory.

### Visualizing Results

Open the [figures.ipynb](figures.ipynb) notebook to view plots of training curves, eigenvalue distributions, and other diagnostics:

```sh
jupyter notebook figures.ipynb
```

The notebook leverages functions from [plot.py](plot.py) to generate detailed visualizations that complement the results reported in the paper.

## Code Structure

- **data.py:** Data loading and preprocessing.
- **tlora.py:** Core implementation of the TLoRA adaptation method.
- **experiments.py:** Scripts for executing experiments and logging results.
- **plot.py:** Plotting and visualization utilities.
- **figures.ipynb:** Interactive notebook for analysis and result visualization.
- **logs/:** Directory containing experiment logs.
- **models/:** Pretrained model checkpoints and adapted model weights.
- **paper/:** Paper, supplementary materials, and related figures.

## Results

Our experiments demonstrate that TLoRA achieves competitive performance with a small fraction of trainable parameters compared to conventional fine-tuning methods. Detailed layer-wise analysis, weight distributions, and training curves are available in the [figures.ipynb](figures.ipynb) notebook and the logged outputs in [logs/](logs/).

## Citation

If you use this work in your research, please cite our paper:

```
@inproceedings{tlorapaper,
  title={TLoRA: Tri-Matrix Low-Rank Adaptation for Large Language Models},
  author={Tanvir Islam},
  booktitle={https://arxiv.org/abs/2504.18735},
  year={2025},
}
```


GitHub Repository: [https://github.com/itanvir/tlora](https://github.com/itanvir/tlora)
