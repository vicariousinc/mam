# Introduction

This repo contains code for reproducing the results in the paper [**Graphical Models with Attention for Context-Specific Independence and an Application to Perceptual Grouping**](https://arxiv.org/abs/2112.03371).

# Setting up the environment

1. The code was tested under python 3.7
2. Set up the virtual environment
```
pip install -r requirements.txt
python setup.py develop
```
3. Point `BASE` in `mam/__init__.py` to the data directory (full sets of data coming)

The code was tested on Ubuntu 18.04 with CUDA 11.3.

# Reproducing the results

Use the scripts in the `scripts` folder to reproduce results in the paper.
