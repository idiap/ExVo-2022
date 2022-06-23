<!-- SPDX-FileCopyrightText: Copyright © 2022 Idiap Research Institute <contact@idiap.ch>

SPDX-FileContributor: Bogdan Vlasenko <bogdan.vlasenko@idiap.ch>

SPDX-License-Identifier: GPL-3.0-only -->

# Extracting pre-trained self-supervised embeddings for ICML ExVO 2022 challenge

This python code generates Generation SSL (WavLM) feature representation:

- Estimate frame-level SSL embeddings from pre-trained models
- Generates fixed-length feature representation by using mean and standard deviation functionals

The extracted feature representations were used during ICML ExVo 2022  challenge, for more details see:

T. Purohit, I. B. Mahmoud, B. Vlasenko, M. Magimai.-Doss. Comparing supervised and self-supervised embedding for ExVo Multi-Task
learning track, workshop ICML ExVo 2022

## Install

Create a Conda environment with python 3.10 and activate it:

```bash
conda create -n env python=3.10
conda activate env
```
Install all required python packages

```bash
pip install -r requirements.txt
```
