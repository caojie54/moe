# MoA: Heterogeneous Mixture of Adapters for Parameter-Efficient Fine-Tuning of Large Language Models

<!-- [![arXiv](https://img.shields.io/badge/arXiv-xxxx.xxxxxx-b31b1b.svg)](https://arxiv.org/abs/xxxx.xxxxxx) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) **[Paper Link (if published)]** | **[Project Demo (if applicable)]** | **[Hugging Face Model (if applicable)]** -->

This is the official PyTorch implementation for the paper "**MoA: Heterogeneous Mixture of Adapters for Parameter-Efficient Fine-Tuning of Large Language Models**".

Our work introduces MoA (Mixture of Adapters), a novel approach that leverages a heterogeneous mixture of adapters to achieve superior parameter efficiency and performance when fine-tuning large language models.

## Table of Contents
- [Introduction](#introduction)
- [Key Features](#key-features)
- [Methodology Overview](#methodology-overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference](#inference)
- [Results](#results)
- [Pre-trained Models/Adapters](#pre-trained-modelsadapters)
- [To-Do (Optional)](#to-do-optional)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements (Optional)](#acknowledgements-optional)
- [Contact (Optional)](#contact-optional)

## Introduction

Fine-tuning Large Language Models (LLMs) typically requires significant computational resources and memory due to the large number of parameters involved. Parameter-Efficient Fine-Tuning (PEFT) methods aim to address this by updating only a small subset of the model's parameters. This repository provides the code for MoA, a novel PEFT technique that utilizes a *heterogeneous mixture of adapters* to enhance model performance and adaptability while maintaining high parameter efficiency.

## Key Features

* **Heterogeneous Adapter Mixture (MoA)**: Implementation of the core MoA mechanism.
* **Parameter-Efficient**: Significantly reduces the number of trainable parameters compared to full fine-tuning.
* **Modular Design**: Easy to integrate with popular LLM libraries (e.g., Hugging Face Transformers).
* **Reproducibility**: Scripts and guidelines to reproduce the results presented in our paper.
* **Extensible**: Designed to be easily extended with new adapter types or model architectures.

## Methodology Overview

MoA distinguishes itself by employing a diverse set of adapter modules, each potentially specializing in different aspects of the task or data. These adapters are then combined, often through a gating mechanism or a fixed combination strategy, to produce the final output. This heterogeneity allows for a richer representation and adaptation capability compared to using a single type of adapter.

## Installation
**Install dependencies:**

    ```
    pip install -r requirements.txt
    ```

## Usage

*(Provide clear instructions on how to run your code. Include examples for training, evaluation, and inference.)*


### Data Preparation

*(Describe the expected data format and any preprocessing steps required.)*
```bash
# Example: python scripts/preprocess_data.py --input_dir path/to/raw_data --output_dir path/to/processed_data