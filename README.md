# 🚀 LLM as GNN: Graph Vocabulary Learning for Text-attributed Graph Foundation Model



## 🔎 Overview

This repository contains the code implementation for the paper **LLM as GNN: Graph Vocabulary Learning for Text-attributed Graph Foundation Model**.

**Abstract**

Graphs typically exhibit distinctive structure and domain-specific knowledge, motivating the development of a Graph Foundation Model (GFM) capable of generalizing across various graphs and tasks. While recent efforts have focused on combining the strengths of Large Language Models (LLMs) and Graph Neural Networks (GNNs), they often struggle to maximize mutual benefit due to the decoupled architectures. Moreover, existing methods assign out-of-vocabulary (OOV) tokens to nodes, which are incompatible with the natural language vocabulary for task-oriented prompt generation, hindering knowledge transfer in GFM. In this paper, we introduce PromptGFM, a versatile GFM grounded in graph vocabulary learning, comprising two key components: (1) Graph Understanding Module, which explicitly replicates the finest GNN workflow in the language space using LLMs, enabling seamless GNN-LLM integration and elegant graph-text alignment; (2) Graph Inference Module, where we establish a novel language-based graph vocabulary to ensure expressiveness, transferability, and scalability. This vocabulary enables the generation of readable instructions for LLM inference, resolving modality incompatibility and facilitating positive transfer. Extensive experiments demonstrate the superiority of PromptGFM in node classification and link prediction, along with its strong transferability across different datasets and tasks. 


## Environment💾
**Set up an environment**

Navigate to the directory containing the `environment.yaml` file in your terminal, then run the following command to create the environment based on the YAML file:

```bash
conda env create -f environment.yaml
```

  
## Run 🧑‍💻
### Step 1: Graph Understanding Module

```bash
python ./generator/generate_textual_id_V3.py --dataset citeseer
```

### Step 2: Readable Instruction Construction in Graph Inference Module

#### For link Prediction:

```bash
python ./data_prepocess/LP_prepocess.py --dataset citeseer
```

#### For Node classification:

```bash
python ./data_prepocess/NC_prepocess.py --dataset citeseer
```

### Step 3: Multi-Prompt Instruction Fine-tuning in Graph Inference Module

#### Train the Model

```bash
python train.py --task link_prediction --dataset citeseer
```


 
