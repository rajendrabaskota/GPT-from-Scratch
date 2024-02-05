# **Implementation of a Generative Pretrained Transformer from Scratch**

## Key Features
* Multi-GPU training using **DistributedDataParallel** from PyTorch
* Implementation of **Gradient Accumulation**
* Implementation of **Byte-Pair Encoding (BPE)** tokenization
* **Learning rate scheduling**

## Hyperparameters Used
* Embedding Dimension: 512
* Number of Layers: 8
* Number of Self-attention Heads: 8
* Batch Size: 32
* Gradient Accumulation Iterations: 8
* Learning Rate: 1.2e-3
* Learning Rate Scheduling Rate: 0.1
* Iterations: 20000

## Training and Results
* Model trained on Machine Translation Data Set of parliament hearings (https://www.kaggle.com/datasets/aadishjoshi/machine-translation-data-set)
* Minimum obtained test loss with above hyperparameters: 3.428
