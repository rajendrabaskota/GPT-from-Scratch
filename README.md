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

![test-loss](https://github.com/rajendrabaskota/GPT-from-Scratch/assets/66084649/d6474571-546d-470b-be52-b266c35e3a2f)
![train-loss](https://github.com/rajendrabaskota/GPT-from-Scratch/assets/66084649/75e0804d-ce62-4862-a873-7cfa5e43280a)
