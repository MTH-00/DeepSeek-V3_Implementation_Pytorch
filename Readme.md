This repository provides a PyTorch implementation and testing scripts for components of the DeepSeek-V3 model, focusing on Multi-Head Latent Attention (MLA) and Rotational Position Embeddings (RoPE).

##  Repository Structure

- **MLA/** *(if present)*: Containing code relating to Multi-Head Latent Attention.
- **ROPE/** *(if present)*: RoPE implementation module.
- **model.py**: Defines DeepSeek-V3 model architecture in PyTorch.
- **requirements.txt**: Python dependencies required to run the code.
- **test_MLA.py**: Test script for verifying MLA implementation.
- **test_ROPE.py**: Test script for validating RoPE correctness.

##  Purpose

This repository offers an open-source reimplementation of DeepSeek-V3’s key architectural components—particularly MLA and RoPE—within PyTorch, facilitating experimentation and educational use.

##  Getting Started

1. Clone this repository:
   ```bash
   git clone https://github.com/Shameerisb/DeepSeek-V3_Implementation_Pytorch
   cd DeepSeek-V3_Implementation_Pytorch
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
3. Explore and understand the model's structure by reviewing:
    ```bash
    model.py
4. Run test scripts to validate implementations:
    ```bash
    python test_MLA.py
    python test_ROPE.py

## Highlights

- **model.py** – Core PyTorch adaptation of DeepSeek-V3 architecture.  
- **test_MLA.py / test_ROPE.py** – Validate critical components (MLA and RoPE).  

---

## Suggested Workflow

1. Review the repository’s structure and core design in `model.py`.  
2. Run tests to confirm the implementations are working correctly.  
3. Study and modify—useful for those exploring the architectural innovations of DeepSeek-V3.  
4. Integrate or extend the components into larger projects or benchmarks.  

---

## Background (DeepSeek-V3 Context)

DeepSeek-V3 is a **Mixture-of-Experts (MoE) language model** with ~**671 billion parameters**, of which ~**37 billion** are activated per token.  

It introduces several innovations:  
- **Multi-Head Latent Attention (MLA)**  
- **DeepSeekMoE** with auxiliary-loss-free load balancing  
- **Multi-token prediction** training objective  
- **Enhanced inference efficiency**  

Trained on ~**14.8 trillion tokens**, DeepSeek-V3 achieves **state-of-the-art performance** on reasoning, math, and coding benchmarks, while maintaining **cost-efficient training**.  

---

