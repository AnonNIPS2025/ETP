# Towards Universal & Efficient Model Compression via Exponential Torque Pruning (ETP)

This repository contains the official PyTorch implementation of **Exponential Torque Pruning (ETP)**, as presented in our NeurIPS 2025 submission.

ETP introduces a novel structured pruning method that applies exponentially increasing regularization forces on neural modules based on their distance from a pivot point. The method achieves **state-of-the-art compression with minimal accuracy drop** across diverse domains including vision, NLP, graphs, and time series.

---

## ✨ Highlights

- 📈 **State-of-the-art speed-up vs. accuracy trade-off** on VGG-19, BERT, GAT, and Informer models
- ⚙️ **Simple, architecture-agnostic regularization** using exponential distance-weighted penalty
- 💡 **Generalizable to modern vision-language models (e.g., BLIP)** with strong empirical results
- 🧩 Plug-and-play for structured sparsity during training

---

## 📊 Results Summary

| Task                | Model      | Speed-up | Acc Drop |
|---------------------|------------|----------|----------|
| CIFAR-100 (Vision)  | VGG-19     | **9×**   | **-2.2%** |
| SST-2 (NLU)         | BERT       | 11×      | -1.4%     |
| PPI (Graph)         | GAT        | 12.16×   | -0.027 F1 |
| ETTh1 (Time-Series) | Informer   | 25×      | +6.15 MAE |
| Flickr8k (VLM)      | BLIP       | 20% pruned | BLEU-4 drop: **-1.07** |

---

## Method: Exponential Torque Pruning (ETP)

The core idea of **Exponential Torque Pruning (ETP)** is to **apply stronger regularization forces to neural modules that are farther from a selected pivot point** during training — promoting structured sparsity in a simple yet highly effective way.
<p align="center">
  <img src="https://github.com/user-attachments/assets/094e3c0f-0e51-4f70-9ba6-586f39e7d8fd" width="600"/>
</p>

### ⚙️ Motivation

Traditional Torque-based pruning applies a linear penalty based on module distance from a pivot, but:

- Distant modules may retain high L2 norms (i.e., they aren't effectively pruned)
- Close modules may be over-penalized, even though they are crucial for accurate predictions

### 💡 ETP Design

ETP replaces the **linear distance-based force** with an **exponential decay function**:

\[
\mathcal{L}_{\text{ETP}} = \sum_{l} \sum_{i} \|w_i^l\|^2 \cdot \lambda^{d_i}
\]

Where:
- \( w_i^l \): Weights of the \(i^{th}\) module in layer \(l\)
- \( d_i = \| \rho^l_i - \rho^l_p \| \): Distance to the pivot module
- \( \lambda = \exp\left(\frac{5}{|G_l|}\right) \): Exponential base based on the number of grouped units (e.g., filters, heads) in the layer
- \( \beta \): Regularization strength (tuned per model)



