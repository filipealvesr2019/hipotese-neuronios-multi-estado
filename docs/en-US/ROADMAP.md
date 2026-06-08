# Research Roadmap — Sparse Routing & Mixture of Experts (V4+)

This document outlines the logical next steps to validate, scale, and apply the sparse routing architecture discovered in Phase 1 (V1-V4).

## Level 1 — Scientific Validation (Confirm it's not a coincidence)
The goal here is to test the architecture on classic computer vision datasets to ensure it scales to real dimensionalities and learns structured representations.

- **E1: MNIST (28x28, 10 classes)**
  Compare Traditional MLP vs V4 Sparse focusing on: Accuracy, FLOPs, Training time, and Inference time. Goal: Same accuracy, fewer FLOPs.
  
- **E2: Fashion-MNIST**
  Test if experts learn distinct semantic categories (e.g., one expert for shirts, another for shoes) without explicit programming.

- **E3: CIFAR-10**
  Strong generalization test on complex RGB images (planes, cats, dogs).

## Level 2 — Test the Real Hypothesis (Computational Efficiency)
Definitively prove the "intelligence per operation" gain.

- **E4: Accuracy vs FLOPs Curve**
  Train dense models of varying widths (64, 128, 256 neurons) and V4 models with varying expert counts (4, 8, 16). The goal is to plot the graph and prove that V4 stays above the traditional curve (More accuracy per FLOP).
  
- **E8: Introspective Specialization Analysis**
  Record *who is used for what*. Discover if E1 focuses on edges, E2 on curves, E3 on noise, etc.

## Level 3 — Approaching LLMs (Transformer Applications)
Replace the *FeedForward (FFN)* layer of a Transformer with the *V4 Sparse Experts* layer.

- **E5: Tiny Transformer (1M to 10M parameters)**
  Evaluate if perplexity is maintained at a reduced cost.
  
- **E6: TinyStories**
  Train a Dense Transformer vs V4 Transformer on the TinyStories dataset. Evaluate Loss, Perplexity, and Tokens/s.
  
- **E7: GPT Mini (e.g., 10M local params)**
  Replace only the FFN blocks and measure VRAM consumption, inference time, and generative quality.
  
- **"Dream" Experiment (The Final Target):**
  Take a small foundational model (GPT-2 Small, Phi, TinyLlama), replace dense layers with V4 experts, and maintain the same text quality while spending a fraction of the FLOPs.
