# Semantic-Staleness-in-AFL

# Semantic Staleness-Aware Asynchronous Federated Learning

This repository contains the implementation of a semantic staleness-aware asynchronous federated learning framework for handling stale client updates under heterogeneous non-IID environments.

## Overview

Federated Learning (FL) enables collaborative model training across multiple distributed clients without sharing raw local data. Conventional synchronous methods such as FedAvg require all selected clients to complete local training before global aggregation, which leads to inefficiencies in realistic environments with heterogeneous devices and communication delays.

Asynchronous Federated Learning (AFL) addresses this limitation by allowing clients to send updates independently without synchronization barriers. However, asynchronous aggregation introduces the stale update problem, where delayed client updates are computed using outdated global models and may negatively affect convergence.

This implementation proposes a Semantic Staleness-Aware AFL strategy that extends conventional temporal staleness handling by incorporating representation-level semantic relevance. Instead of relying only on communication delay, the server estimates semantic drift between local and global class prototypes to determine the usefulness of delayed updates.

---

## Experimental Configuration

The experiments are conducted using the following setup:

### Dataset
- **CIFAR-10**
- 10 image classes
- Training samples: 50,000
- Test samples: 10,000
- Input size: 32 × 32 RGB

### Data Distribution
Non-IID client partitioning is simulated using Dirichlet sampling:

- **Dirichlet alpha = 0.1**

This creates severe statistical heterogeneity, where each client receives a highly skewed subset of classes.

### Model
A lightweight custom CNN is used:

- Conv(3 → 32), ReLU, MaxPool
- Conv(32 → 64), ReLU, MaxPool
- Fully connected feature layer (128-dimensional)
- Final classification layer (10 classes)

This compact architecture is chosen to provide fast experimentation while preserving meaningful representation learning for semantic prototype comparison.

---

## Implemented Methods

The framework compares four federated learning strategies:

### 1. FedAvg
Standard synchronous federated averaging.

Characteristics:
- Global synchronization
- All clients participate per round
- No stale updates

---

### 2. FedAsync
Conventional asynchronous federated learning using temporal delay-based weighting:

\[
w = \frac{1}{1+t}
\]

where:
- \(t\) = temporal staleness

---

### 3. Buffered Async FL
Buffered asynchronous aggregation.

Characteristics:
- Delayed client updates stored in buffer
- Aggregation occurs after buffer threshold

---

### 4. Proposed Semantic Async FL
Semantic staleness-aware asynchronous aggregation.

Aggregation weight:

\[
r_i = \exp(-\alpha s_i^{time}) \cdot \exp(-\beta s_i^{sem})
\]

where:

- \(s_i^{time}\): temporal staleness
- \(s_i^{sem}\): semantic drift between local and global prototypes

This allows the server to assess stale update usefulness beyond delay information alone.

---

## Hyperparameters

| Parameter | Value |
|---------|------|
| Number of clients | 10 |
| Communication rounds | 50 |
| Local epochs | 1 |
| Batch size | 32 |
| Client learning rate | 0.01 |
| Server learning rate | 1.0 |
| Dirichlet alpha | 0.1 |
| Delay range | 1–3 |
| Buffer size | 3 |
| Time alpha | 0.1 |
| Semantic beta | 0.5 |

---

## Evaluation Metrics

The implementation reports:

- Accuracy
- Macro F1-score
- Balanced Accuracy
- Precision
- Recall
- Fairness Standard Deviation
- Jain Fairness Index

---

## Results Summary

Representative results under CIFAR-10 with strong non-IID partitioning:

| Method | Accuracy | Macro F1 | Balanced Accuracy |
|--------|----------|----------|------------------|
| FedAvg | 55.04% | 0.531 | 55.04% |
| FedAsync | 44.67% | 0.394 | 44.67% |
| Buffered | 40.93% | 0.347 | 40.93% |
| Semantic | 47.52% | 0.455 | 47.52% |

Observations:
- FedAvg achieves the highest performance due to synchronous aggregation.
- The proposed semantic asynchronous method outperforms conventional asynchronous baselines.
- Buffered aggregation performs worst due to additional lag.
- Semantic-aware weighting improves stale update handling in non-IID settings.

---

## Generated Outputs

### Logs
Saved in:

```bash
logs/
### Plots
plots/
