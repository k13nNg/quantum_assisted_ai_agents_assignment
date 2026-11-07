# Quantum-Assisted Resource Allocation for Multi-Agent AI Systems

Can quantum optimization efficiently solve resource-constrained task allocation in
multi-agent AI systems? This repository explores that question by formulating the
problem of assigning tasks to heterogeneous LLM agents under token-budget constraints
as a **Generalized Assignment Problem (GAP)** and solving it with variational quantum
algorithms.

The core contribution is an application of the **VQGAP** framework and its
logarithmically-encoded extension **VQGAPe** (Mastroianni et al., 2025) to agent
orchestration, which shrinks the quantum register from `NM` qubits down to
`N⌈log₂ M⌉` qubits while keeping capacity-constraint evaluation on the classical side
of the hybrid loop. This makes the approach viable on near-term (NISQ) hardware.

## Motivation

Modern AI pipelines increasingly rely on multi-agent systems, where specialized
agents, each with distinct capabilities and computational costs, collaborate on
complex workflows. A central operational question is **how to assign tasks to agents**
so that overall completion quality is maximized while every agent stays within its
token budget.

## Problem formulation

Given `N` tasks and `M` agents, let `x_nm ∈ {0,1}` indicate that task `n` is assigned
to agent `m`. Each agent `m` has a token budget `b_m`; assigning task `n` to agent `m`
consumes `a_nm` tokens and yields an expected success rate `c_nm ∈ [0,1]`.

Two competing objectives are combined with a trade-off parameter `λ ∈ [0,1]`:

- **Maximize completion quality** — the sum of success rates of chosen assignments.
- **Minimize normalized token consumption** — using `ā_nm = a_nm / a_max` so quality
  and cost live on the same scale.

The merged objective uses a combined weight `w_nm = λ·ā_nm − (1−λ)·c_nm`, subject to:

- each task assigned to exactly one agent (`Σ_m x_nm = 1`), and
- each agent's token budget respected (`Σ_n a_nm·x_nm ≤ b_m`).

See `report/main.tex` (compiled to `report/main.pdf`) for the full derivation.

## VQE → VQGAP → VQGAPe

The bottleneck in applying a standard Variational Quantum Eigensolver (VQE) to GAP is
qubit count. Each of the three formulations encodes the register differently:

| Approach | Qubit register | Capacity check | Qubits (4×4 example) |
|----------|----------------|----------------|----------------------|
| Standard VQE / QUBO | `NM + Σ_m ⌈log₂(b_m+1)⌉` (assignment + binary-expanded slack) | on-circuit | 72 |
| VQGAP | `NM` (one-hot assignment only) | classical post-processing | 16 |
| VQGAPe | `N⌈log₂ M⌉` (logarithmic assignment index) | classical post-processing | 8 |

- **VQGAP** eliminates the slack qubits by moving the capacity check off the register.
  Residual budgets `s_m = max(0, b_m − Σ_n a_nm·x_nm)` are computed classically, so
  feasible assignments incur zero capacity penalty and infeasible ones a quadratic one.
- **VQGAPe** goes further: instead of one qubit per task-agent pair, it binary-encodes
  the *index* of the assigned agent, using `L = ⌈log₂ M⌉` qubits per task. The
  one-task-one-agent constraint is then satisfied by construction, and the register
  scales logarithmically in the number of agents.

In the 4-task, 4-agent example in this repo, this is an **88.9% qubit reduction**
versus standard VQE (72 → 8 qubits).

## What's in this repo

```
.
├── VQGAPe_Explained_Simplified.ipynb   # Reproducible implementation + figures
├── Quantum-assisted AI Agents Orchestration.pdf   # Presentation slides
└── report/
    ├── main.tex                        # Full technical report (LaTeX source)
    ├── main.pdf                        # Compiled report
    └── photos/                         # VQE vs. VQGAP architecture diagrams
```

## The notebook

`VQGAPe_Explained_Simplified.ipynb` implements the VQGAPe formulation end-to-end on a
worked example: four cross-disciplinary tasks and four specialized LLM agents
(Systems/DB, AI/ML, Security/Infra, Full-Stack) with realistic token costs, success
rates, and per-agent budgets. It walks through:

1. **Example instance** — the `a_nm`, `c_nm`, `b_m` data and combined weights `w_nm`.
2. **Exact reference** — enumerates all `2⁸ = 256` encoded states to find the exact
   feasible optimum, and shows how a naive greedy baseline overflows an agent's budget.
3. **VQGAPe-ESU2 ansatz** — a hardware-efficient parameterized circuit (`R_y`/`R_z`
   rotations with a linear CNOT cascade): 8 qubits, 32 trainable parameters.
4. **Hybrid optimization** — a statevector simulation with `K = 4096` shots per
   evaluation and COBYLA updating the circuit angles.
5. **Decoding and audit** — recovers the agent index per task, computes residual
   capacities classically, and selects the minimum-cost feasible bitstring.

### Result

The variational optimizer recovers the exact feasible global optimum:
`F* = −1.4866`, encoded state `|00111001⟩`, matching the brute-force reference while
the greedy baseline violates Agent 2's budget by 1000 tokens.

## Running the notebook

The notebook was developed on Google Colab (Python 3.13, Qiskit 2.x). To run locally:

```bash
pip install qiskit numpy scipy matplotlib
jupyter notebook VQGAPe_Explained_Simplified.ipynb
```

A fixed random seed makes the simulated shot sequence reproducible.

## Reference

C. Mastroianni, F. Plastina, J. Settino, and A. Vinci,
"Efficient Variational Quantum Algorithms for the Generalized Assignment Problem,"
arXiv preprint [arXiv:2511.02739](https://arxiv.org/abs/2511.02739), 2025.

## Author

Kien Nguyen, University of Waterloo
