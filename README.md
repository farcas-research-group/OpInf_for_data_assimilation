# Hamiltonian Preserving Operator Inference for Data Assimilation

## Contents

This repository contains code for the implementation of Hamiltonian preserving Operator Inference for use in the Korteweg-De Vries equations.

### Config

Stores default values for experimental setup and paths for storing the basis, operators, and snapshots needed for the procedure. 

### Utils

Utility functions for generating snapshots, basis, and operators for use across various source files

### Data

Default basis, operators, and snapshots.

Snapshots:

 - X: Training snaphots
 - Xt: Finite difference matrix of training snapshot
 - gH: Snapshots of gradient of Hamiltonian
 - Xtest: Full order model (integrated over test horizon)

H_OPS:

 - c: Constant term from reduced Hamiltonian dynamics
 - C: Linear term from reduced Hamiltonian dynamics
 - T: Quadratic term from reduced Hamiltonian dynamics

L_OP:

 - L: Term corresponding to spatial derivative in reduced Hamiltonian dynamics

Basis:

 - UU: U matrix of SVD of training snapshot
 - SS: S matrix of SVD of training snapshot

### main.py

Runs entire process of Non-Canonical Hamiltonian Operator Inference given an initial condition. Will pull from existing snapshots, basis, and hamiltonian operators if available. 
User can specify number of POD modes by including an integer argument.

```bash
python3 main.py $optional_modes
```

### generate_operators.py

Recomputes all operators and stores to disk.

### POD_energy.py

Creates a graph of POD energy.

### forecast.py

Generates an animation of the Reduced model compared to the full-order solution.

### compare_errors.py

Generates a graph of errors over number of POD modes.

