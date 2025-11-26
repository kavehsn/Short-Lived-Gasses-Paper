# Short-Lived-Gasses-Paper

This repository accompanies the research paper:

> **Short-lived Gases, Carbon Markets and Climate Risk Mitigation**  
> *Sara Biagini, Enrico Biffis & Kaveh Salehzadeh Nobari (2024)*  
> SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4997380

It contains all datasets, calibration routines, optimisation code, simulation results, and plotting scripts required to reproduce the full numerical analysis in the paper.

---

## What This Repository Provides

-  Calibrations of CO₂ and CH₄ impulse response functions using IPCC AR4–AR6 specifications  
-  Closed-form optimal emission pricing logic  
-  Grid-search optimisation for constrained & unconstrained solutions  
-  SGD-based optimisation for fast approximation  
-  Publication-ready figures (2D & 3D trajectories, MAC overlays, temperature paths)  
-  Complete `.xlsx` optimisation outputs for all scenarios  
-  A fully reproducible Python/Conda environment (`GassesPaper_environment.yml`)
---

## Getting Started

### 1️⃣ Clone the repository

```bash
git clone https://github.com/kavehsn/Short-Lived-Gasses-Paper.git
cd Short-Lived-Gasses-Paper
```
### 2️⃣ Create & activate environment

```bash
conda env create -f GassesPaper_environment.yml -n gasses-paper-env
conda activate gasses-paper-env
```
### 3️⃣ Launch JupyterLab

```bash
jupyter lab
```
---
## Running the Optimizations on a High-Performance Computing Cluster (HPC)

The optimizations carried out in this study require the use of a [high-performance computing cluster](https://www.imperial.ac.uk/computational-methods/hpc/), in our case the Imperial College London system, which uses [PBS](https://en.wikipedia.org/wiki/Portable_Batch_System) as its job scheduler.

Running the optimization codes involves:

1. **Logging in to the HPC server**

   ```bash
   ssh username@login.hpc.ic.ac.uk
   ```
2. **Creating / activating the Conda environment**
   As discussed in the earlier section (using the provided `GassesPaper_environment.yml`)
3. **Copying the necessary files to your HPC working directory**
   - All files in the optimization folders:
     - [Grid search](https://github.com/kavehsn/Short-Lived-Gasses-Paper/tree/64e52102a70f3ad6fe3715a8d6183a2f870881be/HPC%20Optimization%20(GridSearch))
     - [stochastic gradient descent](https://github.com/kavehsn/Short-Lived-Gasses-Paper/tree/main/HPC%20Optimization%20(SGD))
   - The calibration [picke files](https://github.com/kavehsn/Short-Lived-Gasses-Paper/tree/main/Pickle%20Files) on to your working directory of the server.
4. **Submitting the optimization jobs**
   For the **grid-search** implementation:
  ```bash
  python Submit_Optimization_GridSearch_PBS.py
  ```
  **For the stochastic gradient descent implementation:**
  ```bash
  python Submit_Optimization_SGD_PBS.py
  ```
