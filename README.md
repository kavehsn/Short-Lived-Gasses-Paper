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
