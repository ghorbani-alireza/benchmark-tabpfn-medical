# Benchmarking TabPFN on Medical-Like Tabular Data

**Authors:** Alireza Ghorbani, Mohammad Khajehzadeh, Anne‑Laure Boulesteix, Moritz Herrmann  
**Corresponding author:** ghorbani.alireza@campus.lmu.de  
**OSF Repository:** [https://osf.io/hjs92](https://osf.io/hjs92)  
**GitHub Repository:** [https://github.com/yourusername/benchmark-tabpfn-medical](https://github.com/yourusername/benchmark-tabpfn-medical)

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
4. [License](#license)

---

## Overview

This repository contains the complete code for the paper *"Benchmarking TabPFN on Medical‑Like Tabular Data"*. We compare TabPFN against five well‑established methods – linear and polynomial logistic regression, Random Forest, XGBoost, and CatBoost – across eight simulated scenarios and three real‑world clinical datasets. The study evaluates discriminative performance (ROC‑AUC), calibration (Brier score, MSE), and runtime.

All results presented in the manuscript are fully reproducible by running the scripts in this repository. Raw data and pre‑computed results are also available on the [OSF project page](https://osf.io/hjs92).

---

## Repository Structure
benchmark-tabpfn-medical/
├── README.md
├── requirements.txt
├── environment.yml
├── real_data_study/
│ ├── results/
│ ├── 20250723_case_study_1-echo_note.ipynb
│ ├── 20250811_case_study_2_blood_glucose_management.ipynb
│ └── 20250902_case_study_3-bold_blood-gas_and_oximetry.ipynb
|
├── simualtion_study/
│ ├── 11-17-2025-tabpfn_simulation_study.ipynb
│ ├── plots\
│ ├── sim_results\
│ └── stables\

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or issues, please open an issue on GitHub or contact:

- Alireza Ghorbani – ghorbani.alireza@campus.lmu.de
- Moritz Herrmann – moritz.herrmann@lmu.de

---

**Last updated:** July 2026




