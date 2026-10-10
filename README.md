# Benchmarking TabPFN on Tabular Data
  
 


## Repository Structure

```

benchmark-tabpfn-medical/
│
├── pyproject.toml                          # metadata and dependencies
├── requirements.txt                        # pinned deps
├── README.md
│
├── src/                                    # source code
    ├── __init__.py
    ├── config.py                           # paths (from __file__)
    ├── data_loader.py                      # real-data loaders
    ├── settings.py                         # synthetic simulation settings
    ├── model_definitions.py                # model runners
    ├── simulation_core.py                  # synthetic simulation loop
    ├── resampling_core.py                  # real-data resampling loop
    ├── utils.py                            # save/load/discover helpers
    ├── table_functions.py                  # summary tables
    ├── plot_functions.py                   # figures
    ├── setup_vendor.py                     # one-time: clone 4 TabPFN versions
    └── tokens.py                           # HF / PriorLabs tokens
│
├── notebooks/                              # jupyter notebooks
    └── run_analysis.ipynb
│
├── vendor/                                 # vendored TabPFN packages
    ├── tabpfn_v25/
    ├── tabpfn_v26/
    ├── tabpfn_v3/
    └── tabpfn_v35/
│
└── io/                                     # input/output
    ├── real_data/                          # place datasets here
        ├── data1_echo_notes.pkl
        ├── data2_blood_glucose_management.pkl
        ├── data2_blood_glucose_management.pkl
    └── data3_blood_gas_oximetry.pkl
    ├── sim_results/                        # saved simulation outputs
    ├── real_data_results/                  # saved real-data outputs
    ├── tables/                             # generated tables
    └── plots/                              # generated figures

```

---

## Datasets

The three datasets are available on **OSF**:  
[https://osf.io/hjs92](https://osf.io/hjs92)

After downloading, place the `.pkl` files inside `io/real_data/`. 

---

## Tokens

To run TabPFN, you need tokens from **Hugging Face** and **PriorLabs**.

### 1. Obtain tokens
- **Hugging Face**: Sign up at [huggingface.co](https://huggingface.co), go to Settings → Access Tokens, and create a new token.
- **PriorLabs**: Sign up at [priorlabs.ai](https://priorlabs.ai) and generate an API token from your dashboard.

### 2. Place tokens in `src/tokens.py`
1. In the `src/` folder, you will find a file named **`tokens.py.txt`**.
2. **Rename it** to **`tokens.py`** (remove the `.txt` extension).
3. Open `tokens.py` and replace the placeholder values with your actual tokens:

```python
# src/tokens.py
HF_TOKEN = "your_huggingface_token_here"
TABPFN_TOKEN = "your_priorlabs__tabpfn_token_here"
```

---

## License
This project is distributed under the MIT License.

---






