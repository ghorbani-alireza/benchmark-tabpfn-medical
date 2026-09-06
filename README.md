# Benchmarking TabPFN on Tabular Data
  
 


## Repository Structure

```

benchmark-tabpfn-medical/
│
├── pyproject.toml                          # metadata and dependencies
├── requirements.txt                        
├── README.md
│
├── src/                                    # source code
│ ├── init.py
│ ├── config.py
│ ├── data_loader.py
│ ├── model_definitions.py
│ ├── simulation_core.py
│ ├── resampling_core.py
│ ├── utils.py
│ ├── table_functions.py
│ └── plot_functions.py
│
├── notebooks/                               # jupyter 
│ └── run_analysis.ipynb
│
└── io/                                      # input/output data and results
├── real_data/                               # place the OSF datasets in this dir
│ ├── data1_echo_notes.pkl
│ ├── data2_blood_glucose_management.pkl
│ └── data3_blood_gas_oximetry.pkl
├── sim_results/                             # saved simulation outputs
├── real_data_results/                       # saved real‑data resampling outputs
├── tables/                                  # generated tables
└── plots/                                   # generated figures


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






