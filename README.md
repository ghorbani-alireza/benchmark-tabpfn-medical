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

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---






