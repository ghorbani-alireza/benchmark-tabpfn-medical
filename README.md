# Benchmarking TabPFN on medical tabular data

Built with PriorLabs-TabPFN.

Use the notebook for the setting-2 pilot. Use the terminal for the main
simulations, sensitivity analysis, real-data resampling, and saved-result reports.

## 1. Prepare the environment

From the project directory, run:

```bash
uv sync --locked --group notebook
```

This installs the project and dependencies into `.venv` using Python 3.13 and
`uv.lock`. Use the same environment for the notebook and terminal commands.
The `uv` program must be installed first.

Credentials belong in the ignored file `.secrets/token.py`:

```python
HF_TOKEN = "your_huggingface_token"
TABPFN_TOKEN = "your_tabpfn_token"
```

Existing environment variables take priority. Pretrained models may need access
to download their checkpoints; the included vendor source is separate from those
checkpoints.

For real-data resampling, obtain the datasets through [OSF](https://osf.io/hjs92)
and put these files in `io/real_data/`:

- `data1_echo_notes.pkl`
- `data2_blood_glucose_management.pkl`
- `data3_blood_gas_oximetry.pkl`

## 2. Run the pilot in the notebook

Start Jupyter:

```bash
uv run --locked jupyter lab notebooks/run_analysis.ipynb
```

Select this project's `.venv/bin/python` kernel (Windows:
`.venv/Scripts/python.exe`). Keep `RUN_FULL_ANALYSIS = False` and run the pilot
cells. The pilot uses **setting 2, five Monte Carlo repetitions, and all models
except EXAONE**. Keep `SAVE_PILOT_OUTPUT = False` to display results and plots
in memory without writing pilot CSVs, tables, or figures. Warnings and model
messages are still saved under `io/sim_results/warnings/`.

Set `SAVE_PILOT_OUTPUT = True` if you want to keep the pilot results in
`io/sim_results/pilot_setting2/`. In that mode, the plotting section reloads the
saved results and can be rerun without fitting models. Optional pilot files are
ignored by Git. Clear executed notebook cell outputs before committing the
notebook; keep the notebook source itself in Git.

## 3. Run the full analyses from the terminal

These commands run all model runners, including EXAONE. Choose the repetition
count for your intended protocol. The counts below are examples, not a claim
that these reproduce a particular published analysis.

Main simulation: all eight baseline settings:

```bash
uv run --locked python scripts/run_analysis.py simulation --repetitions 100 --output io/sim_results/main
```

Sensitivity analysis: the nine sensitivity settings, including setting 7+:

```bash
uv run --locked python scripts/run_analysis.py sensitivity --repetitions 100 --output io/sim_results/sensitivity
```

Real-data resampling: all three datasets:

```bash
uv run --locked python scripts/run_analysis.py resampling --repetitions 500 --output io/real_data_results/main
```

Each output directory must be new. Choose another name when repeating an
analysis. Omit `--output` to generate a unique directory automatically. You can
also specify `--seed`; the defaults are 2025 for main simulation/resampling and
2026 for sensitivity. Scenario definitions are in `src/benchmark/settings.py`.
The terminal simulation command covers the eight baseline settings; the optional
large-sample setting-2 experiment remains in the notebook's full-analysis cells.

During model execution, only the main progress bar is displayed. Each setting
or dataset saves a results CSV, metadata JSON, warnings JSON, and a log of model
messages. Arrays are saved in full. Real-data resampling currently samples rows;
patient identifiers are not used to separate training and testing patients.

## 4. Make tables and plots from saved results

Reporting uses the saved CSV files, without fitting models or reading raw data:

```bash
uv run --locked python scripts/run_analysis.py report --results io/sim_results/main
uv run --locked python scripts/run_analysis.py report --results io/sim_results/sensitivity
uv run --locked python scripts/run_analysis.py report --results io/real_data_results/main
```

The same command works for the pilot directory when `SAVE_PILOT_OUTPUT = True`
was used to save it. Each report saves `tables/` and
`plots/` inside the selected results directory. It creates a measures table,
two AUC boxplot layouts, ROC curves, calibration curves, Brier-versus-AUC plots,
and a runtime bar chart. The notebook also retains its main results section for
existing baseline, sensitivity, and real-data files. Keep `SHOW_MAIN_RESULTS = True`
to display them while `RUN_FULL_ANALYSIS = False` prevents model execution.

## Optional: recreate the included vendor sources

The four renamed TabPFN source packages are included in `vendor/`. **Normal
analysis does not require running the setup script again.**

The setup script needs only Python 3 and Git, so it can run before the project
environment is installed. If a source copy is missing, run:

```bash
python3 scripts/setup_vendor.py
```

Existing copies are skipped. To replace all four source copies deliberately, run:

```bash
python3 scripts/setup_vendor.py --force
uv sync --locked --group notebook
```

Rebuilding requires Git and internet access. The script downloads the listed
upstream tags, renames package imports, preserves license files and small package
assets, and records the source commit. It does not download model checkpoints.
Restart the notebook kernel after replacing sources.

| Included package | Upstream tag |
|---|---|
| `tabpfn_v25` | `v6.0.0` |
| `tabpfn_v26` | `v7.1.1` |
| `tabpfn_v3` | `v8.5.0` |
| `tabpfn_v35` | `v9.0.0` |

Vendor sources are allowed by `.gitignore`. They still need to be added and
committed to Git to accompany a repository clone.

## Files

- `src/benchmark/`: data loading, settings, model runners, analysis, and reporting.
- `src/quadro/`: the QUADRO classifier and its mathematical solvers.
- `vendor/`: the four renamed TabPFN source copies.
- `scripts/setup_vendor.py`: optional source rebuilding.
- `scripts/run_analysis.py`: terminal analysis and reporting.
- `notebooks/run_analysis.ipynb`: pilot and interactive results.
- `io/`: datasets, saved results, tables, and plots.

## Licenses

The benchmark's own code uses the MIT license. The TabPFN copies retain their
upstream licenses in their individual vendor folders; those licenses apply to
the copied sources. See the upstream license texts for
[v6.0.0](https://github.com/PriorLabs/TabPFN/blob/v6.0.0/LICENSE),
[v7.1.1](https://github.com/PriorLabs/TabPFN/blob/v7.1.1/LICENSE),
[v8.5.0](https://github.com/PriorLabs/TabPFN/blob/v8.5.0/LICENSE), and
[v9.0.0](https://github.com/PriorLabs/TabPFN/blob/v9.0.0/LICENSE).

## Generated files and tests

Keep the README, notebook source, tests, `pyproject.toml`, `uv.lock`, your source
code, and all license files in Git. The following are generated and ignored:

| Folder | What it contains |
|---|---|
| `.venv/` | Installed Python and dependencies; recreated with `uv sync` |
| `benchmark_tabpfn_medical.egg-info/` | Installation metadata for this project |
| `__pycache__/` | Cached Python bytecode; recreated automatically |
| `.ipynb_checkpoints/` | Jupyter's automatic notebook backup copies |

Do not delete `.venv` while using it. The metadata and caches can be regenerated;
you do not need to publish them. Ignore rules do not remove files already tracked
by Git.

Run the lightweight checks with:

```bash
uv run --locked python -m unittest discover -s tests
```

The tests replace real models with simple mock models. They check quiet output,
warning saving, EXAONE exclusion, notebook pilot settings, and complete array
saving/loading. They do not download checkpoints or validate the paper's model
performance. Each test explains the behavior it checks.

## Bayes AUC reference on simulation boxplots

Bayes AUC uses the known class distributions in `settings.py`, not fitted models.
The score is the log density of the positive class minus that of the negative
class. We estimate the probability that an independent positive observation has
a higher score than an independent negative observation, with half credit for ties.
The default uses 100,000 independent pairs and seed 314159. `mc_se` is its Monte
Carlo standard error; an approximate numerical 95% interval is AUC ± 1.96 × mc_se.
This is a numerical estimate of the population optimum, not an exact ceiling for
individual finite-test-sample results. Equal-covariance normal settings use the
exact formula. Class proportions and training sample size do not change this AUC.
For t settings, the covariance-to-shape conversion matches the simulation code.

The notebook keeps `SHOW_BAYES_AUC = True`. Change it to `False` to hide the line.
Both boxplot functions accept `bayes_auc=references, show_bayes_auc=False` to hide
it; the default is `True`. Lines appear only for AUC, when a reference is supplied.
The pilot displays the estimate and numerical uncertainty separately from model results.
New synthetic runs save the reference and full distribution parameters in `.meta.json`.
Reports reuse the saved reference, or calculate it from complete saved parameters:

```bash
uv run --locked python scripts/run_analysis.py report --results io/sim_results/main
# Same report, without the reference line:
uv run --locked python scripts/run_analysis.py report --results io/sim_results/main --no-bayes-auc
```

Older metadata missing means or covariances cannot supply a reliable reference;
those panels omit the line. Real-data resampling also omits it because the true
class distributions are unknown. No benchmark models need to be rerun to report
results with complete distribution metadata. For a more precise estimate, call
`calculate_bayes_auc(cfg, n_pairs=1_000_000, seed=314159)` and pass its returned
dictionary under the relevant setting key to the boxplot function.
