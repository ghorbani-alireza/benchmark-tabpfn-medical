# Benchmarking TabPFN on medical tabular data

**Built with PriorLabs-TabPFN**

A Python benchmark comparing logistic regression, Random Forest, XGBoost,
CatBoost, QDA, QUADRO, TabPFN (2.5, 2.6, 3, and 3.5), EXAONE Tabular, and TabICL v1/v2.
Experiments use synthetic scenarios or repeated sampling from three medical datasets.

QDA uses standardized predictors and selects covariance regularization (`reg_param`)
from `[0.01, 0.05, 0.1, 0.2, 0.5, 1.0]` by validation AUC before a final full-training fit.


## 1. Prepare the environment

Clone the repository and enter its directory:

```bash
git clone https://github.com/ghorbani-alireza/benchmark-tabpfn-medical
cd benchmark-tabpfn-medical
```

From the project directory, run:

```bash
uv sync --locked --group notebook
```

This creates the Python 3.13 environment in `.venv` using `uv.lock`.
Use this environment for both terminal commands and the notebook.

Credentials belong in the ignored file `.secrets/token.py`:

```python
HF_TOKEN = "your_huggingface_token"
TABPFN_TOKEN = "your_tabpfn_token"
```

For real-data resampling, obtain the datasets through [OSF](https://osf.io/hjs92)
and put these files in `io/real_data/`:

- `data1_echo_notes.pkl`
- `data2_blood_glucose_management.pkl`
- `data3_blood_gas_oximetry.pkl`

### Troubleshooting

**`uv: command not found`**

If uv is not installed, use the [official Linux installer](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If uv is installed but the terminal cannot find it:

```bash
source "$HOME/.local/bin/env"
uv --version
```

**`No locks available` during installation or model download**

On servers with this error, move both installation and model caches to `/tmp`.
From the project directory, run:

```bash
export BENCHMARK_CACHE="$(mktemp -d /tmp/benchmark-cache-XXXXXXXX)"
export UV_CACHE_DIR="$BENCHMARK_CACHE/uv"
export TABPFN_MODEL_CACHE_DIR="$BENCHMARK_CACHE/tabpfn"
export HF_HOME="$BENCHMARK_CACHE/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export UV_LINK_MODE=copy

uv sync --locked --group notebook
```

Run the analysis or start Jupyter from this same terminal so the cache settings
apply. For a batch job, include these exports in the job script.

## 2. Run the full analyses from the terminal

The pilot includes EXAONE; the full analyses run all other model runners.

Pilot: setting 2, 20 repetitions, using the normal sequential `simulation()` function:

```bash
uv run --locked python scripts/run_analysis.py pilot --output io/sim_results/pilot_setting2_new
```

The pilot also accepts `--repetitions`, `--seed`, and `--settings`. Choose a new
output directory for each run.

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

To run a subset, add `--settings 2 5` to simulation/sensitivity or `--datasets echo blood_gas` to
resampling. Each option accepts one or more space-separated values:


**`FileNotFoundError` for a TabPFN checkpoint during a parallel run** First run one repetition with a single worker to download and verify the models:

```bash
uv run --locked python scripts/run_analysis.py sensitivity --settings 1 --repetitions 1 --jobs 1 --output io/sim_results/worker_download
```

After it succeeds, retry.


Main setting IDs are `1` through `8`; sensitivity also accepts `7+`. Dataset
names are `echo`, `glucose`, and `blood_gas`. Use one value to select a single
setting or dataset; omit the option to run all of them.



## 3. Tables and plots from saved results

Reporting uses the saved CSV files, without fitting models or reading raw data:

```bash
uv run --locked python scripts/run_analysis.py report --results io/sim_results/main
uv run --locked python scripts/run_analysis.py report --results io/sim_results/sensitivity
uv run --locked python scripts/run_analysis.py report --results io/real_data_results/main
```

For sensitivity changes relative to the matching main simulation settings:

```bash
uv run --locked python scripts/run_analysis.py report --results io/sim_results/sensitivity --baseline io/sim_results/main
```

Reports save `tables/` and `plots/` inside the selected results directory. The
same command works for a saved pilot directory.

Tables include:

- `measures.csv`: each method's mean and SD.
- `best_methods.csv`: the best and second-best classical and foundation methods per measure.
- `best_vs_best.csv`: within-run foundation minus classical differences, summarized by mean and SD.
- `model_effects.csv`: each model versus the best method in the other group, with the same difference convention.
- `sample_size_changes.csv`: sensitivity mean minus baseline mean for every measure (`--baseline` required).

Rankings use averages across repetitions, with higher AUC or lower error/time
preferred. Calibration ranks use mean absolute distance from slope 1 or intercept
0. Effects retain the raw measure: a positive AUC difference favors the foundation
model; negative error/time differences favor it. Rankings and comparisons are
descriptive summaries of the saved runs.

In the notebook, run **Tables from saved results**, then the plot cells. Set
`RESULTS_DIR` to the saved pilot, simulation, sensitivity, or real-data directory;
set `BASELINE_DIR` for sensitivity changes. These cells run without the pilot or
package-check cells. Restart the kernel after changing model code.

## Vendor sources

The four TabPFN source packages are included in `vendor/`. No additional vendor
setup is needed for normal runs.
Their local names (`tabpfn_v25`, `tabpfn_v26`, `tabpfn_v3`, and `tabpfn_v35`)
allow the versions to be imported together for side-by-side benchmarking.

## Foundation models

Default ensemble settings and pretrained checkpoint filenames (check out notebooks/run_analysis.ipynb):

| Model | Ensemble setting | Checkpoint filename |
|--------------------|----------|----------------------------------------------------|
| TabPFN 2.5         | `8`      | `tabpfn-v2.5-classifier-v2.5_default.ckpt`         |
| TabPFN 2.6         | `8`      | `tabpfn-v2.6-classifier-v2.6_default.ckpt`         |
| TabPFN 3           | `"auto"` | `tabpfn-v3-classifier-v3_default.ckpt`             |
| TabPFN 3.5         | `"auto"` | `tabpfn-v3.5-20260909.safetensors`                 |
| EXAONE Tabular 1.0 | `8`      | `exaone-tabular-classifier-v1_default.safetensors` |
| TabICL v1          | `8`      | `tabicl-classifier-v1-20250208.ckpt`               |
| TabICL v2          | `8`      | `tabicl-classifier-v2-20260212.ckpt`               |

Both TabICL runners leave the ensemble size unset and use the installed package's
default (currently 8). Checkpoints select the original v1 and v2 models using
the [TabICL API](https://tabicl.readthedocs.io/en/latest/api.html).

## Files

- `src/benchmark/`: data loading, settings, model runners, analysis, and reporting.
- `src/quadro/`: the QUADRO classifier and its mathematical solvers.
- `vendor/`: the four renamed TabPFN source copies.
- `scripts/run_analysis.py`: terminal analysis and reporting.
- `notebooks/run_analysis.ipynb`: model checks, pilot, and reports from saved results.
- `io/`: datasets, saved results, tables, and plots.

## Licenses

The benchmark's own code uses the MIT license. The TabPFN copies retain their
upstream licenses in their individual vendor folders; those licenses apply to
the copied sources. See the upstream license texts for
[v6.0.0](https://github.com/PriorLabs/TabPFN/blob/v6.0.0/LICENSE),
[v7.1.1](https://github.com/PriorLabs/TabPFN/blob/v7.1.1/LICENSE),
[v8.5.0](https://github.com/PriorLabs/TabPFN/blob/v8.5.0/LICENSE), and
[v9.0.0](https://github.com/PriorLabs/TabPFN/blob/v9.0.0/LICENSE).

