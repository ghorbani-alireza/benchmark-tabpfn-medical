# Benchmarking TabPFN on medical tabular data

A Python benchmark comparing logistic regression, Random Forest, XGBoost,
CatBoost, QUADRO, TabPFN (2.5, 2.6, 3, and 3.5), EXAONE Tabular, and TabICL v2.
Experiments use synthetic scenarios or repeated sampling from three medical datasets.


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

These commands run all model runners except EXAONE, which is enabled in the
notebook pilot.

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

Reports save `tables/` and `plots/` inside the selected results directory. The
same command works for a saved pilot directory.

The notebook contains the 20-repetition pilot, foundation-model checks, and pilot
tables and plots. Run full analyses and report their saved results with the
terminal commands above. Restart the notebook kernel after changing model code.

## Vendor sources

The four TabPFN source packages are included in `vendor/`. No additional vendor
setup is needed for normal runs.

## Foundation models

Default ensemble settings and downloaded pretrained checkpoints (check out notebooks/run_analysis.ipynb):

| Model | Ensemble setting | Checkpoint filename |
|--------------------|----------|----------------------------------------------------|
| TabPFN 2.5         | `8`      | `tabpfn-v2.5-classifier-v2.5_default.ckpt`         |
| TabPFN 2.6         | `8`      | `tabpfn-v2.6-classifier-v2.6_default.ckpt`         |
| TabPFN 3           | `"auto"` | `tabpfn-v3-classifier-v3_default.ckpt`             |
| TabPFN 3.5         | `"auto"` | `tabpfn-v3.5-20260909.safetensors`                 |
| EXAONE Tabular 1.0 | `8`      | `exaone-tabular-classifier-v1_default.safetensors` |
| TabICL v2          | `8`      | `tabicl-classifier-v2-20260212.ckpt`               |

## Files

- `src/benchmark/`: data loading, settings, model runners, analysis, and reporting.
- `src/quadro/`: the QUADRO classifier and its mathematical solvers.
- `vendor/`: the four renamed TabPFN source copies.
- `scripts/run_analysis.py`: terminal analysis and reporting.
- `notebooks/run_analysis.ipynb`: model checks, pilot, and pilot results.
- `io/`: datasets, saved results, tables, and plots.

## Licenses

The benchmark's own code uses the MIT license. The TabPFN copies retain their
upstream licenses in their individual vendor folders; those licenses apply to
the copied sources. See the upstream license texts for
[v6.0.0](https://github.com/PriorLabs/TabPFN/blob/v6.0.0/LICENSE),
[v7.1.1](https://github.com/PriorLabs/TabPFN/blob/v7.1.1/LICENSE),
[v8.5.0](https://github.com/PriorLabs/TabPFN/blob/v8.5.0/LICENSE), and
[v9.0.0](https://github.com/PriorLabs/TabPFN/blob/v9.0.0/LICENSE).

