# Benchmarking TabPFN on medical tabular data

A Python benchmark comparing logistic regression, Random Forest, XGBoost,
CatBoost, QUADRO, TabPFN (2.5, 2.6, 3, and 3.5), EXAONE Tabular, and TabICL v2.
Experiments use synthetic scenarios or repeated sampling from three medical datasets.


## 1. Prepare the environment

Clone the repository and enter its directory:

```bash
git clone YOUR_REPOSITORY_URL tabular-benchmark
cd tabular-benchmark
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

To run a subset, add `--settings` to simulation/sensitivity or `--datasets` to
resampling. Each option accepts one or more space-separated values:

```bash
# Main settings 2 and 5 only
uv run --locked python scripts/run_analysis.py simulation --settings 2 5 --repetitions 100 --output io/sim_results/main_subset

# Sensitivity settings 2 and 7+ only
uv run --locked python scripts/run_analysis.py sensitivity --settings 2 '7+' --repetitions 100 --output io/sim_results/sensitivity_subset

# Echo and blood-gas datasets only
uv run --locked python scripts/run_analysis.py resampling --datasets echo blood_gas --repetitions 500 --output io/real_data_results/subset
```

Main setting IDs are `1` through `8`; sensitivity also accepts `7+`. Dataset
names are `echo`, `glucose`, and `blood_gas`. Use one value to select a single
setting or dataset; omit the option to run all of them.

### Execution and outputs

Main simulation runs sequentially. Sensitivity and resampling run repetitions
in parallel with 10 worker processes by default. Add `--jobs 2` to use fewer
workers and reduce memory use, or `--jobs 1` to run sequentially. Models run on
CPU with one computation thread per process; foundation models use one ensemble
member.

Settings and datasets run one after another. The progress bar shows the current
setting or dataset and advances after a repetition finishes evaluating all models.

Each `--output` directory must be new; omit `--output` for an automatic directory
name. Results include CSV files and metadata. Model warnings and logs are saved
under `warnings/`, which Git ignores. Parallel runs keep per-iteration logs in
`warnings/*.workers/`; check these when a worker fails.

Use `--seed` to override the default seed: 2025 for main simulation/resampling,
2026 for sensitivity. Scenario definitions are in `src/benchmark/settings.py`;
the extra large-sample setting-2 experiment is in the notebook.

Real-data resampling splits rows, so the same patient may appear in both training
and test data.

## 3. Tables and plots from saved results

Reporting uses the saved CSV files, without fitting models or reading raw data:

```bash
uv run --locked python scripts/run_analysis.py report --results io/sim_results/main
uv run --locked python scripts/run_analysis.py report --results io/sim_results/sensitivity
uv run --locked python scripts/run_analysis.py report --results io/real_data_results/main
```

Reports save `tables/` and `plots/` inside the selected results directory. The
same command works for a saved pilot directory.

For notebook results, set `SHOW_MAIN_RESULTS = True` and execute the results
cells. `RUN_FULL_ANALYSIS = False` skips the full analysis cells; the pilot runs
in its own cell.

## Vendor sources

The four TabPFN source packages are included in `vendor/`. No additional vendor
setup is needed for normal runs.

## Files

- `src/benchmark/`: data loading, settings, model runners, analysis, and reporting.
- `src/quadro/`: the QUADRO classifier and its mathematical solvers.
- `vendor/`: the four renamed TabPFN source copies.
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

