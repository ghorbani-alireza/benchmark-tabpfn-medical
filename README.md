# Benchmarking TabPFN on medical tabular data

A Python benchmark comparing logistic regression, Random Forest, XGBoost,
CatBoost, QUADRO, TabPFN (versions 2.5, 2.6, 3, & 3.5), EXAONE Tabular (only pilot run, look at the notebook), and TabICLv2.
Experiments use synthetic scenarios or repeated sampling from three medical datasets.


## 1. Prepare the environment

Clone the repo and Change your directory:

```bash
git clone YOUR_REPOSITORY_URL tabular-benchmark
cd tabular-benchmark
```

From the project directory, run:

```bash
uv sync --locked --group notebook
```

This installs the project and dependencies into `.venv` using Python 3.13 and
`uv.lock`. Use the same environment for the notebook and terminal commands.
The `uv` program must be installed first.

--------------------------------------------------------------------------

**Troubleshooting**: Use the normal setup above first. The fixes below are only needed when the
corresponding error appears. Open a terminal in your actual project directory;
the path may differ between computers. No administrator access is needed for
these fixes.

a. `uv: command not found`

If uv is not installed, use the [official Linux installer](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If installation succeeded but the current terminal cannot find uv, load its
environment and check the version:

```bash
source "$HOME/.local/bin/env"
uv --version
```

If `curl` already works, there is no need to install it or run `sudo apt`.
`uv sync --locked --group notebook` creates `.venv` automatically; you do not
need to create a virtual environment separately.

b. `No locks available` during installation or checkpoint download

Some network filesystems do not provide the file locks used by uv and model
downloads. A uv cache fix alone does not relocate the TabPFN or Hugging Face
model caches. In the same terminal used for installation and analysis, run:

```bash
export BENCHMARK_CACHE="$(mktemp -d /tmp/benchmark-cache-XXXXXXXX)"
export UV_CACHE_DIR="$BENCHMARK_CACHE/uv"
export TABPFN_MODEL_CACHE_DIR="$BENCHMARK_CACHE/tabpfn"
export HF_HOME="$BENCHMARK_CACHE/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export UV_LINK_MODE=copy

uv sync --locked --group notebook
```

These settings move download caches to `/tmp` without changing the source code.

The exports apply only to this terminal and programs started from it. If using
the notebook, start Jupyter from this terminal after setting them. 

-----------------------------------------------------------------------------


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



## 2. Run the full analyses from the terminal

These commands run all model runners, excluding EXAONE.  

The main simulation command uses `simulation()`, the normal sequential loop.
Only the sensitivity command uses `evaluate_on_sim_data()`, which runs Monte
Carlo repetitions in parallel with 10 worker processes, each using one computation
thread. Real-data resampling uses the same process policy. Settings run one after another;
models within each repetition run sequentially. The progress bar counts completed
repetitions. The notebook pilot and main simulation stay sequential; its
sensitivity cell uses the same ten-process evaluation function.

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
or dataset saves a results CSV, metadata JSON, and a log of model
messages. Arrays are saved in full. 


Real-data resampling currently samples rows;
patient identifiers are not used to separate training and testing patients.


### CPU and thread policy

Importing `benchmark` sets OpenMP, OpenBLAS, MKL, BLIS, NumExpr, and PyTorch
computation limits to one thread. Import it before model computations; restart
an existing notebook kernel to load this policy. The startup settings override
inherited numerical-library thread counts so each method uses the same limit.
CatBoost prediction and validation scoring explicitly use one thread; QUADRO's
HiGHS solver is also restricted separately. Foundation models use CPU and one
ensemble member.

Main simulation stays sequential. Sensitivity and resampling use Joblib's `loky`
process backend with 10 workers by default. Each process initializes PyTorch once
and restricts numerical-library pools to one thread. Set `--jobs 2` on those
terminal commands if you need fewer workers; Python calls accept `n_jobs=2`.
Process workers each hold their own models, so ten workers need more memory than
one sequential run. Model workers and their inner grid searches stay sequential.

The code controls computation threads, not CPU affinity or server allocations.
Request one CPU for main simulation and ten CPUs for the default parallel runs
through the server's scheduler. Use its affinity options if strict CPU placement
is required; ten workers alone do not guarantee ten distinct physical cores.
Parallel wall times include resource contention; use sequential runs for isolated
per-method runtime comparisons.

The parent displays the progress bar. Every parallel iteration saves private
warnings and logs in a companion `*.workers/` directory. Completed workers'
messages and warnings are also collected into the main `.log` and warnings JSON.
If a worker fails, check its individual files as well: it may not have returned
its warnings to the parent yet. Keep `--output` directories new on retries.

## 3. Bayes AUC and ROC references

Bayes AUC uses the known class distributions in `settings.py`. The notebook keeps 
`SHOW_BAYES_AUC = True`. Change it to `False` to hide the line.

```bash
uv run --locked python scripts/run_analysis.py report --results io/sim_results/main
# Same report, without the reference line:
uv run --locked python scripts/run_analysis.py report --results io/sim_results/main --no-bayes-auc
```
For a more precise estimate, call
`calculate_bayes_auc(cfg, n_pairs=1_000_000, seed=314159)` and pass its returned
dictionary under the relevant setting key to the boxplot function.


The same known class distributions provide a **Bayes ROC** reference, shown as a
black dashed curve with a label beside it. It uses the true log likelihood ratio
of the positive and negative classes, independently of the fitted methods.
This is the optimal population ROC for the specified distributions; see this
[likelihood-ratio ROC reference](https://arxiv.org/abs/2202.01956).
AUC alone is not enough to reconstruct an ROC curve: the distribution parameters
are needed. Equal-covariance normal settings use the analytic ROC; other settings
use 100,000 independent observations per class with seed 314159. The numerical
ROC is stored on a compact FPR grid. It estimates the population optimum, so
individual test-sample curves can cross it. No Bayes reference is calculated
for real datasets with unknown class distributions.

The notebook keeps `SHOW_BAYES_ROC = True`; set it to `False` to hide the curve.
The plotting function accepts `bayes_auc=references, show_bayes_roc=False` for the
same purpose. In terminal reports, use:

```bash
uv run --locked python scripts/run_analysis.py report --results io/sim_results/main
# Hide only the Bayes ROC (the boxplot AUC line remains on):
uv run --locked python scripts/run_analysis.py report --results io/sim_results/main --no-bayes-roc
```

New Bayes references save `fpr`, `tpr`, and `roc_auc` alongside the existing AUC
and its Monte Carlo standard error. `roc_auc` is the empirical ROC area and can
differ slightly from the independent-pair `auc` estimate; both estimate the same
population quantity. Saved results with complete distribution metadata can have
the curve calculated at report time without rerunning models. Older references
keep their saved AUC value. If distributions are missing and no curve was saved,
the ROC reference is omitted.

## 4. Tables and plots from saved results

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
The script downloads the listed upstream tags, renames package imports, preserves 
license files and small package assets, and records the source commit. 

| Included package | Upstream tag |
|---|---|
| `tabpfn_v25` | `v6.0.0` |
| `tabpfn_v26` | `v7.1.1` |
| `tabpfn_v3` | `v8.5.0` |
| `tabpfn_v35` | `v9.0.0` |


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

