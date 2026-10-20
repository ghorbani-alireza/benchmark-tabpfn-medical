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

**`FileNotFoundError` for a TabPFN checkpoint during a parallel run** First run one repetition with a single worker to download and verify the models:

```bash
uv run --locked python scripts/run_analysis.py sensitivity --settings 1 --repetitions 1 --jobs 1 --output io/sim_results/sensitivity_test
```

After it succeeds, retry in the same terminal with the same cache settings:

```bash
uv run --locked python scripts/run_analysis.py sensitivity --repetitions 100 --jobs 10 
```


Main setting IDs are `1` through `8`; sensitivity also accepts `7+`. Dataset
names are `echo`, `glucose`, and `blood_gas`. Use one value to select a single
setting or dataset; omit the option to run all of them.

### Execution and outputs

Main simulation runs sequentially. Sensitivity and resampling run repetitions
in parallel with 10 worker processes by default. Add `--jobs 2` to use fewer
workers and reduce memory use, or `--jobs 1` to run sequentially. Models run on
CPU with one computation thread per process. Foundation models use the ensemble
settings below; TabPFN 3 and 3.5 retain automatic ensemble selection.

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

### Model comparison settings

Every repetition uses the same training/test data for all methods. Classical
methods select parameters by validation AUC on a stratified 80/20 split of the
training data, then fit the selected configuration once on all training rows.
The test set is used for evaluation. Foundation models use pretrained weights
without a hyperparameter search or weight updates.

| Method | Parameters searched |
| --- | --- |
| Linear sparse logistic regression (L-SLR) | L1 penalty; 10 `C` values from `np.logspace(-4, 4, 10)`. |
| Quadratic sparse logistic regression (SLR) | Original features, squares, and pairwise products; the same L1 penalty grid. |
| Random Forest | `n_estimators`: 50, 100, 200; `max_depth`: 3, 5, unlimited. |
| XGBoost | `n_estimators`: 50, 100, 200; `max_depth`: 3, 5, 6; fixed learning rate 0.1. |
| CatBoost | `iterations`: 50, 100, 200; `depth`: 3, 5, 6; fixed learning rate 0.1. |
| QUADRO | `(lambda_quadratic, lambda_linear)`: (0, 0), (0.01, 0.01), (0.01, 0.1), (0.1, 0.1), (0.1, 1). |

Foundation-model ensemble settings follow the defaults of the pinned packages:

| Method | Ensemble setting | Official reference |
| --- | --- | --- |
| TabPFN 2.5 | `n_estimators=8` | [Package v6.0.0](https://github.com/PriorLabs/TabPFN/blob/v6.0.0/src/tabpfn/classifier.py) |
| TabPFN 2.6 | `n_estimators=8` | [Package v7.1.1](https://github.com/PriorLabs/TabPFN/blob/v7.1.1/src/tabpfn/classifier.py) |
| TabPFN 3 | `n_estimators="auto"`; starts at 8 and can increase for feature coverage. | [Package v8.5.0](https://github.com/PriorLabs/TabPFN/blob/v8.5.0/src/tabpfn/preprocessing/ensemble.py) |
| TabPFN 3.5 | `n_estimators="auto"`; uses the checkpoint's count, or an automatic baseline of 8 that can increase for feature coverage. | [Package v9.0.0](https://github.com/PriorLabs/TabPFN/blob/v9.0.0/src/tabpfn/classifier.py) |
| TabICL v2 | `n_estimators=8` | [Package v2.2.0](https://github.com/soda-inria/tabicl/blob/v2.2.0/src/tabicl/_sklearn/classifier.py) |
| EXAONE Tabular | `ensemble_count=8`; included in the notebook pilot. | [Pinned source](https://github.com/LGAI-Research/EXAONE-Tabular/blob/8638e07d09fad154249bd75ba4786181491f7025/src/exaonetabular/presets.py) |

This compares the configured methods using their default ensemble policies;
automatic selection means the ensemble counts need not be identical across models.
The runners in `src/benchmark/model_definitions.py` define these settings.

Calibration slope and intercept are diagnostic estimates fitted jointly on the
test outcomes: `logit(P(y=1)) = intercept + slope * logit(predicted_probability)`.
Predicted probabilities are clipped to [0.001, 0.999] for this calculation only.
The diagnostic logistic regression uses `penalty=None`, `solver="lbfgs"`, and
`max_iter=2000`; it does not modify predictions or other metrics. Brier scores
use the original probabilities; MSE uses predicted class labels.

### Runtime measurements

New runs save the following timings in seconds for every model and repetition:

| Column suffix | What is measured |
| --- | --- |
| `_SetupTime` | Imports, model construction, and explicit checkpoint loading. |
| `_TuneTime` | Candidate fitting, validation scoring, and search overhead, excluding the final refit. Zero for foundation models. |
| `_FitTime` | One final fit on the full training set, including fitting its scaler. For foundation models, preprocessing and training-context preparation in `.fit()`, without updating pretrained weights. |
| `_PredictTime` | Final test predictions, including test transformations and deriving class labels. Evaluation metrics are excluded. |
| `_Time` | Existing total model-wrapper runtime, including setup and evaluation metrics, excluding the separate diagnostic calibration slope/intercept calculation. |

Timers are written directly around the relevant operations in
`src/benchmark/model_definitions.py`. For example:

```python
predict_start = perf_counter()
probabilities = model.predict_proba(X_test)
pred = model.classes_[np.argmax(probabilities, axis=1)]
proba = probabilities[:, 1]
predict_time = perf_counter() - predict_start
```

All runners always return eleven values in this order:

```python
(balanced_accuracy, auc, mse, fpr, tpr, proba, brier,
 setup_time, tune_time, fit_time, predict_time) = run_tabpfn_v25(
    X_train, X_test, y_train, y_test,
)
```

There is no optional timing argument or separate output dictionary. Simulation
and resampling save the four returned times alongside the existing total runtime.

`GridSearchCV` uses `refit=False`: its timed `.fit()` only searches candidates and
scores validation data. The chosen model is then fitted exactly once on the full
training set inside a separate timer. All stages use `time.perf_counter()`, and
all times are elapsed wall-clock seconds per repetition. QUADRO keeps its separate
label and probability calls within prediction time.

Both logistic runners and QUADRO use explicit `StandardScaler` calls. During
tuning, the scaler is fitted only on `X_train_sub` and reused for `X_val`.
During the final fit, a new scaler is fitted on all of `X_train`; prediction
transforms `X_test` with that final scaler. The quadratic logistic model expands
features before scaling. QUADRO's internal feature standardization is disabled
in this wrapper because scaling is already performed explicitly. No pipelines
are used in these runners.

Some libraries load or download weights lazily inside `.fit()`; that work is
included in `_FitTime`, rather than `_SetupTime`. First calls can therefore be
slower than calls with cached weights and imported libraries. Parallel workers
can compete for CPU and memory. Old CSV files retain their existing columns;
rerun experiments to collect the additional timing columns. Tables include the
new timing measurements; the existing runtime plot continues to show `_Time`.

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

The notebook pilot runs 20 repetitions. Its baseline and sensitivity cells retain
their two-repetition demonstration runs; use the terminal commands above for the
full analyses, or update both `iter` and the corresponding notebook metadata.
Restart the kernel after changing model code.

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

