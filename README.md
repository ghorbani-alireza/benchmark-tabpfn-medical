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

## Troubleshooting on a server or network filesystem

Use the normal setup above first. The fixes below are only needed when the
corresponding error appears. Open a terminal in your actual project directory;
the path may differ between computers. No administrator access is needed for
these fixes.

### `uv: command not found`

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

### `No locks available` during installation or checkpoint download

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

These settings move download caches to `/tmp` without changing the source code,
result locations, or `.venv` location. `UV_LINK_MODE=copy` avoids the hardlink
warning when the cache and project are on different filesystems. See the
[uv cache documentation](https://docs.astral.sh/uv/concepts/cache/) and
[Hugging Face cache settings](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables).

The exports apply only to this terminal and programs started from it. If using
the notebook, start Jupyter from this terminal after setting them. If Jupyter is
already running, stop it and relaunch it from the configured terminal so its
kernels inherit the settings. A new terminal needs the settings again.
`/tmp` may be cleared by the server and is usually specific to
one machine, so checkpoints may need downloading again. For repeated runs, ask
which local scratch directory supports locks and has enough space for model
checkpoints. If another lock error names `.venv` rather than a cache, changing
the cache is not sufficient; check the supported environment location with the
server administrator.

### Missing script or `cannot import name 'SETTINGS' from 'benchmark'`

First check that this copy contains the necessary files:

```bash
ls -l scripts/run_analysis.py src/benchmark/__init__.py src/benchmark/settings.py
```

If any are missing, confirm that the files were committed and pushed from the
original computer, then update this copy before continuing. Reinstalling cannot
restore source files that are absent from the repository.

If the files exist but the import fails, inspect what Python loads:

```bash
uv run --locked --no-sync python -c "import benchmark; print('File:', benchmark.__file__); print('Locations:', list(benchmark.__path__)); print('Has SETTINGS:', hasattr(benchmark, 'SETTINGS'))"
```

The file should be this project's `src/benchmark/__init__.py`. In the observed
server failure, it was `None` and `Has SETTINGS` was `False`: stale editable
installation metadata treated the directory as a namespace package. Rebuilding
only the project installation fixed it:

```bash
uv sync --locked --group notebook --reinstall-package benchmark-tabpfn-medical
uv run --locked --no-sync python -c "from benchmark import SETTINGS; print(list(SETTINGS))"
```

The check should print `[1, 2, 3, 4, 5, 6, 7, 8]`. The reinstall updates the
project's installation in `.venv`, not its source code. Use it for this import
problem rather than as a required step for every run. Select this project's
`.venv` kernel in the notebook and restart it after code updates.

### First repetition stays at 0%, or a failed run leaves an output folder

The simulation command runs all eight baseline settings sequentially, with the
requested number of repetitions for each setting. One iteration generates new
training and test samples and evaluates all 12 models. The progress bar advances
only after the entire iteration finishes and resets for the next setting.
Checkpoint downloads and model messages are captured in the log, so the first
iteration can take longer. The foundation-model runners currently use CPU even
if the machine has a GPU. A bar at 0% alone does not establish that a run is stuck.

From a second terminal, inspect the current setting's log. For setting 1 saved
under `io/sim_results/main`, use:

```bash
tail -n 30 io/sim_results/main/simulation_setting1.warnings.log
```

Use the actual output folder and setting number. Logs are written while the
analysis runs; results CSVs are saved after a setting completes. A quiet log
alone does not establish that computation has stopped either.

A failed run may already have created its output directory. Keep it for checking
the warnings and log, and choose a new name for the retry:

```bash
uv run --locked python scripts/run_analysis.py simulation --repetitions 100 --output io/sim_results/main_retry1
```

This starts a new analysis; it does not resume the failed run. Choose another
unused name if `main_retry1` already exists.

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

Do not delete `.venv` while using it. The metadata and caches can be regenerated;
you do not need to publish them. Ignore rules do not remove files already tracked
by Git.


## Bayes AUC reference on simulation boxplots

Bayes AUC uses the known class distributions in `settings.py`. The notebook keeps `SHOW_BAYES_AUC = True`. Change it to `False` to hide the line.

```bash
uv run --locked python scripts/run_analysis.py report --results io/sim_results/main
# Same report, without the reference line:
uv run --locked python scripts/run_analysis.py report --results io/sim_results/main --no-bayes-auc
```
For a more precise estimate, call
`calculate_bayes_auc(cfg, n_pairs=1_000_000, seed=314159)` and pass its returned
dictionary under the relevant setting key to the boxplot function.
