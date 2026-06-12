# Running the Evaluation on a Fresh Linux Machine

A start-to-finish guide to run the SCAPES model evaluation (`evaluate_model.py`)
on a **clean Linux box that has nothing installed**. It covers system packages,
NVIDIA/CUDA, cloning the repo + submodules, obtaining the model weights and
dataset, building **both** Python environments, and running the smoke test and
full benchmark.

If you already have the repo set up, you only need the shorter
[`RUNNING_THE_BENCHMARKS.md`](./RUNNING_THE_BENCHMARKS.md) and the
`# Testing — Model Evaluation` section of the root `README.md`. This document is
the long-form version for a brand-new machine.

> All paths below are relative to the **repo root** unless stated otherwise.
> Tested for Ubuntu/Debian; adapt the `apt` lines for other distros.

---

## 0. What you are about to run

The evaluation resynthesizes ~60 s of each source texture, then compares the
embedding distributions of originals vs generations with three distances
(**lower is better**): **FAD-VGGish**, **KAD-CLAP**, **KAD-TexStat**. It runs in
**two phases that only share WAV files on disk**, because they need
**incompatible Python environments**:

| Phase        | Environment                                   | Python | Does                              |
| ------------ | --------------------------------------------- | ------ | -------------------------------- |
| **generate** | project venv (`torch>=2.11`, SCAPES, msclap)  | 3.12   | GPU-heavy resynthesis → writes WAVs |
| **score**    | isolated venv (`kadtk`, pins `torch<2.6`)     | 3.11   | reads WAVs → writes the Excel    |

You will therefore build **two** virtual environments. `uv` manages both.

Final output: `app/backend/inference/evaluation/out/eval_results.xlsx`.

---

## 1. System prerequisites

Install base tooling (git, a compiler toolchain, curl, and audio libs that
`librosa`/`soundfile` rely on):

```bash
sudo apt-get update
sudo apt-get install -y \
  git curl ca-certificates build-essential \
  libsndfile1 ffmpeg
```

* `git` — clone the repo + submodules.
* `build-essential` — some wheels compile native code.
* `libsndfile1` — backing library for `soundfile` (reading/writing WAVs).
* `ffmpeg` — decoding fallback for `librosa.load`.

### Install `uv`

`uv` provisions the right Python versions and both venvs without you installing
Python system-wide:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# load it into the current shell (or open a new terminal)
source "$HOME/.local/bin/env"
uv --version
```

`uv` will fetch Python **3.12** (generate venv) and **3.11** (scoring venv)
on demand — you do **not** need a system Python.

---

## 2. NVIDIA driver + CUDA (strongly recommended)

The **generate** phase is GPU-heavy. CPU works but is *very* slow for the full
dataset. A GPU only helps if the NVIDIA **driver** is installed; the CUDA
**runtime** ships inside the PyTorch wheels, so you do **not** need a separate
CUDA toolkit install.

1. Check whether a GPU + driver are already present:

   ```bash
   nvidia-smi
   ```

   If this prints your GPU name and a driver/CUDA version, skip to step 3.

2. If `nvidia-smi` is missing, install the driver and **reboot**:

   ```bash
   sudo apt-get install -y nvidia-driver-535   # or the version your distro recommends
   sudo reboot
   ```

   After reboot, confirm `nvidia-smi` works.

> CUDA does **not** work on machines without an NVIDIA GPU (no AMD, no Apple
> Silicon). On those, run the generate phase on CPU (it still produces correct
> results, just slowly) — pass `--device cpu` where relevant and expect long
> runtimes.

CUDA is verified in Python in **step 6** once the project venv exists.

---

## 3. Clone the repository (with submodules)

The scoring phase (KAD-TexStat) needs the `ddsp_textures` submodule, and SCAPES
itself is a submodule. Clone everything:

```bash
git clone --recurse-submodules \
  https://github.com/<your-org>/S103-Interface-for-Generative-Audio-Latent-Interpolation.git
cd S103-Interface-for-Generative-Audio-Latent-Interpolation
```

If you already cloned without `--recurse-submodules`, initialise the two
submodules the evaluation needs:

```bash
git submodule update --init modules/scapes modules/ddsp_textures
```

### Clone `torch_filterbanks` manually

`ddsp_textures` references `torch_filterbanks` as a **bare gitlink** with no
`.gitmodules` entry, so `git submodule` won't fetch it. Clone it into place:

```bash
git clone https://github.com/cordutie/torch_filterbanks.git \
  modules/ddsp_textures/experiments/texstat/torch_filterbanks
```

---

## 4. ⚠️ Obtain the model weights and the dataset (NOT in git)

These are **gitignored** and will be **absent in a fresh clone** — the
evaluation cannot run without them. You must copy them from a teammate, a shared
drive, or wherever your group stores them.

### 4a. Model checkpoints (required for `generate`)

The generate phase loads a ~887 MB SCAPES flow model from:

```
app/backend/inference/models/Full_150e/checkpoints/
├── best_flow_model.pt          (~380 MB)
├── best_local_encoder.pt       (~4 MB)
├── flow_model_config.json
└── local_encoder_config.json
```

Create the folder and copy the four files in:

```bash
mkdir -p app/backend/inference/models/Full_150e/checkpoints
# then copy best_flow_model.pt, best_local_encoder.pt, and the two .json files here
```

> You can override these paths with the `SCAPES_FLOW_MODEL_CKPT`,
> `SCAPES_FLOW_MODEL_CONFIG`, `SCAPES_LOCAL_ENCODER_CKPT`, and
> `SCAPES_LOCAL_ENCODER_CONFIG` environment variables if your weights live
> elsewhere (see `app/backend/inference/constants.py`).

### 4b. Source dataset WAVs (required for `generate`)

By default the dataset is **every `*.wav`** in
`app/backend/inference/assets/` (the original project has ~25 textures there).
These WAVs are gitignored, so copy them in:

```bash
# place the source .wav files here:
ls app/backend/inference/assets/*.wav   # should list your textures
```

Alternatively, point the script at any other folder of source WAVs at run time
with `--data-dir <dir>` (see step 7).

---

## 5. Build the **project** venv (generate phase)

This env uses the `pyproject.toml` in `app/backend` (Python 3.12, `torch>=2.11`,
SCAPES, `msclap` from GitHub). Let `uv` create and populate `app/backend/.venv`:

```bash
cd app/backend
uv sync
cd ../..
```

`uv sync` reads `app/backend/pyproject.toml` (including the `msclap` git source)
and resolves a CUDA-enabled `torch` build automatically on Linux with an NVIDIA
GPU. You don't activate this venv manually — you run it via `uv run` (step 7).

> If `uv sync` errors on your setup, the fallback is the requirements file:
> `uv pip install -r app/backend/requirements.txt`.

---

## 6. Verify CUDA in the project venv

Confirm PyTorch sees the GPU **inside the project venv** (run from `app/backend`):

```bash
cd app/backend
uv run python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
cd ../..
```

Expected on a GPU box: `CUDA available: True` and your GPU name. If it prints
`CPU only` but `nvidia-smi` works, your `torch` wheel is the CPU build — reinstall
a CUDA build, e.g.:

```bash
cd app/backend
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
cd ../..
```

---

## 7. Build the **scoring** venv (score phase)

`kadtk` pins `torch<2.6` and `python<3.12`, so it **must not** share the project
env. Build a separate Python 3.11 venv at the repo root:

```bash
uv venv --python 3.11 .venv-eval
uv pip install --python .venv-eval \
  -r app/backend/inference/evaluation/requirements-scoring.txt
```

> `torchaudio` must match the `torch` version `kadtk` resolves (e.g. torch
> 2.5.x → torchaudio 2.5.x). If the install picks an incompatible build, pin it:
> `uv pip install --python .venv-eval "torchaudio==2.5.*"`.

The scoring venv downloads model weights (VGGish, CLAP) on first use — that's
expected.

---

## 8. Smoke test (single sound)

Runs the whole pipeline end-to-end on **one** sound to catch setup problems fast.
**Both phase commands must run with `CWD = app/backend`.**

```bash
# PHASE 1 — generate, in the PROJECT venv (uv run)
(cd app/backend && uv run python -m inference.evaluation.evaluate_model --phase generate --limit 1)

# PHASE 2 — score, in the SCORING venv
source .venv-eval/bin/activate
(cd app/backend && python -m inference.evaluation.evaluate_model --phase score)
deactivate
```

**Confirm success:**

* `app/backend/inference/evaluation/out/reference/` and `.../generated/` each
  contain **one** ~60 s WAV.
* `app/backend/inference/evaluation/out/eval_results.xlsx` opens with **finite**
  FAD-VGGish / KAD-CLAP / KAD-TexStat values.

---

## 9. Full benchmark (complete dataset)

Once the smoke test passes, run the same two phases **without `--limit`**:

```bash
# PHASE 1 — generate all sounds (PROJECT venv) — long, GPU recommended
(cd app/backend && uv run python -m inference.evaluation.evaluate_model --phase generate)

# PHASE 2 — score (SCORING venv)
source .venv-eval/bin/activate
(cd app/backend && python -m inference.evaluation.evaluate_model --phase score)
deactivate
```

The Excel has one row per sound plus a pooled **OVERALL** row — the pooled row is
the headline number. Scores reflect the **full** dataset (no cherry-picking).

Because the phases are decoupled through the WAVs on disk, you can re-run
`--phase score` as often as you like **without** regenerating audio.

### Useful flags

| Flag                 | Meaning                                        |
| -------------------- | ---------------------------------------------- |
| `--data-dir <dir>`   | evaluate a different folder of source WAVs     |
| `--duration <sec>`   | seconds to generate / trim (default 60)        |
| `--nfe <n>`          | ODE solver steps for generation (default 32)   |
| `--seed <n>`         | torch seed for reproducible generation         |
| `--only <stem>`      | evaluate a single sound by file stem           |
| `--limit <n>`        | evaluate at most N sounds (smoke test)         |
| `--device cpu\|cuda` | scoring device for kadtk (default: auto)       |

Run `uv run python -m inference.evaluation.evaluate_model --help` for the full list.

---

## 10. Troubleshooting

| Symptom | Cause / fix |
| ------- | ----------- |
| `data-dir does not exist` / `no .wav files found` | Dataset missing — copy WAVs into `app/backend/inference/assets/` or pass `--data-dir` (step 4b). |
| Generate fails loading the flow model / file-not-found on a `.pt` | Checkpoints missing — copy them into `models/Full_150e/checkpoints/` (step 4a). |
| `TexStat sources not found` | `ddsp_textures` / `torch_filterbanks` not initialised — redo the submodule + manual clone (step 3). |
| `Scoring needs kadtk` | You're not in the `.venv-eval` scoring venv, or it wasn't installed — redo step 7 and `source .venv-eval/bin/activate`. |
| `missing WAVs in .../reference` | You ran `--phase score` before `--phase generate` — run generate first. |
| `CUDA available: False` but `nvidia-smi` works | CPU-only torch wheel installed — reinstall a CUDA build (step 6). |
| torchaudio / torch version mismatch in scoring | Pin torchaudio to kadtk's resolved torch: `uv pip install --python .venv-eval "torchaudio==2.5.*"`. |
| `libsndfile`/audio load errors | Install system audio libs: `sudo apt-get install -y libsndfile1 ffmpeg` (step 1). |

---

## Quick reference (the whole thing, condensed)

```bash
# 1. system deps + uv
sudo apt-get update && sudo apt-get install -y git curl build-essential libsndfile1 ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh && source "$HOME/.local/bin/env"

# 2. clone + submodules + torch_filterbanks
git clone --recurse-submodules <REPO_URL> && cd S103-Interface-for-Generative-Audio-Latent-Interpolation
git submodule update --init modules/scapes modules/ddsp_textures
git clone https://github.com/cordutie/torch_filterbanks.git \
  modules/ddsp_textures/experiments/texstat/torch_filterbanks

# 3. >>> copy model checkpoints into app/backend/inference/models/Full_150e/checkpoints/
#    >>> copy source WAVs into app/backend/inference/assets/   (both are gitignored!)

# 4. project venv (generate)
(cd app/backend && uv sync)

# 5. scoring venv (score)
uv venv --python 3.11 .venv-eval
uv pip install --python .venv-eval -r app/backend/inference/evaluation/requirements-scoring.txt

# 6. smoke test
(cd app/backend && uv run python -m inference.evaluation.evaluate_model --phase generate --limit 1)
source .venv-eval/bin/activate
(cd app/backend && python -m inference.evaluation.evaluate_model --phase score)
deactivate
```
