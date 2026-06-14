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

You will therefore build **two** virtual environments with the standard-library
`venv` module — one Python 3.12 (`.venv`, the project env) and one Python 3.11
(`.venv-eval`, the scoring env).

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

### Install Python 3.11 and 3.12

You need **both** interpreters (3.12 for the project venv, 3.11 for scoring),
each with its `venv` module. On Ubuntu the `deadsnakes` PPA provides them:

```bash
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv python3.11 python3.11-venv
```

Confirm both are available:

```bash
python3.12 --version
python3.11 --version
```

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

   If this prints your GPU name and a driver/CUDA version, skip to step 3. Note
   the **CUDA version** it reports — you'll match the torch wheel to it in step 5.

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

The **model checkpoints** are **gitignored** and will be **absent in a fresh
clone** — the evaluation cannot run without them. The **library WAVs** (under
`app/backend/inference/assets/short/` and `.../long/`) *are* committed, so a
fresh clone already has them; you only need to copy the checkpoints.

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

### 4b. Source dataset WAVs

The library textures live in `app/backend/inference/assets/short/` and
`app/backend/inference/assets/long/` and are committed to the repo. To evaluate a
**different** folder of source WAVs, point the script at it at run time with
`--data-dir <dir>` (see step 9).

---

## 5. Build the **project** venv (generate phase)

Create the project env at the **repo root** with Python 3.12 and install the
pinned dependencies from `requirements.txt` (which already bundles SCAPES,
`msclap`-from-git, and the backend deps):

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

> **Match the torch CUDA build to your driver.** `requirements.txt` pins the
> **cu128** torch wheels (for a CUDA 12.8-capable driver). If your `nvidia-smi`
> from step 2 reports an older CUDA, edit the two `torch`/`torchaudio` lines and
> the `--extra-index-url` at the top of `requirements.txt` to the matching build
> (e.g. `+cu121` with `https://download.pytorch.org/whl/cu121`), per the comment
> in the file. For CPU-only boxes, use the plain `==2.11.0` versions.

You run this env via its python directly (`.venv/bin/python …`) or by
`source .venv/bin/activate`.

---

## 6. Verify CUDA in the project venv

Confirm PyTorch sees the GPU **inside the project venv** (run from `app/backend`):

```bash
cd app/backend
../../.venv/bin/python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
cd ../..
```

Expected on a GPU box: `CUDA available: True` and your GPU name. If it prints
`CPU only` but `nvidia-smi` works, your `torch` wheel is the CPU build — reinstall
a CUDA build that matches your driver, e.g.:

```bash
.venv/bin/python -m pip install --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 7. Build the **scoring** venv (score phase)

`kadtk` pins `torch<2.6` and `python<3.12`, so it **must not** share the project
env. Build a separate Python 3.11 venv at the repo root:

```bash
python3.11 -m venv .venv-eval
.venv-eval/bin/python -m pip install \
  -r app/backend/inference/evaluation/requirements-scoring.txt
```

> `torchaudio` must match the `torch` version `kadtk` resolves (e.g. torch
> 2.5.x → torchaudio 2.5.x). If the install picks an incompatible build, pin it:
> `.venv-eval/bin/python -m pip install "torchaudio==2.5.*"`.

The scoring venv downloads model weights (VGGish, CLAP) on first use — that's
expected.

---

## 8. Smoke test (single sound)

Runs the whole pipeline end-to-end on **one** sound to catch setup problems fast.
**Both phase commands must run with `CWD = app/backend`.**

```bash
# PHASE 1 — generate, in the PROJECT venv
(cd app/backend && ../../.venv/bin/python -m inference.evaluation.evaluate_model --phase generate --limit 1)

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
(cd app/backend && ../../.venv/bin/python -m inference.evaluation.evaluate_model --phase generate)

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

Run `../../.venv/bin/python -m inference.evaluation.evaluate_model --help` (from
`app/backend`) for the full list.

---

## 10. Troubleshooting

| Symptom | Cause / fix |
| ------- | ----------- |
| `data-dir does not exist` / `no .wav files found` | Pass an existing `--data-dir`, or run from the repo so the committed `assets/short/` + `assets/long/` are present (step 4b). |
| Generate fails loading the flow model / file-not-found on a `.pt` | Checkpoints missing — copy them into `models/Full_150e/checkpoints/` (step 4a). |
| `TexStat sources not found` | `ddsp_textures` / `torch_filterbanks` not initialised — redo the submodule + manual clone (step 3). |
| `Scoring needs kadtk` | You're not in the `.venv-eval` scoring venv, or it wasn't installed — redo step 7 and `source .venv-eval/bin/activate`. |
| `missing WAVs in .../reference` | You ran `--phase score` before `--phase generate` — run generate first. |
| `CUDA available: False` but `nvidia-smi` works | CPU-only torch wheel installed — reinstall a CUDA build matching your driver (step 6). |
| torchaudio / torch version mismatch in scoring | Pin torchaudio to kadtk's resolved torch: `.venv-eval/bin/python -m pip install "torchaudio==2.5.*"`. |
| `libsndfile`/audio load errors | Install system audio libs: `sudo apt-get install -y libsndfile1 ffmpeg` (step 1). |

---

## Quick reference (the whole thing, condensed)

```bash
# 1. system deps + python 3.11/3.12
sudo apt-get update && sudo apt-get install -y git curl build-essential libsndfile1 ffmpeg
sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv python3.11 python3.11-venv

# 2. clone + submodules + torch_filterbanks
git clone --recurse-submodules <REPO_URL> && cd S103-Interface-for-Generative-Audio-Latent-Interpolation
git submodule update --init modules/scapes modules/ddsp_textures
git clone https://github.com/cordutie/torch_filterbanks.git \
  modules/ddsp_textures/experiments/texstat/torch_filterbanks

# 3. >>> copy model checkpoints into app/backend/inference/models/Full_150e/checkpoints/
#    (library WAVs under assets/short|long are already committed)

# 4. project venv (generate) — adjust the torch CUDA build in requirements.txt to your driver
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# 5. scoring venv (score)
python3.11 -m venv .venv-eval
.venv-eval/bin/python -m pip install -r app/backend/inference/evaluation/requirements-scoring.txt

# 6. smoke test
(cd app/backend && ../../.venv/bin/python -m inference.evaluation.evaluate_model --phase generate --limit 1)
source .venv-eval/bin/activate
(cd app/backend && python -m inference.evaluation.evaluate_model --phase score)
deactivate
```
