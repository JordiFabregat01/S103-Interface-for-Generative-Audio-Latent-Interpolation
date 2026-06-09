# Running the Model Evaluation Benchmarks

This document is a self-contained guide for running the SCAPES model evaluation.
You can hand it to whoever runs the benchmarks (e.g. on a machine with a GPU).

## What it does

For every source sound, the script resynthesizes ~60 s of the *same* texture,
then measures how close the **generated** audio's embedding distribution is to
the **original** one. Three audio-distribution distances are reported
(**lower is better**):

| Metric          | Embedding   | Distance                  |
| --------------- | ----------- | ------------------------- |
| **FAD-VGGish**  | VGGish      | Fréchet Audio Distance    |
| **KAD-CLAP**    | CLAP-2023   | Kernel (MMD) distance     |
| **KAD-TexStat** | TexStat     | Kernel (MMD) distance     |

Output: `app/backend/inference/evaluation/out/eval_results.xlsx` — one row per
sound plus a pooled **OVERALL** row. The pooled row is the headline number.
Scores reflect the **full** dataset (no cherry-picking).

## Why there are two phases

The evaluation runs in **two phases that talk only through WAV files on disk**,
because they need **different, incompatible environments**:

| Phase        | Environment                                   | Does                                  |
| ------------ | --------------------------------------------- | ------------------------------------- |
| **generate** | project venv (torch 2.11 / Py 3.12 / SCAPES)  | GPU-heavy resynthesis → writes WAVs   |
| **score**    | separate Py 3.11 venv (`kadtk`)               | reads WAVs → writes the Excel         |

`kadtk` pins `torch<2.6` and `python<3.12`, so it **cannot** share the project
environment — hence the dedicated scoring venv.

> ⚠️ Both phase commands must run with **`CWD = app/backend`**.

## Requirements

* The project backend installed and working (see the main `README.md`).
* `uv` installed (`pip install uv`).
* **GPU strongly recommended** for the generate phase.
* Git access to the submodules (`scapes`, `ddsp_textures`, `torch_filterbanks`).

---

## 1. One-time setup

Run from the **repo root**.

### Submodules (needed for KAD-TexStat)

```bash
git submodule update --init modules/scapes modules/ddsp_textures
# ddsp_textures references torch_filterbanks as a bare gitlink (no upstream
# .gitmodules entry), so clone it explicitly into place:
git clone https://github.com/cordutie/torch_filterbanks.git \
  modules/ddsp_textures/experiments/texstat/torch_filterbanks
```

### Isolated scoring venv

**macOS / Linux (bash):**

```bash
uv venv --python 3.11 .venv-eval
uv pip install --python .venv-eval -r app/backend/inference/evaluation/requirements-scoring.txt
```

**Windows (PowerShell):**

```powershell
uv venv --python 3.11 .venv-eval
uv pip install --python .venv-eval -r app/backend/inference/evaluation/requirements-scoring.txt
```

> ℹ️ `torchaudio` must match the `torch` version `kadtk` resolves (e.g. torch
> 2.5.x → torchaudio 2.5.x). If the install picks an incompatible build, run
> `uv pip install --python .venv-eval "torchaudio==2.5.*"` afterwards.

---

## 2. Smoke test (single sound)

Quickly checks the whole pipeline end-to-end on one sound (~minutes).

**macOS / Linux (bash):**

```bash
# PHASE 1 — generate (PROJECT venv)
(cd app/backend && uv run python -m inference.evaluation.evaluate_model --phase generate --limit 1)

# PHASE 2 — score (SCORING venv)
source .venv-eval/bin/activate
(cd app/backend && python -m inference.evaluation.evaluate_model --phase score)
deactivate
```

**Windows (PowerShell):**

```powershell
# PHASE 1 — generate (PROJECT venv)
cd app/backend
uv run python -m inference.evaluation.evaluate_model --phase generate --limit 1
cd ../..

# PHASE 2 — score (SCORING venv)
.venv-eval\Scripts\Activate.ps1
cd app/backend
python -m inference.evaluation.evaluate_model --phase score
cd ../.. ; deactivate
```

**Confirm:** `app/backend/inference/evaluation/out/reference/` and `.../generated/`
each contain one ~60 s WAV, and `.../out/eval_results.xlsx` opens with finite
**FAD-VGGish / KAD-CLAP / KAD-TexStat** values.

---

## 3. Complete test (full dataset)

Once the smoke test passes, run the same two phases **without `--limit`**.

**macOS / Linux (bash):**

```bash
(cd app/backend && uv run python -m inference.evaluation.evaluate_model --phase generate)
source .venv-eval/bin/activate
(cd app/backend && python -m inference.evaluation.evaluate_model --phase score)
deactivate
```

**Windows (PowerShell):**

```powershell
cd app/backend
uv run python -m inference.evaluation.evaluate_model --phase generate
cd ../..

.venv-eval\Scripts\Activate.ps1
cd app/backend
python -m inference.evaluation.evaluate_model --phase score
cd ../.. ; deactivate
```

The Excel (`app/backend/inference/evaluation/out/eval_results.xlsx`) has one row
per sound plus the pooled **OVERALL** row.

---

## Useful flags

| Flag                | Meaning                                         |
| ------------------- | ----------------------------------------------- |
| `--data-dir <dir>`  | evaluate a different folder of source WAVs       |
| `--duration <sec>`  | seconds to generate / trim (default 60)          |
| `--nfe <n>`         | ODE solver steps for generation (default 32)     |
| `--seed <n>`        | torch seed for reproducible generation           |
| `--only <stem>`     | evaluate a single sound by file stem             |
| `--limit <n>`       | evaluate at most N sounds (smoke test)           |
| `--device cpu\|cuda` | scoring device for kadtk (default: auto-detect)  |

Run `python -m inference.evaluation.evaluate_model --help` for the full list.

---

## What to expect on the first run

* The scoring venv **downloads model weights** (VGGish, CLAP) on first use.
* The `generate` phase loads the ~887 MB SCAPES flow model and resynthesizes
  ~60 s per sound, so the full dataset takes a while (GPU strongly recommended).
* The two phases are decoupled through the WAV files on disk, so `--phase score`
  can be re-run as often as you like **without regenerating audio**.

## Troubleshooting

* **`TexStat sources not found`** → submodules/`torch_filterbanks` weren't
  initialised; re-run the step 1 submodule + clone commands.
* **`Scoring needs kadtk`** → you're not in the `.venv-eval` scoring venv, or it
  wasn't installed; redo step 1's isolated-venv install.
* **`missing WAVs in .../reference`** → run `--phase generate` first.
* **torchaudio / torch version mismatch in scoring** → pin torchaudio to kadtk's
  resolved torch (see the note in step 1).
