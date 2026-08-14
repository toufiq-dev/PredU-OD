# PredU-OD — Predicting Synthetic-Data Utility for Object Detection

**One-sentence thesis:** Can we predict which diffusion-synthesized training examples will
improve an object detector *before* retraining it, and does structured knowledge of the
detector's failures provide predictive information beyond generic example difficulty and
distributional imbalance?

This repository contains the **Week 1–2 go/no-go pilot** for the PredU-OD protocol
(full protocol: `docs/PREDU_OD_PREREGISTRATION.md`).

## Pilot objectives (5 measurements)

1. **GLIGEN at 512 px on Colab** — throughput, artifact rate, and **box fidelity μ_Q**
   (do generated objects actually sit in their conditioning boxes?).
2. **YOLO11s on a 10% COCO subset** — baseline mAP + failure-profile extraction
   (FN/FP/confusion/localization/uncertainty) and EL2N-style difficulty on the training set;
   **corr(F, D)** sanity number.
3. **Probe signal** — 8–12 probe subsets (|S| = 250): spread of ΔmAP across subsets vs.
   measurement noise (2-seed duplicates) → signal-to-noise ratio. This is the make-or-break
   measurement.
4. **DMR construction** — can a difficulty-matched random pool be built that matches
   class/scale/count distributions while differing only in failure-conditioned context?
5. **Calibration** — measured GPU-hours per probe train → sets `n_probes` for the full run
   (64–80 subsets under the $100 ceiling).

## Layout

```
src/preduod/
  geometry.py            IoU, greedy matching            (pure stdlib)
  rank_metrics.py        Kendall-τ, Spearman, nDCG@K     (pure stdlib)
  stats.py               entropy, pearson, spearman, VIF, bootstrap CI (pure stdlib)
  generation/
    layouts.py           neutral-pool layout sampling    (pure stdlib)
    gligen_pipeline.py   diffusers GLIGEN wrapper         (lazy torch/diffusers)
    box_fidelity.py      μ_Q computation                  (pure core + lazy detector)
  detection/
    coco.py              build 10% subset, COCO→YOLO     (pure stdlib + json)
    train_yolo.py        ultralytics wrapper              (lazy)
    infer.py             batch inference → metadata JSON  (lazy)
  failure/
    failure_profile.py   FN/FP/confusion/loc/uncertainty  (pure stdlib)
    difficulty.py        EL2N-style early-training difficulty (pure stdlib)
    redundancy.py        corr(F,D), VIF, ablation         (pure stdlib)
  probes/
    sample_subsets.py    stratified + DMR sampling        (pure stdlib)
    train_probe.py       short-schedule probe training + GPU-hour timing (lazy)
    utility.py           ΔmAP utilities                   (pure stdlib)
  pilot/report.py        probe spread, SNR, prelim τ      (pure stdlib)
scripts/
  01_build_coco_subset.py
  02_generate_pool.py
  03_train_baseline.py
  04_failure_profile.py
  05_probe_subsets.py
  06_pilot_report.py     calibration report (GPU-hrs, spread, corr(F,D), τ)
tests/                   dependency-free test runner (python3 tests/run_all.py)
configs/pilot.json       all knobs + go/no-go thresholds
```

## Setup (Colab Pro)

```bash
pip install -r requirements.txt
# download COCO train2017 + instances_train2017.json (10% subset source; val2017 stays untouched as the eval set)
```

Run the steps in order:

```bash
python scripts/01_build_coco_subset.py --ann instances_train2017.json --out data/coco10 --fraction 0.10
python scripts/02_generate_pool.py          --out data/pool --n 300
python scripts/03_train_baseline.py         --data data/coco10 --out runs/baseline
python scripts/04_failure_profile.py        --model runs/baseline/weights/best.pt \
                                            --data data/coco10 --out results/failure_profile.json
python scripts/05_probe_subsets.py          --pool data/pool --data data/coco10 --out results/probes
python scripts/06_pilot_report.py           --probes results/probes/probe_results.json \
                                            --failure results/failure_profile.json \
                                            --config configs/pilot.json
```

## Go / No-Go thresholds (from `configs/pilot.json`)

| Signal | Threshold |
|---|---|
| GLIGEN box fidelity μ_Q | ≥ 0.55 mean IoU vs conditioning boxes |
| Probe ΔmAP spread | ≥ 2× measurement noise (SNR ≥ 1.0) |
| corr(F, D) | report; if ≥ 0.9, failure-vs-difficulty story weakens |
| Preliminary Kendall-τ (best univariate feature) | ≥ 0.3 |
| GPU-hours per probe train | ≤ 0.6 h (else shrink probes, not quality) |

A **no-go** on the probe signal means: reduce scope (classification-domain pilot) or
restructure the probe protocol before committing to the full thesis.
