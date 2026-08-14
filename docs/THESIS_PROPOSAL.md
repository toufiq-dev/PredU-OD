# PredU-OD: Predicting Synthetic-Data Utility for Object Detection

## The Role of Detector Failure Structure

**Version 5.0 — locked research question · August 2026**

**Status:** MSc thesis proposal, ready for supervisor review. The experimental protocol is
pre-registered separately in `docs/PREDU_OD_PREREGISTRATION.md`; the Week 1–2 go/no-go
pilot lives in this repository (`configs/pilot.json`, `scripts/`).

---

## 0. One-sentence thesis

> **Can we predict which diffusion-synthesized training examples will improve an object
> detector *before* retraining it, and does structured knowledge of the detector's failures
> provide predictive information beyond generic example difficulty and distributional
> imbalance?**

---

## 1. Abstract

Diffusion-synthesized training data demonstrably improves object detection in data-sparse
regimes: InstaGen (CVPR 2024) reports +4.5 AP in open-vocabulary detection and +1.2–5.2 AP
across data-sparse detection settings (23.3 → 28.5 AP with only 10% of COCO training data),
and TADA (ICLR 2026) shows that
targeted augmentation of *slow-learnable* examples reduces how much synthetic data is
needed. What the field has not established is a *predictive* account: given a detector and a
pool of candidate synthetic images, which subset will actually improve the detector — and
does the *structure* of the detector's errors carry information that generic difficulty and
rebalancing do not?

This thesis proposes **PredU-OD**, a probe-based utility-prediction framework for synthetic
detection data. The method: train a small detector on a limited real set, generate a
*neutral* box-conditioned candidate pool (GLIGEN, CVPR 2023; annotations known by
construction), sample probe subsets, measure their utility by retraining, and learn a sparse
subset-level utility predictor from **8 features** — FN rate, FP rate, localization error,
early-training difficulty, class/scale entropy, CLIP diversity, and generator fidelity —
validated by rank correlation (Kendall-τ, Spearman, nDCG) under **pool-level
cross-validation** (train on pools G₁…G_{K−1}, test on an unseen pool G_K) with explicit
leakage controls. The mechanism: a nested four-model analysis quantifying whether failure
features add predictive value beyond rarity, class, scale, difficulty, quality, and
diversity (bootstrapped ΔR²_CV), together with a redundancy audit (corr(F,D), VIF, feature
ablation). The application: greedy, redundancy-aware selection evaluated on formally defined
synthetic-data efficiency curves under both image and GPU-hour budgets, benchmarked against
random, class-balanced, difficulty-matched-random, and a documented TADA reimplementation.

All experiments run under a **USD 100 hard compute ceiling** (Colab; GLIGEN inference,
YOLO11s on a 10% COCO regime). A rigorously executed null result — failure information
reduces to difficulty and rebalancing — is itself a publishable finding, because it would
show that apparent gains from failure-guided synthesis are explained by distributional
rebalancing. The contribution is not another generator: it is a validated,
detection-specific account of *which properties of synthetic data predict downstream
utility, whether detector failures explain utility beyond difficulty, and how much synthetic
data to generate under a fixed budget.*

---

## 2. Introduction

### 2.1 The data bottleneck in object detection

Object detection performance is bounded by training data: its quantity, diversity, and
representativeness. Modern detectors (YOLO-family, RT-DETR) reach excellent accuracy on
large benchmarks, but real-world deployment routinely exposes them to regimes that are
poorly represented in training data — small objects, heavy occlusion, unusual viewpoints and
lighting, crowded scenes, visually confusable classes, rare classes, rare object/context
combinations, background-induced false positives, and domain shift between training and
deployment. The conventional remedy is to collect and manually annotate more real images;
this is expensive, slow, and does not scale to long-tail regimes.

### 2.2 The synthetic-data promise

Generative AI offers an alternative. Latent diffusion models (Rombach et al., CVPR 2022)
made high-quality synthesis computationally tractable; ControlNet (Zhang et al., ICCV 2023)
added spatial conditioning (edges, depth, segmentation, pose); GLIGEN (Li et al., CVPR 2023)
added open-set *bounding-box* conditioning to a frozen text-to-image model, which is
exactly the representation an object detector consumes. A substantial body of work now
shows this is not hypothetical:

- **DatasetDM** (NeurIPS 2023) established the "diffusion as dataset generator" paradigm,
  producing images *with* perception annotations for segmentation, instance segmentation,
  depth, pose, and long-tail perception.
- **InstaGen** (CVPR 2024) turned a diffusion model into a detection-data synthesizer and
  reported +4.5 AP in open-vocabulary detection and +1.2–5.2 AP across data-sparse
  detection settings, including +5.2 AP (23.3 → 28.5) with only 10% of COCO training data.
- **ODGEN** (IEEE TPAMI 2026) and **AeroGen** (CVPR 2025) demonstrated box/layout-conditioned
  synthesis for general and remote-sensing detection with substantial downstream gains.
- **TADA** (ICLR 2026) showed that augmenting only *slow-learnable* examples (identified
  early in training) can match or beat augmenting the whole dataset, reporting promising
  results on object detection benchmarks in addition to classification.

### 2.3 The open question

These results establish that synthetic images *can* be useful. But they also expose the
central problem of this thesis:

> **Generating more synthetic images is not the same as generating more useful training
> data.**

A detector may benefit enormously from 500 carefully selected synthetic images and little —
or negatively — from 10,000 poorly chosen ones. Existing pipelines are almost all
open-loop: generate according to prompts/layouts/randomization, add everything, retrain.
The handful of closed-loop methods either operate on classification (Longtail-Guided
Diffusion, ICML 2025), on industrial domain-randomization settings (Synthetic Active
Learning, CAiSE 2025), or on generic early-training difficulty rather than the *structure*
of a detector's errors (TADA, ICLR 2026). Two 2026 studies probe the prediction question
for detection directly: Marian et al. (arXiv:2602.18525) ask whether *pre-training
generative metrics* (FID/IS in Inception-v3 and DINOv2 spaces; object-centric bbox-statistic
distances) predict downstream YOLO11 mAP across six generators and seven augmentation
ratios — i.e., **which generator and how much** to use — and find the signal
regime-dependent; and the one systematic study of *what makes synthetic data effective* —
SENSE (ICML 2026) — targets image **segmentation**, where the
utility factors (dense scene composition, fine instance fidelity) are not the same as those
of detection (bounding-box fidelity, scale, localization, occlusion, false-positive
context).

The thesis therefore asks a question that no verified prior work has asked for object
detection at the **subset-selection** level:

> **Can the downstream utility of a *subset* of a single generated pool be predicted
> *before* retraining — from detector-failure structure rather than global generative
> metrics — and does structured detector-failure information improve that prediction
> beyond generic difficulty, rarity, and rebalancing?**

### 2.4 Thesis statement

We will develop and validate **PredU-OD**: a framework that (i) measures the utility of
synthetic subsets by actually retraining on probe subsets, (ii) learns a sparse predictor of
subset utility from cheap pre-training features, (iii) isolates whether detector-specific
failure structure carries *incremental* predictive information beyond difficulty and
rebalancing, and (iv) uses the predictor to allocate a fixed synthetic-data budget. The
outcome is a controlled, detection-specific, empirically validated account of *when
synthetic data helps, whether failures explain it, and how much to generate* — plus, if the
failure signal survives the controls, evidence that a detector can tell us what synthetic
data it needs.

---

## 3. Background and Related Work

### 3.1 Diffusion-based synthetic data for object detection (generation-centric)

- **DatasetDM** (Wu et al., NeurIPS 2023): diffusion models as dataset generators with
  perception annotations; established feasibility across several perception tasks. This
  thesis goes one step further: from *dataset generation* to *dataset generation conditioned
  on measured model need*.
- **InstaGen** (Feng et al., CVPR 2024; arXiv:2402.05937): the canonical "diffusion →
  detection data" work; box-conditioned synthesis, open-vocabulary gains of +4.5 AP and
  data-sparse gains of +1.2–5.2 AP (including +5.2 AP at 10% COCO training data).
  Establishes the premise; does not study selection.
- **GLIGEN** (Li et al., CVPR 2023; arXiv:2301.07093): open-set grounded generation via
  trainable gated attention layers on a frozen text-to-image model; `(text, boxes) → image`.
  This is the **primary generator** for this thesis because the intended annotation is known
  by construction, removing the annotation-error confound.
- **ODGEN** (Zhu et al., IEEE TPAMI 2026): box + object-description conditioning for complex
  multi-object scenes; domain-specific gains. Generation-centric.
- **AeroGen** (Tang et al., CVPR 2025): layout-controllable diffusion for remote-sensing
  detection with explicit generation *and filtering* mechanisms — evidence that quality
  filtering matters, but domain-specific and without a utility model.
- **Background augmentation** (Li et al., ECCV 2024): diffusion-based background/object
  augmentation; background augmentation improves robustness and generalization — evidence
  that a synthetic image's value lives partly in its *context*, which motivates context as
  an explicit variable here.
- **LLM-DiffAug** (Jiang et al., Knowledge-Based Systems 2025): LLM-generated prompts +
  inpainting + bounding-box constraints for few-shot detection. Shows language can diversify
  generation instructions; this thesis deliberately does *not* center an LLM, because the
  claim is that *detector-derived evidence*, not prompt diversity, drives useful generation.
- **DiffusionDet** (Chen et al., ICCV 2023): diffusion *as a detector* (denoising boxes);
  explicitly out of scope — diffusion stays on the data-generation side.

### 3.2 Model-guided and targeted synthetic data

- **Generative Data Mining with Longtail-Guided Diffusion** (Hayden et al., ICML 2025,
  PMLR 267): uses model-derived long-tail signals (epistemic uncertainty) to guide diffusion
  toward rare/difficult examples. **Classification only.** The conceptual predecessor of
  this thesis: it replaces the single classification signal u(x) with a structured detection
  error vector.
- **Synthetic Data and Active Learning for Efficient Object Detection** (Tavakoli Ghinani
  et al., CAiSE 2025 workshops / GenSyn): detects detector weaknesses and generates targeted
  synthetic data; 2–6 pp mAP@50 over static synthetic training in industrial settings.
  Closest existing "failure-guided detection" work — but industrial, domain-randomization
  based, and without a quantitative utility model or budget-aware selection.
- **TADA — "Do We Need All the Synthetic Data? Targeted Image Augmentation via Diffusion
  Models"** (Nguyen et al., ICLR 2026; arXiv:2505.21574): selects *slow-learnable* examples
  via clustering of early-training model outputs (the higher-average-loss cluster), then
  performs faithful diffusion-based augmentation of those examples only. Evaluated on image
  classification and reports promising results on object detection benchmarks (COCO,
  YOLOv5m). **This is a direct baseline for the thesis** — see §9. Its signal is generic
  early-training difficulty; the thesis's claim is that *structured detector failure*
  (FN/FP/localization/confusion/uncertainty) adds predictive value beyond that.
- **Synthetic Designed Experiments for Diagnosing Vision Model Failures** (Sarkar, CVPR
  2026 Workshop, SynData4CV): argues open-loop generation is insufficient and applies
  Design-of-Experiments principles to failure diagnosis. Motivates the controlled-design
  philosophy here; does not propose utility prediction or budget allocation for detection.
- **Uncertainty-Guided Data Curation for 3D Object Detection** (Durasov et al., CVPR 2026
  Workshop): uncertainty-guided curation for 3D detection — evidence the direction is
  active; 3D and single-signal, no failure-vs-difficulty decomposition.
- **Targeted False Positive Synthesis** (Zhou et al., MICCAI 2025): detector-guided
  adversarial diffusion for polyp detection — failure-targeted synthesis in a medical
  niche.

### 3.3 Synthetic-data utility analysis

- **SENSE — "What Makes Synthetic Data Effective in Image Segmentation"** (Zhang et al.,
  ICML 2026; arXiv:2605.19289): the landmark systematic study; finds **dense scene
  composition** and **fine instance fidelity** govern segmentation utility and proposes the
  SENSE unified framework, evaluated across Cityscapes, COCO, and ADE20K. This thesis asks
  the analogous question for detection — but detection introduces variables segmentation
  does not have: bounding-box fidelity, scale strata, localization error, occlusion, and
  false-positive context. Critically, SENSE does not ask whether *model failure structure*
  predicts utility.

### 3.4 Data valuation and data curation (the selection side)

These fields study *selection* of real data; none addresses synthetic detection data:

- **Data valuation.** DVRL (Yoon et al., ICML 2020) learns a data-value estimator jointly
  with training; Data Shapley (Ghorbani & Zou, ICML 2019) values data via set-level marginal
  contributions — the formal statement of "U(A+B) ≠ U(A)+U(B)"; CHG Shapley
  (arXiv:2406.11730) performs data selection with a subset-utility function at
  single-retraining cost (cite as arXiv; not accepted at ICLR 2025).
- **Data curation.** Sorscher et al. (NeurIPS 2022) show data pruning can beat power-law
  scaling; DataComp (NeurIPS 2023) benchmarks curation; Data Filtering Networks (Fang et
  al., ICLR 2024) show filtering-model quality ≠ downstream dataset quality; Goyal et al.
  (CVPR 2024) show curation depends on training compute; JEST (Laurençon et al., NeurIPS
  2024 D&B) shows **joint/batch composition** matters beyond independent example scoring —
  direct motivation for set-level (not image-level) utility and redundancy-aware selection.
- **Early-training importance.** Deep Learning on a Data Diet (Paul et al., NeurIPS 2021)
  introduced EL2N/GraNd scores: early-training signals that identify important examples. We
  adopt the EL2N-style definition for difficulty (with the caveat that its transfer beyond
  pruning settings is not guaranteed — hence validation, not assumption).

### 3.5 Adjacent and recent work (2025–2026 landscape)

- **Marian, Kang & Buddery** (arXiv:2602.18525, 2026): "Do Generative Metrics Predict YOLO
  Performance?" — the closest prior work to RQ1. A controlled study of whether
  *pre-training* dataset metrics (global FID/IS-style scores in Inception-v3 and DINOv2
  feature spaces; object-centric Wasserstein/JS distances over bounding-box statistics)
  predict downstream YOLO11 mAP across six generators, seven augmentation ratios (10–150%),
  and three single-class regimes, with augmentation-controlled (residualized) correlations.
  **Closest to RQ1 at the pool/generator level** — but it predicts *which generator / how
  much data*, from global dataset metrics; it does **not** predict *which subset within one
  pool to select*, does not use detector-failure structure, does not learn a predictor from
  probe retraining, and does not evaluate budget allocation. Must be cited and
differentiated (see §3.6).
- **CMU thesis** (Xiao Fang, MS thesis, Robotics Institute, May 2026): studies synthetic
  diffusion data for detection under domain shifts and detector vulnerabilities.
- **Araya-Martinez et al.** (Springer, 2025): data-centric evaluation of detection
  algorithms on synthetic industrial data.
- **Mütze et al.** (WACV 2026): open-vocabulary detectors vs. generated street-scene content.

These works reinforce that synthetic-data-for-detection is an active area — and therefore
that this thesis must *not* claim "first study of synthetic-data utility for detection."
The defensible gap is narrower and stated in §3.6.

### 3.6 The specific gap

> Current literature establishes several adjacent capabilities: diffusion models generate
> useful detection training data (InstaGen, ODGEN, AeroGen); targeted augmentation based on
> early-learning difficulty reduces the required data (TADA); pre-training *generative*
> metrics predict *which generator/ratio* to use, weakly and regime-dependently (Marian et
al., arXiv:2602.18525); systematic factor analysis exists for segmentation (SENSE); and
data valuation/curation studies subset selection (DVRL, Data Shapley, JEST, Sorscher, DFN,
Goyal). **Unaddressed: whether structured detector-specific failure information provides
> incremental predictive value beyond rarity, class/scale composition, and generic
> early-training difficulty — and whether that information can be folded into a pre-training
> utility estimator that reliably ranks *candidate subsets within a single generated pool*
> (not just generators/ratios) under a fixed data/compute budget.**

---

## 4. Research Questions and Hypotheses

### 4.1 Research questions (hierarchical)

| RQ | Question | Role |
|---|---|---|
| **RQ1** | Can the downstream utility of a synthetic detection subset be predicted *before* retraining, from cheap pre-training features? | **Primary (method)** |
| **RQ2** | Does structured detector-failure information add predictive value *beyond* rarity, difficulty, and distributional rebalancing? | **Mechanism (scientific core)** |
| **RQ3** | Can the predicted utility allocate a fixed synthetic-data budget better than random or difficulty-matched selection? | Application |

### 4.2 Hypotheses

- **H1 (prediction).** A sparse utility predictor trained on probe-subset retraining
  outcomes ranks unseen synthetic subsets better than random (Kendall-τ > 0 on unseen
  generation pools).
- **H2 (failure beyond difficulty).** Failure features add predictive value beyond
  difficulty and distribution: ΔR²_CV(M₁ − M₀) > 0, bootstrapped convincingly positive.
- **H3 (failure beyond rebalancing).** Failure-guided selection outperforms
  difficulty-matched-random selection at equal budgets: AP(Failure) > AP(DMR).
- **H4 (budget allocation).** Utility-aware selection beats random and TADA-style selection
  at equal image *and* GPU budgets (AUC comparison).
- **H5 (transfer, secondary).** The utility-ranking relationship remains positive under one
  detector or dataset shift (YOLO11s → RT-DETR **or** COCO → VisDrone; attempted only if the
  primary gates pass).

**Null positions are explicit.** H2 and H3 may fail: failure information may reduce to
difficulty and rebalancing. That is a publishable result if the controls are rigorous.

---

## 5. Definitions

### 5.1 Utility (the target)

$$U(S) \;=\; M(D_{\text{real}} \cup S;\, \theta_S) \;-\; M(D_{\text{real}};\, \theta_0)$$

where $D_{\text{real}}$ is a fixed limited real training set, $\theta_S$ is the detector
trained on $D_{\text{real}} \cup S$, $\theta_0$ is the detector trained on $D_{\text{real}}$
alone, and $M$ is mAP@[.5:.95] on the **same untouched real validation set**.

> The real validation set is never used for generation, selection, feature fitting, probe
> construction, or any hyperparameter choice. This is a hard rule (see §11.4).

Secondary utility targets: ΔAP_S, ΔAP_M, ΔAP_L, ΔAP_rare, recall, FPR, localization quality.

### 5.2 Difficulty

$$D(x) \;=\; \tfrac{1}{T}\sum_{t=1}^{T} \lVert p_t(x) - y \rVert_2$$

over an early-training window (EL2N-style, Data Diet, NeurIPS 2021). Measured on real
training images during the baseline's early epochs. Called **early-training prediction
difficulty**; we implement the exact EL2N formulation and say so.

### 5.3 Rarity

$$R(c) = 1/N_c$$ per class — a property of the data distribution, distinct from difficulty
(per-example learning signal) and failure (model-relative error).

### 5.4 Failure

For each object instance i, a failure vector

$$f_i = [\,u_i,\; e^{\text{cls}}_i,\; e^{\text{loc}}_i,\; s_i,\; o_i,\; c_i,\; d_i\,]$$

comprising uncertainty, classification error, localization error (1 − IoU for matched
predictions), scale stratum, occlusion proxy, contextual complexity, and scene density;
aggregated into subset-level FN rate, FP rate, confusion counts, localization error, and
confidence entropy. **Severity is computed from pre-training diagnostics only** (frequency ×
error rate × confidence) — never from measured synthetic ΔAP, to avoid target leakage
(§11.4).

### 5.5 Fidelity

$$Q_{\text{box}}(g)$$ = adherence of the generated object to its conditioning box
(re-detection/SAM-based check): IoU of the generated object's mask/box with the conditioning
box, plus a semantic check that the requested class is present. This answers the reviewer
objection "your predictor just selects images where GLIGEN got it right."

### 5.6 Failure alignment (candidate-level)

$$F(g) = \sum_k w_k \cdot \operatorname{sim}(z_g,\, z_{\text{failure},k})$$

where $z_g = [\text{class}, \text{scale}, \text{density}, \text{occlusion-proxy},
\text{context-embedding}]$ and $z_{\text{failure},k}$ is the failure-profile
representation for stratum k. This makes "alignment with the failure distribution" an
implementable quantity.

---

## 6. Feature Set (subset-level, aggregated)

All 8 features are defined at the **subset** level, with explicit aggregation:

| # | Feature | Aggregation |
|---|---------|-------------|
| 1 | μ_FN | mean per-image FN rate over g ∈ S |
| 2 | μ_FP | mean per-image FP rate over g ∈ S |
| 3 | μ_Loc | mean localization error (1 − IoU) over matched predictions in S |
| 4 | μ_EL2N | mean early-training difficulty over g ∈ S |
| 5 | H_class | entropy of S's class distribution |
| 6 | H_scale | entropy of S's scale-stratum distribution |
| 7 | D_CLIP | CLIP-embedding diversity of S (mean pairwise distance / volume proxy) |
| 8 | μ_Q | mean generator fidelity (box + semantic) over g ∈ S |

The predictor input is therefore $X_S = [\mu_{FN}, \mu_{FP}, \mu_{Loc}, \mu_{EL2N},
H_{class}, H_{scale}, D_{CLIP}, \mu_Q]$. **Exactly 8 features, not 25** — chosen to keep
n ≈ 64–80 probe observations viable with regularized linear models.

---

## 7. Model Stack and Statistical Core

### 7.1 Four-model null stack

| Model | Specification | Question |
|-------|---------------|----------|
| M_D | U ~ D | Is generic difficulty enough? |
| M_Dist | U ~ R + C + S | Is rebalancing enough? |
| M_0 | U ~ R + C + S + D + Q + V | Conventional targeting baseline |
| M_1 | U ~ R + C + S + D + Q + V + F | Does structured failure add information? |

**Central statistic:** $\Delta R^2_{CV}(M_1 - M_0)$, bootstrapped (block bootstrap over
subsets within pools), reported with confidence intervals. Wording discipline: we claim
*incremental predictive value*, never *causal contribution* — this is an observational,
model-relative analysis.

### 7.2 Estimators

- **Primary:** elastic net (α tuned by pool-level CV).
- **Secondary:** ridge.
- **Exploratory only:** shallow gradient boosting (RF/GBM explicitly not the main
  predictor at n ≈ 64–80).
- **No neural utility predictor.**

### 7.3 Redundancy audit (mandatory reporting)

Because uncertainty/difficulty/failure can be highly correlated, the thesis reports:
- Pearson/Spearman correlation between every failure feature and difficulty;
- VIF per feature;
- feature ablation (leave-one-out ΔR²_CV);
- incremental R² with bootstrap CIs.

If corr(F, D) ≈ 0.9+, the "new signal" claim is weakened and the thesis says so.

---

## 8. Pre-registered Gates (go/no-go for the paper)

1. **Gate 1 — Prediction:** τ > 0 on *unseen* generation pools (pool-level CV, rotate over
   K = 4 pools).
2. **Gate 2 — Failure increment:** ΔR²_CV(M₁ − M₀) > 0, bootstrapped convincingly positive.
3. **Gate 3 — DMR:** AP(Failure-guided) > AP(Difficulty-matched random) — otherwise the
   result is rebalancing, not failure utility.
4. **Gate 4 — Selection:** AUC_PredU > AUC_DMR, and ideally AUC_PredU > AUC_TADA-reimpl, at
   equal image *and* GPU budgets.
5. **Gate 5 — No-harm:** ΔmAP on full COCO val ≥ 0 while chasing AP_S/rare gains.

Gates are evaluated in order; the thesis's claims are scoped to the gates that pass.

---

## 9. Baselines and Ablations

| Method | Difficulty | Failure | Utility model | Diversity | Purpose |
|--------|:---------:|:-------:|:-------------:|:---------:|---------|
| Real only | — | — | — | — | Reference |
| Random synthetic | ✗ | ✗ | ✗ | ✗ | Basic baseline |
| Class-balanced | ✓ | ✗ | ✗ | ✗ | Rebalancing |
| Difficulty-matched random (DMR) | ✓ | ✗ | ✗ | ✗ | **H2/H3 control** |
| TADA reimplementation | ✓ | ✓* | ✗ | ✗ | Direct literature baseline |
| Failure-guided (no selection) | ✓ | ✓ | ✗ | ✗ | Mechanism, no prediction |
| Utility without failure (U₋F) | ✓ | ✗ | ✓ | ✓ | Ablation |
| Utility with failure (U₊F) | ✓ | ✓ | ✓ | ✓ | Ablation |
| **PredU-OD** | ✓ | ✓ | ✓ | ✓ | Full method |

\* TADA uses early-training slow-learnability (cluster split), which we classify and report
as *difficulty*; its "failure" column is marked with an asterisk precisely because the
thesis must show structured failure adds value over TADA's signal.

**TADA reimplementation** is documented, not assumed: we reproduce (i) early-training
cluster split on the baseline's training set, (ii) faithful noise-swap diffusion
augmentation of the targeted subset, (iii) targeted-subset training, and state exactly which
components of the original we reproduce and which differ (e.g., generator choice).

**Selection algorithm.** Greedy selection with a redundancy penalty:

$$g^* = \arg\max_{g \notin S} \big[\, \hat{U}(S \cup \{g\}) - \hat{U}(S) - \lambda\,\text{Redundancy}(g, S) \,\big]$$

Explicitly a heuristic greedy approximation to a set-level objective; **no submodularity
guarantee is claimed**. JEST (NeurIPS 2024) motivates why set-level, composition-aware
selection matters.

---

## 10. Methodology: Three-Stage Pipeline

Clean causal ordering (no failure information contaminates the candidate pool):

```
D₀ → F → Stage A: neutral pool G₀ → Stage B: X(g) = [difficulty, failure-alignment,
fidelity Q] → Stage C: probe subsets Sᵢ → retrain → Yᵢ = U(Sᵢ) → learn Û → select → AP(B)
```

### Stage A — Neutral candidate generation

Generate a **balanced candidate pool** G₀ *without* failure conditioning, spanning strata:
classes, object scales, object counts, spatial arrangements, contexts, difficulty levels,
and (implicitly) failure-alignment levels. The predictor must therefore discriminate
useful from useless samples **within one broad candidate distribution** — not merely
recognize which pool was failure-conditioned (the leakage the second review identified).

Generator: **GLIGEN** (box + text conditioning, 512 px) as primary; annotations known by
construction. A small **sanity experiment** (200–300 SDXL or Flux images) checks that
findings are not GLIGEN-specific artifacts — directionality check only, no full matrix.

### Stage B — Candidate scoring

For each candidate g: difficulty (EL2N-style), failure alignment F(g) (§5.6), fidelity
Q_box(g) (§5.5), and the context/CLIP embedding. No retraining.

### Stage C — Probe training, prediction, selection

Sample probe subsets Sᵢ (stratified over the feature space), retrain the detector on
D_real ∪ Sᵢ with a fixed short schedule, measure Yᵢ = U(Sᵢ). Learn Û (§7). Select under
budget (§9). Evaluate on AP(B) curves (§11.7).

---

## 11. Experimental Design

### 11.1 Datasets and splits

- **Primary:** MS-COCO — 10% of train2017 as D_real (stratified), untouched val2017 as the
  evaluation set. The 10% regime matches the data-sparse setting where synthetic data is
  known to help (InstaGen).
- **Detector:** YOLO11s primary; RT-DETR secondary (only if primary gates pass).
- **Domain-shift test:** VisDrone (only if the main pipeline is stable). **PASCAL VOC is
  dropped** — three datasets are not needed to answer the questions.

### 11.2 Probe protocol

- Target **64–80 probe subsets** (pilot-calibrated; §13). Sizes |Sᵢ| ∈ {100, 150, 250} —
  probes are for *ranking*, not final performance.
- Fixed short schedule, frozen hyperparameters, mixed precision; **1–2 seeds** in the probe
  stage, **3 seeds** for the final headline comparison (3 strong seeds beat 5 nominal ones
  under the budget).
- Measurement noise floor estimated by duplicate probe trainings (2 seeds on a few subsets).

### 11.3 Pool-level cross-validation (the generalization test)

Independent generation pools G₁…G₄ (different seeds/condition distributions). Train the
utility predictor on probes from pools G₁…G_{K−1}; test on probes from unseen pool G_K;
rotate. **This prevents the predictor from memorizing one generated pool's peculiarities**
and is the difference between a real generalization claim and same-pool overfitting.

### 11.4 Leakage controls (explicit rules)

1. Validation set untouched by generation, selection, feature fitting, probe construction.
2. Failure severity from pre-training diagnostics only — never from measured synthetic ΔAP
   (removes target leakage from feature construction).
3. Probe subsets sampled from multiple pools; features computed per-subset from the same
   data that defines the subset, never pooled across train/test pools.
4. All hyperparameters (probe schedule, elastic-net α, selection λ) chosen on
   probe-training data, locked before the final matrix.

### 11.5 DMR construction (the key control)

Difficulty-matched random pool: matches D_real∪S's class, object-count, size, occlusion
(using **box overlap + density as cheap geometric proxies**, with a manually audited subset
validating the proxy), and difficulty distributions — differing only in whether generation
conditions were selected according to detector failures. Then H3: AP(Failure) vs AP(DMR).

### 11.6 Statistical reporting

Kendall-τ, Spearman ρ, nDCG@K, out-of-fold R²; bootstrap CIs on ΔR²; corr(F,D), VIF,
ablation table; seeds and compute logged per run (wandb/CSV).

### 11.7 Evaluation: synthetic-data efficiency curves

$$\text{AUC} = \frac{1}{B_{\max}} \int_0^{B_{\max}} AP(B)\,dB$$

computed by **trapezoidal rule over B ∈ {250, 500, 1000}**, normalized by B_max. Reported
for both axes: AP vs. number of synthetic images **and** AP vs. GPU-hours (generation +
selection + retraining). The two-axis reporting prevents a method that generates 2,000
images to select 250 from looking efficient on images alone.

---

## 12. Compute Budget

**USD 100 is a hard ceiling, not a quote.** All hour estimates below are provisional and
will be recalibrated by the pilot (§13); probe size/epochs shrink first if over budget
(probes are for ranking, not final performance).

| Item | Provisional estimate |
|------|----------------------|
| Colab Pro × 3 months | ~$30 |
| One Colab Pro+ month | ~$50 |
| Generation (GLIGEN 512 px, ~1,500–2,000 images) | 10–15 GPU-h |
| Probe stage (64–80 × ~0.3–0.5 h) | ~20–40 GPU-h |
| Final matrix (8 methods × 3 budgets × 3 seeds × ~0.75 h) | ~50 GPU-h |
| **Total** | **~$80–100, recalibrated by pilot** |

RT-DETR, VisDrone, and the SDXL sanity check run **only after** the primary gates pass.

---

## 13. Go/No-Go Pilot (Weeks 1–2)

Five measurements de-risk everything else:

1. **GLIGEN at 512 px on Colab:** throughput, artifact rate, and box fidelity μ_Q on 300
   generated images. **Go:** μ_Q ≥ 0.55 mean IoU vs. conditioning boxes.
2. **YOLO11s on 10% COCO:** baseline mAP, failure-profile extraction (FN/FP/loc/confusion/
   uncertainty), EL2N difficulty, and the **corr(F,D) sanity number** (warn if ≥ 0.9).
3. **Probe signal (the make-or-break measurement):** 8–12 pilot probe subsets (|S| = 250) →
   spread of U(S) vs. 2-seed measurement noise; **Go:** SNR ≥ 1.0; preliminary Kendall-τ of
   the best univariate feature ≥ 0.3. This answers: *is the probe signal strong enough to
   predict within ~$100?*
4. **DMR construction:** can the difficulty-matched pool be built while differing only in
   failure-conditioned context?
5. **Calibration:** measured GPU-hours per probe train → sets n_probes for the full
   protocol (64–80) under the ceiling. **Go:** ≤ 0.6 h per probe train.

**No-go** on the probe signal → reduce scope (one generator, fewer probes) or pivot the
utility question to image classification where probe trainings are cheaper.

---

## 14. Timeline (post-pilot)

| Weeks | Work |
|-------|------|
| W3–6 | Full neutral pool + 64–80 probes + utility predictor (RQ1; Gates 1–2) |
| W7–9 | Mechanism + baselines + selection curves (RQ2/RQ3; Gates 3–5) |
| W10–12 | Write-up, release (pipeline, protocol, Streamlit dashboard), submission to one mid-tier venue |

---

## 15. Expected Contributions

1. **A controlled, detection-specific account of synthetic-data utility** (SENSE-analog for
   detection, with detection-specific factors: box fidelity, scale, localization,
   false-positive context).
2. **A failure-vs-difficulty-vs-rarity decomposition**, tested by nested models with
   difficulty-matched controls, answering whether detector-specific failures carry
   incremental predictive information.
3. **A validated pre-training utility predictor** (probe-trained, pool-level CV) for
   synthetic detection subsets — validated for this setting under pool-level generalization.
4. **A budget-aware selection framework** with formally defined efficiency curves under
   image and GPU budgets, benchmarked against TADA and DMR.
5. **Open artifacts:** pipeline, pre-registered protocol, and an interactive Streamlit
   dashboard for inspecting failure profiles and predicted utility.

---

## 16. Publication and PhD Positioning

**Target publication:** one peer-reviewed computer-vision venue appropriate to the
empirical contribution — BMVC / ACCV / WACV / Pattern Recognition-class, or a strong
workshop (CVPR SynData4CV, ICCV Data-Centric AI). A workshop submission may be pursued
during the thesis if the study matures early. The thesis is not reverse-engineered around a
venue; one strong paper beats two promised ones.

**PhD narrative:** the thesis sits at the intersection of computer vision, generative AI,
and data-centric ML, and tells one coherent story — *"can a vision model determine what
training data would most improve itself?"* That question connects object detection, active
learning, data valuation, model diagnostics, and synthetic data, and is a substantially
stronger PhD-admissions story than "I built a YOLO + diffusion augmentation system."

---

## 17. Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Probe signal too weak to predict within $100 | Pilot gate 3 is the explicit go/no-go; fallback to classification-domain probes or reduced scope |
| corr(F, D) high → failure ≠ new signal | Pre-registered redundancy audit; thesis reports it honestly; a null H2 is publishable |
| GLIGEN-specific artifacts | SDXL/Flux sanity check (directionality only) |
| Probe label noise (ΔmAP noisy at 1 seed) | 2-seed duplicates for noise floor; 3 seeds in final matrix |
| Annotation errors in synthetic data | GLIGEN's boxes known by construction; μ_Q fidelity feature; no SDXL auto-annotation in the core protocol |
| Compute overrun | $100 hard ceiling; probes shrink first; final matrix only after gates |
| "First" claims attacked | Gap statement is narrow (§3.6); no "first study of synthetic utility" claim |

---

## 18. References

All references were verified against primary sources (arXiv, CVF/IEEE proceedings, PMLR,
Springer, NeurIPS/ICLR proceedings) in August 2026.

### Synthetic data generation for object detection
1. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. *High-Resolution Image
   Synthesis with Latent Diffusion Models.* CVPR 2022.
2. Zhang, L., Rao, A., & Agrawala, M. *Adding Conditional Control to Text-to-Image Diffusion
   Models (ControlNet).* ICCV 2023.
3. Li, Y., et al. *GLIGEN: Open-Set Grounded Text-to-Image Generation.* CVPR 2023.
   (arXiv:2301.07093; github.com/gligen/GLIGEN)
4. Wu, W., et al. *DatasetDM: Synthesizing Data with Perception Annotations Using Diffusion
   Models.* NeurIPS 2023.
5. Feng, C., et al. *InstaGen: Enhancing Object Detection by Training on Synthetic Dataset.*
   CVPR 2024. (arXiv:2402.05937; +4.5 AP open-vocabulary; +1.2–5.2 AP data-sparse,
   including +5.2 AP = 23.3→28.5 at 10% COCO training data)
6. Zhu, J., Ma, H., Chen, J., & Yuan, J. *Object Detection Data Synthesis via Box-to-Image
   Generation Based on Diffusion Models (ODGEN).* IEEE TPAMI, 2026.
7. Tang, D., et al. *AeroGen: Enhancing Remote Sensing Object Detection with
   Diffusion-Driven Data Generation.* CVPR 2025.
8. Li, Y., Dong, X., Chen, C., Zhuang, W., & Lyu, L. *A Simple Background Augmentation
   Method for Object Detection with Diffusion Model.* ECCV 2024.
9. Jiang, Y., Qiang, S., Li, W., & Liang, Y. *LLM-DiffAug: Enhancing Few-Shot Object
   Detection via LLM-Guided Diffusion Augmentation.* Knowledge-Based Systems, 2025.
10. Chen, S., Sun, P., Song, Y., & Luo, P. *DiffusionDet: Diffusion Model for Object
    Detection.* ICCV 2023.

### Model-guided and targeted synthetic data
11. Hayden, D. S., et al. *Generative Data Mining with Longtail-Guided Diffusion.* ICML 2025,
    PMLR 267.
12. Tavakoli Ghinani, H., Singh, N., Legler, T., Wagner, A., & Ruskowski, M. *Synthetic Data
    and Active Learning for Efficient Object Detection.* CAiSE 2025 Workshops (GenSyn).
13. Nguyen, et al. *Do We Need All the Synthetic Data? Targeted Image Augmentation via
    Diffusion Models (TADA).* ICLR 2026. (arXiv:2505.21574; code: BigML-CS-UCLA/TADA)
14. Sarkar, K. *Synthetic Designed Experiments for Diagnosing Vision Model Failures.* CVPR
    2026 Workshop (SynData4CV).
15. Durasov, N., et al. *Uncertainty-Guided Data Curation for 3D Object Detection.* CVPR
    2026 Workshop.
16. Zhou, Q., et al. *Targeted False Positive Synthesis via Detector-guided Adversarial
    Diffusion Attacker for Robust Polyp Detection.* MICCAI 2025.

### Synthetic-data utility analysis
17. Zhang, J., Guo, X., Jin, Y., Zhou, N., & Huang, D. *What Makes Synthetic Data Effective
    in Image Segmentation (SENSE).* ICML 2026. (arXiv:2605.19289)

### Data valuation and data curation
18. Yoon, J., Arik, S., & Pfister, T. *Data Valuation using Reinforcement Learning (DVRL).*
    ICML 2020.
19. Ghorbani, A., & Zou, J. *Data Shapley: Equitable Valuation of Data for Machine
    Learning.* ICML 2019.
20. *CHG Shapley* (arXiv:2406.11730) — data selection via subset-utility at single
    retraining cost. Cited as arXiv (not accepted at ICLR 2025).
21. Sorscher, B., Geirhos, R., Shekhar, S., Ganguli, S., & Morgenstern, S. *Beyond Neural
    Scaling Laws: Beating Power Law Scaling via Data Pruning.* NeurIPS 2022.
22. Gadre, S. Y., et al. *DataComp: In Search of the Next Generation of Multimodal
    Datasets.* NeurIPS 2023.
23. Fang, A., et al. *Data Filtering Networks.* ICLR 2024. (venue corrected from earlier
    drafts)
24. Goyal, S., et al. *Scaling Laws for Data Filtering — Data Curation cannot be Compute
    Agnostic.* CVPR 2024.
25. Laurençon, H., et al. *Data Curation via Joint Example Selection Further Accelerates
    Multimodal Learning (JEST).* NeurIPS 2024 Datasets & Benchmarks.
26. Paul, M., Ganguli, S., & Dziugaite, G. K. *Deep Learning on a Data Diet: Finding
    Important Examples Early in Training.* NeurIPS 2021. (EL2N/GraNd; transfer beyond
    pruning settings not assumed)

### Benchmarks and adjacent work
27. Lin, T.-Y., et al. *Microsoft COCO: Common Objects in Context.* ECCV 2014.
28. Cao, Y., et al. *VisDrone-DET2021: The Vision Meets Drone Object Detection Challenge
    Results.* ICCV Workshops 2021.
29. Everingham, M., et al. *The PASCAL Visual Object Classes Challenge.* IJCV 88, 2010.
30. Fang, X. *Synthetic Data for Object Detection: Improving Robustness and Revealing
    Vulnerabilities.* MS thesis, CMU Robotics Institute, 2026.
31. Araya-Martinez, et al. *A Data-centric Evaluation of Leading Multi-class Object
    Detection Algorithms Using Synthetic Industrial Data.* Springer, 2025.
32. Mütze, A., Ilyas, S., Dörpelkus, C., & Rottmann, M. *Can We Challenge Open-Vocabulary
    Object Detectors with Generated Content in Street Scenes?* WACV 2026.
33. Ultralytics. *YOLO11.* Software (no formal paper; cited as software).

### Added in this verification pass (August 2026)
34. Marian, V., Kang, Y.-B., & Buddery, A. *Do Generative Metrics Predict YOLO Performance?
    An Evaluation Across Models, Augmentation Ratios, and Dataset Complexity.*
    arXiv:2602.18525, 2026. (Closest prior work to RQ1 at the generator/ratio level; cited
    and differentiated in §2.3, §3.5, §3.6.)

---

## Appendix A — Revision history

| Version | Change |
|---------|--------|
| v1 (DOSE-OD) | Failure-guided, budget-aware generation; hand-designed U = F·Q·D — rejected: heuristic unvalidated |
| v2 (FUSE-OD) | Utility prediction as instrument; failure-vs-difficulty decomposition; difficulty-matched control |
| v3 | TADA added as first-class baseline; 32–48 probes; leakage controls |
| v4 (PredU-OD) | Utility prediction made centerpiece; 7-feature predictor; formal AUC; GLIGEN sanity check |
| **v5 (this document)** | Neutral candidate pool (no failure leakage into G₀); μ_Q fidelity feature; four-model null stack; five gates; 64–80 probes; redundancy audit; $100 as hard ceiling; three-stage causal ordering; ∇_data L overclaim dropped in favor of an empirical hypothesis |
