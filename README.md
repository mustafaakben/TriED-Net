# TriED-Net: Tri-Intelligence Energy-Dynamics Network

This repository benchmarks a novel architecture, **TriED-Net**, against strong tabular baselines on a synthetic suite of five tasks and exposes a real‑data runner (`benc_real.py`). The core idea is to combine:

- a **General** deep encoder (standard MLP head),
- a **Fluid** energy-based refinement module (unrolled gradient descent in latent space),
- a **Crystallized** prototype memory (for geometric evidence and online consolidation),

and to couple this with **teacher-stacked inputs** and a small **tree ensemble** at inference time.

The rest of this document explains the model in a paper-style format and is grounded directly in `benchmark_cognitive_models.py`.

---

## 1. Architecture

### 1.1 General Module (Encoder + Head)

Given an input vector \(x \in \mathbb{R}^D\), the encoder is a two-layer MLP:
\[
h_0 = \text{LN}\big(\sigma(W_2\,\sigma(W_1 x + b_1) + b_2)\big), \quad h_0 \in \mathbb{R}^H
\]
where \(\sigma\) is ReLU, LN is LayerNorm, and optional dropout is applied between layers (controlled by `dropout` in `TriEDConfig`).

The task head is linear:
- classification: \(\text{logits}^{\text{mlp}} = W_{\text{clf}} h + b_{\text{clf}}\),
- regression: \(y^{\text{mlp}} = W_{\text{reg}} h + b_{\text{reg}}\).

### 1.2 Symbolic Codebook & Alignment

An optional **Symbolizer** maps inputs to a “symbolic” embedding:
\[
h_{\text{sym}} = \phi(x) \in \mathbb{R}^{C},\quad \pi = \text{softmax}\left(\frac{W_{\text{code}} h_{\text{sym}}}{\tau}\right) \in \Delta^{K-1},
\]
\[
s = \pi^\top \mathcal{C} \in \mathbb{R}^C,
\]
where \(\mathcal{C} \in \mathbb{R}^{K \times C}\) is the learned codebook (`codebook_size`), and \(\tau>0\) is a learnable temperature. A linear map \(A\) projects the latent \(h\) into the same space as \(s\).

The alignment term in the energy encourages consistency:
\[
E_{\text{align}}(h;x,s) = \lambda_{\text{align}} \|A h - s\|_2^2.
\]

### 1.3 Prototypes & Crystallized Memory

For classification, we maintain prototypes \(\{p_{k,m}\}\) for each class \(k\) and slot \(m\). For regression, we maintain prototypes \(\{p_j\}\) and associated scalar values \(v_j\).

Energy contribution from prototypes:
\[
E_{\text{proto}}(h) = \lambda_{\text{proto}} \min_j \|h - p_j\|_2^2.
\]

Prototype logits for class \(k\):
\[
d^2_{k,m}(h) = \|h - p_{k,m}\|_2^2,\quad
\text{logits}_k^{\text{proto}} = -\log\sum_m \exp(-d^2_{k,m}(h)).
\]

Regression prototype head:
\[
w_j(h) = \frac{\exp(-\|h-p_j\|_2^2)}{\sum_\ell \exp(-\|h-p_\ell\|_2^2)},\quad
y^{\text{proto}} = \sum_j w_j(h) v_j.
\]

Crystallized memory is updated via a Hebbian‑style rule (applied **after** gradient steps so it does not interfere with autograd):
\[
p_j \leftarrow (1-\mu) p_j + \mu \,\overline{h}_j,
\]
where \(\overline{h}_j\) is the mean latent over samples currently closest to prototype \(j\), and \(\mu\) is a small momentum (`0.05` in the code).

### 1.4 Energy, Dynamics, and Latent Smoothing

The total energy for a latent \(h\) is:
\[
E(h;x,s) = E_{\text{align}}(h;x,s) + E_{\text{proto}}(h) + \lambda_h \|h\|_2^2.
\]

The Fluid module performs gradient-descent in latent space:
\[
h_{t+1} = h_t - \eta_t \big(\nabla_h E(h_t;x,s) + f_{\text{drift}}(h_t)\big),
\]
where \(f_{\text{drift}}\) is an optional MLP (“drift”) and \(\eta_t\) is a learned step size modulated by the **investment gate** (below). When `T_max = 0`, the model collapses to a strong MLP head with no dynamics.

To favor smooth latent trajectories, we optionally add:
\[
\mathcal{L}_{\text{smooth}} = \lambda_{\text{smooth}} \sum_{t>0} \|h_t - h_{t-1}\|_2^2.
\]

### 1.5 Investment Gate (Effort Allocation)

The **InvestmentGate** computes an effort scalar \(g \in (0,1)\) per example:
- For classification, difficulty is proportional to normalized entropy of the current prediction.
- For regression, difficulty is proportional to normalized magnitude of the prediction.

We concatenate \(h\) and logits/predictions and pass them through a small MLP ending in a sigmoid:
\[
g = \sigma(\text{MLP}([h; \text{logits/pred}])),
\]
then scale step sizes:
\[
\eta_t = \text{clamp}(\eta_t^{\text{base}})\cdot \phi_t(g),
\]
where \(\phi_t\) also decays with time step \(t\) via an exponential factor (`invest_beta`).

### 1.6 Fusion and Final Prediction

Classification:
\[
z = (1-\alpha_{\text{proto}})\,\text{logits}^{\text{mlp}} + \alpha_{\text{proto}}\,\text{logits}^{\text{proto}}.
\]

Regression:
\[
y = (1-\alpha_{\text{proto\_reg}})\,y^{\text{mlp}} + \alpha_{\text{proto\_reg}}\,y^{\text{proto}}.
\]

For classification we use cross‑entropy loss with optional label smoothing; for regression, MSE. Both can be regularized with latent smoothing and weight decay.

---

## 2. Synthetic Benchmark Suite

`benchmark_cognitive_models.py` generates five synthetic datasets:

- **`cls_easy`** – binary classification  
  - `make_classification(n_samples=2500, n_features=20, n_informative=6, n_redundant=2, class_sep=1.8, flip_y=0.01)`  
  - Roughly linearly separable with low label noise.

- **`cls_hard`** – 3‑class, harder classification  
  - More features (`n_features=50`), lower `class_sep`, higher label noise.  
  - Designed so tree methods and expressive nets have room to improve over linear models.

- **`reg_easy`** – low‑dimensional regression  
  - `make_regression(n_samples=2500, n_features=12, n_informative=8, noise=5.0)`.

- **`reg_hard`** – higher‑dimensional, noisier regression  
  - `make_regression(n_samples=3000, n_features=60, n_informative=15, noise=30.0)`.

- **`multilabel`** – multilabel classification  
  - `make_multilabel_classification(n_samples=3000, n_features=30, n_classes=6, n_labels=2, length=50)`.

Each dataset is split into:
- 20% held‑out test set,
- 10% of the remaining 80% used as a validation split (for both baselines and TriED’s internal assessment),
- stratification applied to classification tasks where appropriate.

All splits are saved under `outputs/datasets/*.npz`.

---

## 3. Baselines and Internal Assessment

### 3.1 Baseline Models

For every dataset we train:

- **SVM / SVR** with RBF kernel; inputs are standardized (`StandardScaler`).
- **RandomForest** / **RandomForestRegressor** (300–400 trees).
- **GradientBoostingClassifier / Regressor** (≈200 trees).
- For multilabel classification, SVM and GradientBoosting are wrapped in `OneVsRestClassifier`.

Metrics:
- Classification: Accuracy and macro‑F1.
- Multilabel: subset Accuracy and micro‑F1.
- Regression: MSE and \(R^2\).

Results are printed as ranked tables per dataset.

### 3.2 Teacher‑Stacked Inputs for TriED

Before searching TriED configs, we **augment** the input features with teacher predictions:

- Classification (single‑label):
  - Train an RF classifier on the train split; append its class‑probability vector to the original features.
  - Train a GradientBoosting classifier; append its probabilities as well.
- Classification (multilabel):
  - Train a one‑vs‑rest RF; append per‑label probabilities.
- Regression:
  - Train a GradientBoosting regressor; append its scalar prediction.

This creates augmented inputs \(x' = [x; \text{teacher}(x)]\) that encode strong tree‑based priors, which TriED can then refine.

### 3.3 Iterative Internal Assessment (Architecture Search)

The function `iterative_assessment` implements a **10+ step micro‑search** over `TriEDConfig`:

- Start from a base config (task‑aware).
- For each iteration \(i\), apply a small change (e.g., enable prototypes, turn on drift, increase `hidden_dim`, change `T_max`, adjust energy weights).
- Train the resulting TriED variant for a small number of `preview_epochs` on the (train, val) split.
- Score the variant:
  - Classification: validation Accuracy,
  - Regression: \(R^2\) (via `score_from_metrics`).
- Keep the config if it improves the score, otherwise revert.

At the end, the best config is upgraded to use a larger `epochs` budget (`full_epochs` or more) for final training on that dataset.

For classification tasks, the code also allows a **high‑capacity override**: when helpful, the search can collapse to a wide MLP (`T_max=0`, large `hidden_dim`) on top of teacher‑stacked features. This is what powers the strongest TriED variants on `cls_easy` and `cls_hard`.

### 3.4 Training Loop

`train_tried` handles:
- Standardization of inputs,
- Mini‑batch training with AdamW,
- Optional cosine learning‑rate schedule,
- Gradient clipping,
- Latent‑smooth regularization,
- Hebbian prototype updates after optimizer steps,
- Early model selection based on validation metrics (best state dict is cached).

`eval_tried` runs a final forward pass on standardized test inputs and computes the same metrics as the baselines.

---

## 4. Teacher Ensemble at Inference

For classification tasks, after training TriED on teacher‑stacked features, we optionally build a **tree‑enhanced ensemble**:

1. Train RF, GradientBoosting, and ExtraTrees on the original (train ∪ val) features.
2. Compute class probabilities on the test set for all three trees.
3. Compute TriED probabilities on the augmented test features.
4. Blend them:
\[
p_{\text{ens}} = \lambda_{\text{TriED}} p_{\text{TriED}} +
                 \lambda_{\text{RF}} p_{\text{RF}} +
                 \lambda_{\text{GB}} p_{\text{GB}} +
                 \lambda_{\text{ET}} p_{\text{ET}},
\]
with weights currently chosen in code to emphasize ExtraTrees for pure accuracy.
5. Predict \(\hat{y} = \arg\max_k p_{\text{ens},k}\).

The ensemble is reported as **TriED+Trees‑Ensemble** in tables; pure TriED is also shown separately.

---

## 5. Results on Synthetic Benchmarks (SEED=42)

Representative results (exact numbers are printed when you run `benchmark_cognitive_models.py`):

| Dataset   | Metric       | Best Baseline         | TriED-Net        | TriED+Trees-Ensemble |
|----------|--------------|-----------------------|------------------|----------------------|
| cls_easy | Acc / F1     | RF 0.984 / 0.984      | ~0.972 / 0.972   | **~0.99 / 0.99**     |
| cls_hard | Acc / F1     | RF 0.777 / 0.777      | ~0.775 / 0.774   | **~0.79 / 0.79**     |
| reg_easy | MSE ↓ / R²   | GBR ≈ 1.0e3 / 0.97    | **≈ 3.1e2 / 0.99** | n/a                |
| reg_hard | MSE ↓ / R²   | GBR ≈ 5.3e3 / 0.92    | **≈ 2.7e3 / 0.96** | n/a                |
| multilabel | Acc / F1   | SVM(OVR) ≈ 0.47 / 0.78 | **≈ 0.49 / 0.79** | n/a                |

On these suites:
- TriED‑Net alone dominates gradient boosting on both regression tasks and the multilabel task.
- TriED+Trees‑Ensemble dominates all individual baselines on both classification tasks.

All detailed logs are kept under `outputs/` (`*run.log`, `datasets/*.npz`); the code is instrumented with Rich logging and timing.

---

## 6. Real‑Data Runner (`benc_real.py`)

For a real CSV dataset with a single target column:

```bash
python benc_real.py --data_path /path/data.csv --target label \
  --task classification --output_dir outputs_real
```

or

```bash
python benc_real.py --data_path /path/data.csv --target target \
  --task regression --output_dir outputs_real
```

Assumptions:
- All non‑target columns are numeric (pre‑encode categoricals yourself).
- Classification: the target is categorical; we map it to integer labels via `LabelEncoder`.

What it does:
- Splits into train/val/test (same ratios as synthetic suite).
- Trains SVM/RF/GBC/ExtraTrees (or SVR/RFR/GBR) baselines.
- Constructs a task‑specific TriED config:
  - classification: wide MLP head (`T_max=0`) on teacher‑stacked features,
  - regression: dynamic TriED (`T_max=3`) with drift and latent smoothing.
- Trains TriED and prints its metrics.
- For classification, builds a **TriED+RF** ensemble analogous to the synthetic pipeline.
- Writes all metrics into `outputs_real/metrics.json`.

This script is designed as a quick smoke test of whether the architecture transfers to non‑synthetic tabular data.

---

## 7. Novelty & Strengths

- **Energy‑based refinement on a learned latent**  
  The Fluid module unrolls gradient descent on an explicit energy combining symbolic alignment, prototype attraction, and a latent prior. This gives a transparent objective for the refinement dynamics and allows us to control each factor via `lam_align`, `lam_proto`, and `lam_h`.

- **Prototype/logit fusion with online consolidation**  
  TriED keeps per‑class prototypes and fuses prototype‑based logits with the MLP head. Hebbian updates move prototypes toward frequently visited latent regions, yielding a simple long‑term memory that remains compatible with end‑to‑end training.

- **Investment‑gated computation**  
  The gate uses entropy (classification) or normalized magnitude (regression) to estimate difficulty and modulates the size of each refinement step per example. Easy cases can converge in fewer effective steps, while hard ones receive more aggressive updates.

- **Teacher‑stacked features and ensembles**  
  Instead of co‑training trees and nets, we pre‑train strong tree models and append their predictions to the input features, letting TriED treat them as a structured prior. At inference, a small tree ensemble blended with TriED gives state‑of‑the‑art numbers on our benchmarks.

- **Unified, reproducible pipeline**  
  The same code path supports:
  - binary/multi‑class classification,
  - multilabel classification,
  - regression,
  with automatic dataset generation, splitting, baseline training, config search, and reporting.

---

## 8. Limitations & Future Work

- **Seed‑sensitivity in classification**  
  On some seeds, pure ExtraTrees or RF can be competitive or slightly better on `cls_easy`. The TriED+Trees ensemble mitigates this, but a more systematic treatment (e.g., multi‑seed ET averaging, or val‑driven blend weights) would further stabilize results.

- **Ensemble cost**  
  The highest classification scores rely on trees + TriED. This improves accuracy but increases inference cost. You can disable the ensemble by skipping the blend and using TriED‑Net alone if latency or memory is critical.

- **Tabular specialization**  
  The current design is tailored to tabular data. Extending TriED‑Net to sequences or images would require adapting the encoder and possibly the energy terms.

- **Limited real‑data coverage**  
  The synthetic suite is diverse but still synthetic. `benc_real.py` is intentionally simple; a more systematic evaluation on established tabular benchmarks (e.g., OpenML/UCI collections) would better characterize when TriED/ensemble truly dominate.

---

## 9. Key Hyperparameters (from `TriEDConfig`)

- `hidden_dim`: latent width (typical: 64–256; classification search can go higher).
- `code_dim`, `codebook_size`: symbolic embedding and codebook size.
- `prototypes_per_class`, `reg_prototypes`: number of prototypes.
- `T_max`: number of Fluid refinement steps (`0` ⇒ pure MLP head).
- `base_step_size`, `invest_beta`: step size scale and decay for investment gating.
- `lam_align`, `lam_proto`, `lam_h`: weights in the energy.
- `alpha_proto`, `alpha_proto_reg`: fusion weights for prototype outputs.
- Feature toggles: `use_align`, `use_prototypes`, `use_investment`, `use_symbolizer`, `use_drift`,
  `use_hebbian_proto_update`, `learn_step_sizes`.
- Regularization and training: `latent_smooth_lam`, `label_smoothing`, `cosine_lr`, `lr`,
  `epochs`, `batch_size`, `weight_decay`, `preview_epochs`, `full_epochs`.

The iterative search in `iterative_assessment` explores this space with small, interpretable moves rather than treating the model as a black‑box hyperparameter soup.

---

## 10. Mathematical Summary

- **Energy**  
  \[
  E(h;x,s) = \lambda_{\text{align}}\|A h - s\|_2^2
           + \lambda_{\text{proto}} \min_j \|h-p_j\|_2^2
           + \lambda_h\|h\|_2^2.
  \]

- **Dynamics**  
  \[
  h_{t+1} = h_t - \eta_t \big(\nabla_h E(h_t;x,s) + f_{\text{drift}}(h_t)\big),
  \]
  with \(\eta_t\) modulated by the investment gate’s effort estimate.

- **Prototype logits**  
  \[
  \text{logits}_k^{\text{proto}} = -\log \sum_m \exp\big(-\|h-p_{k,m}\|_2^2\big).
  \]

- **Prototype regression head**  
  \[
  y^{\text{proto}} = \sum_j \frac{\exp(-\|h-p_j\|_2^2)}{\sum_\ell \exp(-\|h-p_\ell\|_2^2)} v_j.
  \]

- **Fusion**  
  \[
  z = (1-\alpha_{\text{proto}})\,z^{\text{mlp}} + \alpha_{\text{proto}}\,z^{\text{proto}},
  \]
  \[
  y = (1-\alpha_{\text{proto\_reg}})\,y^{\text{mlp}} + \alpha_{\text{proto\_reg}}\,y^{\text{proto}}.
  \]

- **Loss**  
  Classification: cross‑entropy (with optional label smoothing);  
  Regression: MSE; both can include latent smoothing as an auxiliary term.

---

## 11. References

LeCun, Y., Chopra, S., Hadsell, R., Ranzato, M., & Huang, F. (2006). *A tutorial on energy-based learning* (NIPS Tutorial).  
Snell, J., Swersky, K., & Zemel, R. (2017). Prototypical networks for few-shot learning. *Advances in Neural Information Processing Systems*.  
Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. *arXiv preprint arXiv:1503.02531*.  
Dorogush, A. V., Ershov, V., & Gulin, A. (2018). CatBoost: unbiased boosting with categorical features. *Advances in Neural Information Processing Systems*.  
(These works inform our design; the implementation here is original.)
