# ViewGroundRL

View-grounded reinforcement learning for multi-view driving VLMs.

A surround-view VLM can answer correctly while looking at the wrong camera. This
repo adds a term to GRPO that pushes the policy to answer *from the camera view
that actually contains the evidence*, and a diagnostic that measures whether it
does.

## Method

For a rollout `y` sampled with all six views present, score the same `y` under
three conditions — all views, the evidence view blanked, a control (non-evidence)
view blanked — and contrast the two masked branches.

```
δ_t^ev   = log πθ(y_t | V) − log πθ(y_t | V \ {e})      e ~ U(E)   evidence view
δ_t^ctrl = log πθ(y_t | V) − log πθ(y_t | V \ {c})      c ~ U(C)   control view

kl₁ = Σ_t m_t δ_t^ev  / Σ_t m_t          m_t = 1 for content tokens only
kl₂ = Σ_t m_t δ_t^ctrl / Σ_t m_t
Δ   = kl₁ − kl₂

L_view = λ · softplus(−Δ)                 λ = 0.04
L      = L_GRPO + 1[step > warmup] · L_view
```

Three choices matter, and each is there because the obvious alternative was
measured to fail:

**Directional estimator (k1, not k3).** `k3 = r − log r − 1` is non-negative by
construction, so it scores "the distribution moved" rather than "the correct
answer got harder". On NuInstruct the untrained model scores k3 kl₁/kl₂ =
0.839/0.255 — an apparent 3.3× selectivity — while its k1 values are
−0.030/−0.029, i.e. none at all.

**Matched control branch.** Both branches blank the same number of views, so
`Δ` reflects *which* view was removed rather than how much image was removed.
Without it (PAPO's kl₁-only form) a model that reacts equally to every view
scores as grounded: Qwen3-VL-8B on DriveLM has kl₁ ≈ kl₂ ≈ 0.60.

**Bounded transform.** `softplus(−Δ)` has gradient `−σ(−Δ)`, which vanishes as
the margin grows, so the term switches itself off once grounding is achieved.
The linear form `−λΔ` keeps pushing: it reached Δ = +4.64 nats (k3: 3.8e13) with
task performance dropping.

Since the full-view term cancels in `Δ`, the view term is exactly a contrastive
log-likelihood between the two masked branches, and `softplus(−Δ) = −log P`
where `P` is the two-way posterior of picking the control branch — the InfoNCE
form, a bounded lower bound on `I(Y; V_E | V_C)`.

## Diagnostic

`tools/measure_view_kl.py` reports the per-token log-probability drop of the
*reference* answer in nats, split three ways — all tokens, content tokens only,
grounded spans only. Function words (`the`, `is`, punctuation) contribute almost
equally to both branches and so dilute the average; NuInstruct answers are 62%
content tokens, OmniDrive answers 45%.

Two conditions have to hold together, and the margin alone hides it:

- `drop_ev ≫ 0` — the evidence actually matters
- `drop_ctrl ≈ 0` — other views do not, **in either direction**

The second one is easy to miss. Untrained Qwen3-VL-8B on DriveLM has
`drop_ctrl = −0.154`: blanking an irrelevant view makes the answer *easier*
because a distractor disappeared. That inflates the margin without any
grounding.

## Measured

NuInstruct, Qwen2.5-VL-7B, current frame only, full validation split (15,046).
Grounding is the content-token drop in nats; `Average*` is the composite from
VGGDrive Table 2, `max((Acc + MAP + BLEU − MAE)/4, 0)`.

| | drop_ev | drop_ctrl | Average* |
|---|---|---|---|
| untrained | −0.056 | −0.049 | — |
| SFT | +0.472 | −0.002 | 47.35 |
| + GRPO, no view term | +0.533 | −0.006 | 47.04 |
| + GRPO, view term | +2.386 | −0.005 | **47.61** |

The view arm differs from the arm above it only in the view term — same reward,
data, and step budget — so the +0.57 is attributable.

What the numbers say that the headline does not:

- **Plain GRPO barely advances grounding.** SFT +0.472 → +0.533. The
  problem is not that RL fails to ground; it is that it plateaus where SFT left
  off.
- **The untrained model is the only one at zero.** So "existing models do not
  look at the right view" holds for pre-SFT models, not for SFT or GRPO ones.
- **SFT grounds by suppressing kl₂, not by raising kl₁.** 0.255 → 0.010, a 24×
  drop, while kl₁ *fell* 0.839 → 0.539.

### Known failures

**Multiple evidence views.** `num_views_to_mask = 1` blanks one view even when
the answer depends on several, so the remainder still supports it and the loss
reads that as a grounding failure. On rows with `|E| ≥ 2` the normalized margin
is **−0.47** — the wrong sign. Reproduced on four model/dataset pairs.

**Mask detection.** Blanking to black is a signal absent from natural images, so
`Δ` can rise without reading the view. In the view arm the non-content-token
drop reached +1.976 against +2.386 for content tokens — a 1.2× ratio, where SFT
is 9.5× and plain GRPO 12.5×. `the` should not depend on a camera. Substituting
noise (`mask_mode: noise`) rather than black is the control that separates them;
not yet run.

### Dataset notes

- **NuInstruct** — usable, but 65.6% of `perception-closest` questions name a
  direction that maps 1:1 to a camera, so blanking it destroys the answer
  trivially. Only ~7% of rows are cleanly usable for the diagnostic.
- **DriveLM** — unusable. An image-blind baseline reaches 88% of the full f1,
  and the untrained model's margin is 1.08×: the labels are recoverable from
  text.
- **OmniDrive** — cleanest question distribution (0.7% direction words), but the
  coordinate tokens carry *negative* margin, so the answer coordinates are not
  view-grounded.

## Layout

```
vgrl/losses.py             view-grounding term, k1/k3 estimators, view picking
vgrl/grpo_view_trainer.py  GRPO subclass: three forward passes, loss injection
vgrl/masking.py            pixel-space view blanking (separate images or 2x3 grid)
vgrl/views.py              evidence-view parsing from answer text
vgrl/rewards.py            rule-based rewards (DriveLM, DriveLMM-o1, NuInstruct, OmniDrive)
vgrl/tokens.py             content-token mask, shared by training and diagnostic
vgrl/cvge.py               VGGT cross-view geometric encoder (VGGDrive stage 1)
tools/measure_view_kl.py   the diagnostic
tools/*_eval.py            rule-based metrics for each benchmark
data_prep/                 dataset preparation, incl. 3D→camera projection for labels
configs/                   one YAML per run
docs/METHOD.md             longer write-up
```

## Running

```bash
scripts/dr.sh 0,1,2,3 accelerate launch --num_processes 4 \
  train_rl.py --config configs/rl_nuins_content.yaml
```

`scripts/dr.sh <gpu-list> <cmd>` runs inside the training image; paths in the
configs are absolute and will need editing.

Requires `transformers` 5.x with `trl` 1.9.x. `vgrl/cvge.py` additionally needs
[VGGT](https://github.com/facebookresearch/vggt) importable as `vggt`.

## Attribution

Developed alongside a fork of
[MindDriver](https://github.com/hotdogcheesewhite/MindDriver) (Apache-2.0); none
of its code is included here, and nothing in this repo imports it. Apache-2.0,
inherited from that project.

The KL-masking family this builds on:
[PAPO](https://arxiv.org/abs/2507.06448) (kl₁ only, no control branch),
[Evidence-RL/CED](https://arxiv.org/abs/2608.08021) (region-level control
regions, tanh-bounded reward gate),
[VEPO](https://arxiv.org/abs/2606.03937) (JSD, token selection).
The view-level evaluation setting is
[Where Does the Answer Come From?](https://arxiv.org/abs/2606.09644), which is
test-only and proposes no training method.
