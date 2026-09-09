# View-grounded RL for driving VLMs

## The problem

A driving VLM given six surround views can answer a lot of questions without
looking at the right one. Language priors carry it a long way: "is the vehicle
behind me moving?" is usually answerable as "yes, at a constant speed" without
inspecting `CAM_BACK` at all. The reward in RL does not distinguish an answer that
was read off the correct view from one that was guessed, so nothing pushes the
policy toward perception.

## The term

Sample a completion normally. Then re-score that *same* completion twice, with one
camera view blanked each time:

```
kl1 = KL( p(y | all views)  ||  p(y | evidence view blanked) )
kl2 = KL( p(y | all views)  ||  p(y | control view blanked) )

margin = kl1 - kl2
L_view = -coef * margin
L      = L_GRPO + L_view
```

The *evidence view* is the camera the ground-truth answer's object references
point at; the *control view* is any camera they do not. A model that reads the
right view has a large `kl1` — blanking the evidence changes its answer — and a
small `kl2`, because blanking an irrelevant view should change nothing. So
`margin` should be large, and since trainers minimise, the term carries a minus
sign.

### Why the control branch

`kl1` alone is PAPO's implicit perception loss, and it is maximised just as well by
a degenerate solution: become brittle to *any* missing pixels. Such a model raises
`kl1` without ever learning which view matters. Subtracting `kl2` removes that
shortcut — uniform brittleness raises both terms and nets zero gain, so the only
way to increase the margin is to make the dependence *view-specific*.

`sign: kl1_only` recovers PAPO exactly, which makes the control branch's
contribution measurable as a single-flag ablation.

### Matched masking budget

Both branches blank the **same number of views** (`num_views_to_mask`, default 1).
Without this the margin is confounded: masking two evidence views against one
control view raises `kl1` simply because more of the image is gone. This is
enforced in `pick_mask_views`, which also shrinks the count to whatever both sides
can supply.

### Samples that cannot contribute

A QA pair needs at least one evidence view *and* one non-evidence view to form the
contrast. Two cases fail:

- the answer names no camera (many `behavior` questions) — no evidence view;
- the answer names all six — no control view.

Those samples are marked `valid=False` and are excluded from the perception term.
They still receive the normal GRPO gradient, so no data is wasted; only the
perception signal is skipped. `vg/frac_valid` logs the fraction that contributes,
and `data_prep/drivelm_prepare.py --require_evidence_for_rl` can drop them
entirely instead.

## On the sign

The method as originally described adds `kl1 - kl2` **as a loss**. Under a
minimised objective that trains the model to make `kl1` small (become insensitive
to the evidence view) and `kl2` large (become sensitive to irrelevant views) —
the opposite of grounding. The implementation therefore defaults to
`L_view = -coef * (kl1 - kl2)`, i.e. it maximises the margin.

`sign: literal` is kept as a deliberate ablation. It should *degrade* grounding,
which makes it a useful negative control: if it does not, the term is not doing
what the method claims.

## Implementation notes

**KL estimator.** `per_token_kl_k3` is the standard k3 estimator
`r - log r - 1` with `r = q(x)/p(x)`, evaluated on the sampled tokens. Since the
completion was drawn from `p` (the unmasked policy), this is a non-negative,
low-variance estimate of `KL(p || q)`. It needs only per-token log-probs — no
full 168k-way distribution — which is what makes three branches affordable. Same
estimator TRL uses for its reference-model KL and PAPO for its perception term.

**Masking.** Views are blanked in *processed patch space*. Qwen2.5-VL concatenates
every image into patch rows, so image *j* owns `image_grid_thw[j].prod()` rows and
a view's span is exactly addressable — no re-running the image processor per
branch. The fill value is taken from the processor's own output on a black image
rather than assumed, so it stays correct if the patch layout or normalisation
constants change.

**Cost.** Two extra teacher-forced forward passes per optimiser step (no extra
generation). Both are differentiable by default; `detach_control_branch: true`
trades the `kl2` gradient for a cheaper backward.

**Gradient safety.** When no sample in a batch is valid the term returns
`logps.sum() * 0.0` rather than a fresh zero, keeping the autograd graph connected
so DDP does not stall on an unused-parameter mismatch.

**KL hacking.** PAPO reports the perception term collapsing training by driving
divergence up degenerately. `margin_clip` (default 10.0) bounds the per-sequence
contribution; `warmup_steps` delays the term until plain GRPO has settled. Watch
`vg/kl1` and `vg/kl2`: both climbing together means hacking, `kl1` climbing while
`kl2` stays flat is the intended behaviour.

## Two stages

**Stage 1 — SFT, no view term.** DriveLM + nuScenes planning, plain
cross-entropy on teacher-distilled CoT. The perception contrast is meaningless
before the model reliably emits the `<think>/<answer>` structure and the
`<cID,CAMERA,x,y>` reference format, because `kl1` would then be dominated by
format noise rather than perception.

**Stage 2 — GRPO with the view term.** DriveLM only, since it is the dataset whose
answers carry per-view object references and therefore the only one that can label
an evidence view without extra annotation.

## Building the SFT labels

Neither dataset ships reasoning traces, so they are distilled from a teacher VLM
(Qwen2.5-VL-72B by default) — the same idea as MindDriver's annotation pipeline,
restructured around two properties it lacked.

**The ground truth is never the teacher's.** `<answer>` is always the dataset
label: the nuScenes trajectory, or the DriveLM reference answer. The teacher only
ever supplies the text inside `<think>`. A weak teacher can therefore give a weak
rationale but can never introduce a wrong label.

**Two passes (STaR-style).**

1. Generate reasoning *without* showing the teacher the answer. Keep the sample if
   the reasoning is consistent with the ground truth.
2. For the rest, regenerate *with* the answer supplied and ask the teacher to
   justify it.

Pass 1 gives genuine reasoning; pass 2 recovers hard samples rather than
discarding them. Every row records `cot_pass`, so the share of rationalised
labels is measurable and can be capped with `--max_hint_fraction`.

**Consistency is checked for free on nuScenes.** The trajectory itself says what
the ego did, so `data_prep/behavior.py` derives the manoeuvre (lateral: straight /
left / right; longitudinal: constant / accelerate / decelerate / rapid variants /
stop / stationary) and a CoT that concludes something contradictory is rejected
deterministically. It also rejects CoTs that leak waypoint coordinates, which
would let the student copy numbers out of its own reasoning instead of planning.
MindDriver spent a second LLM call per sample on this
judgement (`gen_data/check.py`); here it costs nothing and is reproducible.

On DriveLM the check compares the teacher's own stated answer against the
reference using the *same* scoring the RL reward uses — exact match on the closed
behaviour vocabulary, option letter for multiple choice, token F1 or
object-reference overlap otherwise — so the two stages agree on what "correct"
means.

**The CoT is not asked to name its evidence view.** This is deliberate. If the
reasoning template said "the answer comes from CAM_BACK", the student could learn
to emit that phrase without looking, and the RL term above would be rewarding a
text pattern rather than perception. The prompt therefore asks for ordinary
chain-of-thought; a camera named incidentally in prose is left alone.
`--forbid_camera_mentions` strips them if an experiment needs it, though for
DriveLM that fights the reference answers, which cite `<cID,CAMERA,...>`
themselves.

## What to measure

The term is only worth its cost if it changes *grounding*, not just reward. Three
checks:

1. **`vg/margin` over training** — should rise. If `vg/kl1` and `vg/kl2` rise
   together, the model is gaming the term, not grounding.
2. **Blank-view sensitivity at eval** — accuracy with the evidence view blanked
   should drop *more* for the view-grounded model than for the plain-GRPO one,
   while accuracy with a control view blanked should drop *less*. This is the
   direct test of the claim and does not depend on the training term at all.
3. **Object-reference accuracy** (`vgrl.rewards.object_reference_reward`) — whether
   the model cites the right `<cID,CAMERA,...>`, which it cannot do without having
   looked.

Baselines worth running, all one flag apart: `off` (plain GRPO), `kl1_only`
(PAPO), `literal` (wrong-sign negative control), and the full method.
