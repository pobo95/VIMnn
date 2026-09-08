# OT entropy and training Newton-Krylov

```yaml
ot_solver:
  epsilon_ot: 0.5
  train_sinkhorn_iterations: 256
  training: newton_krylov
  newton:
    warmup_iterations: 16
    max_newton_iterations: 30
    pcg_max_iterations: 256
  inference: sinkhorn_newton_krylov
```

Training accepts `sinkhorn` or `newton_krylov`. Entropy is shared by training and
inference. `train_sinkhorn_iterations` controls pure Sinkhorn training only;
Newton uses `newton.warmup_iterations` instead. Omitting entropy/iteration options
preserves epsilon=0.5 and 256 Sinkhorn iterations. Epsilon must be finite and
positive. Pure Sinkhorn counts must be positive integers; Newton warmup may be 0.

Newton uses a deterministic zero initial dual, Sinkhorn warmup, Newton-PCG with
Armijo line search, and two differentiable root corrections. Accepted updates
remain in the autograd graph. Stopping and line-search decisions select a branch;
the decisions themselves are not differentiated. Failure to converge raises an
error before the optimizer update; it does not silently substitute Sinkhorn.
The existing TRAIN_FIXED name is retained for the fixed training **phase** path
and checkpoint API. It no longer implies fixed-count OT when Newton is selected.
The inference-only adaptive phase path still rejects create_graph=True.

The two final corrections use dense gauge-fixed linear solves. They preserve
first/second root sensitivities even when the primal solver stops immediately at
an already-converged state. Separate support-component gauge null directions are
fixed. OT arithmetic uses float64 internally, with differentiable casts back to
the model dtype and marginal residuals checked on the returned plan. Network and
phase dtypes remain as configured. This is currently a **dense transport backend**
implementation; edge-list Newton training is rejected explicitly. Memory and cost
of the final dual-system solves grow with the number of atoms and reference sites.
`newton_iterations` diagnostics include the two final corrections; `cg_iterations`
counts PCG work in the preceding Newton solver.

Optional Newton controls:

| Field | Default |
|---|---|
| warmup_iterations | 16 |
| max_newton_iterations | 30, before the two final corrections |
| pcg_max_iterations | 256 per Newton step |
| convergence_tolerance | null: 1e-6 for float32, 1e-11 for float64 |
| pcg_absolute_tolerance | null: 1e-8 for float32, 1e-13 for float64 |
| pcg_relative_tolerance | null: 1e-6 for float32, 1e-11 for float64 |

Explicit tolerances must be finite and positive. The defaults above follow the
model input dtype even though the OT internal arithmetic uses float64. Requesting
a marginal tolerance finer than the returned dtype can represent may fail.
Compiled configs and bundles store the selected training solver and all Newton
settings. Legacy Sinkhorn config bytes remain unchanged when new fields are
omitted. Changing solver or entropy is a new experiment, not a compatible strict
resume of old training. Startup logs report the selected training OT solver.

`inference: sinkhorn_newton_krylov` retains the existing adaptive inference hybrid
preference and its automatic qualification workflow. The inference preference
is applied at prediction/evaluation call time; using the fixed training path on
a loaded Newton model still uses its stored training Newton settings.

Epsilon changes the regularized OT problem, not the MP cutoff or convergence
tolerance. Smaller epsilon can sharpen vacancy localization but requires measured
convergence and derivative checks. A converged OT solve alone does not establish
better energy/force accuracy. Start from a baseline epsilon, then compare controlled
experiments on pristine, vacancy and strained structures.
