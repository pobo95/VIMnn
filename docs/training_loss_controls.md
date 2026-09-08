# Loss normalization and training stability controls

New recipe options connect to existing training implementations. Omitting them
preserves per-structure energy, scales of 1, no gradient clipping and no scheduler.
The resolved run config and checkpoints store the effective values.

```yaml
loss:
  energy_weight: 1.0
  forces_weight: 100.0
  stress_weight: 10.0
  energy_normalization: per_atom
  energy_scale: 1.0
  force_scale: 1.0
  stress_scale: 1.0
training:
  max_epochs: 500
  learning_rate: 1.0e-4
  gradient_clip_norm: 1.0
  scheduler:
    kind: reduce_on_plateau
    factor: 0.5
    patience: 5
    threshold: 1.0e-3
    threshold_mode: rel
    cooldown: 2
    min_lr: 1.0e-6
  early_stopping_patience: 30
  early_stopping_relative_delta: 1.0e-3
```

These are example controls, not dataset-independent optimal hyperparameters.

Energy loss averages `((E_pred-E_ref)/(N*energy_scale))**2` in `per_atom`
mode; `per_structure` omits N. Force/stress losses divide component residuals by
their scale before squaring and use the existing masks and symmetric stress
reduction. Each loss is multiplied by its weight. Thus a scale smaller by 10
increases that term's loss and gradient by 100 at identical weights. Energy
scale is in eV/atom or eV, according to normalization; force scale is eV/angstrom
and stress scale is eV/angstrom^3. Scales must be finite and strictly positive.

Only the energy **objective** is normalized by atom count. Model total energy,
forces as derivatives of total energy, MP neighbor selection and physical RMSE
reporting retain their existing meanings. Loss values from different settings
are not comparable accuracy metrics; compare E/F/S RMSE in physical units.

Choose scales using training labels only. For example, fit atomic baselines on
train, compute RMS energy residual in the selected energy units, and compute
component RMS forces and six independent stresses about zero. Do not use
validation/test labels. Set explicit unit-bearing floors (for example 1e-6 eV/atom,
1e-6 eV/angstrom, 1e-8 eV/angstrom^3) for nearly constant targets and inspect
whether that supervision is useful; tiny scales can amplify noise. No automatic
scale fitting is performed by the recipe. Supply and record the chosen numbers.
Check weighted gradients with `diagnose_training_batch` after choosing scales.
Equal target RMS does not guarantee equal parameter-gradient norms.

`gradient_clip_norm` clips the global gradient AFTER all loss terms are summed.
It limits update inputs but does not balance the terms against each other. Null
or omission disables clipping. The existing training result reports norms before
and after clipping. `scheduler.kind` accepts `none` or `reduce_on_plateau`.
Supported scheduler fields are factor, patience, threshold, threshold_mode,
cooldown, min_lr and eps; monitor/mode must remain total_loss/min, matching
recipe model selection. Patience/cooldown count validation events, not batches.
Allow early stopping enough time for an LR reduction and subsequent recovery;
null disables early stopping. Scheduler threshold and early stopping delta are
separate criteria.

Resume restores optimizer, scheduler and selection state under the saved config.
Changing objective or clipping/scheduler settings creates a different experiment;
start a new run rather than expecting strict resume to accept changed settings.
Old recipe defaults and old serialized run configs remain supported. Deterministic
sample ordering is unchanged; this update does not introduce shuffling.
