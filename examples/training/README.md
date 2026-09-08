# Training examples

These YAML files are templates for your own periodic reference structures and
labeled extended-XYZ data. No training dataset is bundled here. Copy a recipe to
your experiment directory, prepare its input files, and choose an output directory
that does not already exist. Relative paths are resolved beside the YAML file.

| Recipe | Training OT | Runtime | Energy objective | Stability controls |
|---|---|---|---|---|
| [minimal.yaml](minimal.yaml) | Sinkhorn, 256 iterations | CPU / float64 | Per structure, scales=1 | Public defaults; no clipping or scheduler |
| [advanced.yaml](advanced.yaml) | Newton-Krylov, 16 Sinkhorn warmup steps | CUDA / float32 | Per atom, scales=1 | Clipping, plateau scheduler, relative early stopping |

The minimal recipe shows the defaults, including an energy weight of 1, force
weight of 100 and disabled stress supervision. The advanced recipe illustrates
explicit controls and two reference bindings. Its values are a starting point,
not demonstrated optimal settings for every material. For a controlled solver
comparison, copy one recipe and change only the training solver and output path;
comparing minimal against advanced would also change model size, loss and runtime.

## Prepare the input files

For `minimal.yaml`:

```text
experiment/
  minimal.yaml
  references/POSCAR
  data/train.xyz
  data/validation.xyz
```

For `advanced.yaml`:

```text
experiment/
  advanced.yaml
  references/POSCAR_222
  references/POSCAR_333
  data/train_pristine_222.xyz
  data/train_vacancy_222.xyz
  data/train_333.xyz
  data/validation_222.xyz
  data/validation_333.xyz
```

Each POSCAR describes a fully periodic reference structure. Several pristine and
vacancy data files may share one reference. Use the matching reference for each
supercell and composition; reference aliases such as `POSCAR_222` are chosen by
the author. An incompatible binding is rejected rather than reassigned.

Extended-XYZ frames need the labels enabled by the loss: energy, forces, and
stress when its weight is positive. Internal units are angstrom, eV, eV/angstrom
and eV/angstrom^3; stress uses the tensile-positive convention. `fit_full_rank`
fits the atomic baseline from training labels and requires identifiable species
baselines. Check data splits independently: similar structures on both sides of
a split can make validation overoptimistic.

## Validate, then train

Use an environment with the current repository installed. From the directory
containing `experiment/`, run:

```bash
refsite-mlip resolve-train-config experiment/minimal.yaml --dry-run --json
refsite-mlip validate-train-config experiment/minimal.yaml --json
refsite-mlip train experiment/minimal.yaml --dry-run --quiet --json
```

Replace `minimal.yaml` with `advanced.yaml` for the Newton example. Add
`--device cpu` to these commands to validate on CPU when CUDA is unavailable.
Validation reads the inputs and can perform reference/inference qualification;
it does not create model parameters, an optimizer, or the output directory.
Qualification can therefore take time even in a dry run.

After reviewing the effective config and validation result, start training:

```bash
refsite-mlip train experiment/minimal.yaml
```

For the advanced example, use `experiment/advanced.yaml` instead. Start each new
experiment with a new `output_directory`. Changing solver, entropy or loss settings
is not a compatible strict resume of a previous checkpoint.

## OT controls

- `epsilon_ot` changes entropy regularization, shared by training and inference.
  Smaller values can localize assignments, but require convergence and derivative
  checks. The examples use 0.5.
- `training: sinkhorn` uses `train_sinkhorn_iterations`. These are differentiable
  fixed-count updates; the omitted default is 256.
- `training: newton_krylov` uses `newton.warmup_iterations` followed by Newton-PCG
  and two differentiable root corrections. It does not use the pure Sinkhorn
  iteration count. Training failure raises an error without a silent fallback.
- Newton training currently requires dense OT. Its OT arithmetic is float64
  internally, including the final dense linear solves, while returned tensors
  follow the model dtype. Large reference systems can be costly.
- `inference: sinkhorn_newton_krylov` is the separate adaptive inference preference
  and requires dataset-local qualification. It is applied at prediction/evaluation
  call time. Evaluating a Newton-trained model through its fixed training path
  still uses the stored training Newton solver.

See [OT settings and derivative behavior](../../docs/training_ot_controls.md)
for tolerances, iteration accounting and supported configurations.

## Geometry and loss normalization

`r_ot` is the atom-to-reference OT support cutoff. `r_mp` is the reference-site
message-passing cutoff. MP radial inputs are normalized using `r_mp` automatically;
do not add `edge_length_scale` to the recipe. This changes distance features,
not which sites communicate within the specified cutoff.

`per_atom` divides the total-energy error by atom count before squaring. It does
not divide the model energy used to calculate forces. Loss scales divide residuals
before squaring: decreasing a scale by ten increases that term's weighted loss
and gradient by one hundred at identical parameters. Estimate any non-neutral
scales from training labels only, then inspect the weighted gradients.

Clipping limits the combined gradient; it does not balance energy against forces.
The scheduler monitors validation total loss. With `threshold_mode: rel` and
`threshold: 1e-3`, an improvement must exceed 0.1% of the scheduler's best loss.
Early stopping has its own improvement threshold and patience; allow time for
recovery after a learning-rate reduction.

Compare physical energy/force/stress RMSE across experiments, not raw total losses
with different scales. The displayed training epoch metric is accumulated before
individual optimizer updates, while validation evaluates fixed epoch-end weights.

Further guidance: [loss controls](../../docs/training_loss_controls.md) and
[batch activation/gradient diagnostics](../../docs/training_diagnostics.md).
