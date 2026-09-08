# Training batch diagnostics

Call this opt-in function on a small representative batch before training, or on
an instantiated checkpoint. It does not take an optimizer or update parameters.

```python
import json
from refsite_mlip.training.diagnostics import diagnose_training_batch

report = diagnose_training_batch(
    model, batch, template_contexts, loss_config=loss_config,
)
with open('batch-diagnostics.json', 'w') as stream:
    json.dump(report, stream, indent=2, allow_nan=False)
```

Pass the same `StructureBatch`, template contexts and `LossConfig` used by
`train_step`. The solver is `TRAIN_FIXED`. Existing parameter `.grad` tensors,
module training flags and PyTorch RNG state are preserved. Run separately from
an optimizer step. Computing force/stress parameter gradients needs higher
order derivatives and can use substantially more time and memory than energy
inference; start with one vacancy and one pristine structure.

- `activations`: hidden-state input/output RMS, absolute maximum and nonfinite
  count for each MP layer, plus encoders and readout branches. Values pool all
  tensor elements across structures, not per-structure averages.
- `terms`: mean loss after configured normalization/scales, weighted loss, and
  global L2 norm of that weighted term's gradients over trainable parameters.
  Only positive-weight terms with valid labels are included. Parameters with
  `None` gradients are counted separately from connected zero gradients.
- `total`: gradient norm of the actual summed objective. Individual norms do
  not add because gradients can align or cancel.

Nonfinite values are represented by null and finite/count fields. A large energy
term gradient relative to force/stress suggests checking initialization and loss
scales. Rapid layer-to-layer activation growth suggests checking radial inputs
and residual blocks. Neither measurement alone demonstrates better trained
accuracy. Compare identical weights/data with one controlled change at a time.

Old bundles retain their serialized radial scale. A new recipe deriving the
scale from `r_mp` does not retroactively change an old checkpoint.
