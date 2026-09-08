# Reference-Site Probability-Field MLIP

Independent reference-site MLIP research code.  This repository does not
depend on MACE and does not represent vacancies as atoms. The implemented
components include the typed reciprocal translation gauge, balanced aggregate-
vacancy entropic transport, probability multipoles, equivariant message passing,
and a training pipeline with checkpoint/resume support. Training OT supports
Sinkhorn and Newton-Krylov; Newton training currently uses the dense backend.

Start with the [training examples and setup guide](examples/training/README.md).
The [minimal recipe](examples/training/minimal.yaml) demonstrates CPU/Sinkhorn
defaults; the [advanced recipe](examples/training/advanced.yaml) demonstrates
Newton training, multiple references, and loss/optimizer controls. Supply your own
reference structures and labeled data before running either example.

Runtime baseline: PyTorch 2.6.0+cu118, CUDA 11.8, and e3nn 0.4.4 in eager mode.


PyTorch 2.6 defaults torch.load to weights_only=True, while the trusted
e3nn 0.4.4 package constant file contains the built-in slice type. Importing
refsite_mlip registers only slice as a safe global; it does not disable the
PyTorch safety default or modify site-packages.
