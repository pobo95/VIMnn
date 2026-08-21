# Reference-Site Probability-Field MLIP

Independent reference-site MLIP research code.  This repository does not
depend on MACE and does not represent vacancies as atoms. The implemented
milestones contain the typed reciprocal translation gauge and dense balanced
aggregate-vacancy entropic transport. Probability multipoles, equivariant
message passing, and training-pipeline code are intentionally absent.

Runtime baseline: PyTorch 2.6.0+cu118, CUDA 11.8, and e3nn 0.4.4 in eager mode.


PyTorch 2.6 defaults torch.load to weights_only=True, while the trusted
e3nn 0.4.4 package constant file contains the built-in slice type. Importing
refsite_mlip registers only slice as a safe global; it does not disable the
PyTorch safety default or modify site-packages.
