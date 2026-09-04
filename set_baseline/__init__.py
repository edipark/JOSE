"""SET baseline: State Estimation Transformers for Agile Legged Locomotion.

A from-scratch implementation of Yu et al., IROS 2024 (arXiv:2410.13496). No code
was released by the authors, and the paper specifies only three numbers -- six
transformer blocks, context length H=20, and an MSE loss -- so every other
hyperparameter here is ours and is recorded in the run metadata.

The package is deliberately self-contained. `ablation_catalog.py` fingerprints an
explicit list of files, and any byte change to one of them invalidates the JOSE
dataset caches and marks previously logged results as a different variant.
Nothing in this package appears in that list, and nothing here modifies a file
that does: the pieces it needs from `estimator/` are imported and subclassed.
"""
