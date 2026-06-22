"""temvirus — clean, faithful reimplementation of the multi-protocol TEM virus
classification pipeline.

This package is the *reproduction code* for the study. It re-expresses, in a
structured and runnable form, exactly the method recorded in the two research
notebooks under ``notebooks/`` (which remain the experimental log). Nothing here
changes the methodology, hyperparameters, or results: every constant and routine
is ported from the notebooks (the canonical CP3.6 enhanced pipeline).

Modules
-------
- ``config``   : dataclasses + the canonical hyperparameters and protocol/seed map
- ``seed``     : deterministic seeding
- ``data``     : dataset, transforms, dataloaders, manifest handling
- ``models``   : timm model builder (DenseNet201, EfficientNetV2-S)
- ``mixup``    : mixup data + loss
- ``tta``      : 4-view test-time augmentation
- ``metrics``  : accuracy / macro-F1 / etc.
- ``engine``   : scheduler, train/eval loops, the ``fit`` driver, checkpointing
- ``ensemble`` : softmax combination strategies + validation-based selection
- ``stats``    : paired deltas, bootstrap CI, Wilcoxon, Cohen's d, Bonferroni
- ``splits``   : protocol definitions + manifest leakage validation
"""

__version__ = "1.0.0"

from . import config, seed, data, models, mixup, tta, metrics, engine, ensemble, stats, splits  # noqa: F401
