"""Manager-based G1 29-DOF PPO walk task and its rsl-rl training stack.

Ported from unitree_rl_lab (Apache License 2.0),
https://github.com/unitreerobotics/unitree_rl_lab
"""

# Registers the friction-randomized and sloped task ids. Kept here rather than in
# the package ``__init__.py`` because that file is hashed into
# ``TASK_IMPLEMENTATION`` for all four tasks (``ablation_catalog.py``), so adding
# unrelated ids to it would change the implementation digest of every study
# already recorded and refuse a resume on any of them. This file is in no
# fingerprint tuple.
#
# Safe to run before ``AppLauncher``: ``terrain_tasks`` imports gymnasium and
# nothing else, registering by entry-point *string* exactly as
# ``jose/__init__.py`` does, so no Isaac Lab module is pulled in at import time.
from . import terrain_tasks  # noqa: F401,E402
# The push-disturbed ids, kept out of ``terrain_tasks`` because that file is
# hashed into the two terrain tuples. Same import-time guarantee as above.
from . import push_tasks  # noqa: F401,E402
