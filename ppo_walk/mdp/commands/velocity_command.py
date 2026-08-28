"""Ported verbatim from unitree_rl_lab (Apache License 2.0).

Source: https://github.com/unitreerobotics/unitree_rl_lab
Only import paths were adjusted for JOSE; behaviour is unchanged.
"""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.utils import configclass


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING
