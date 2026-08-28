"""Estimator-compatible variant of the G1 29-DOF walk environment.

This is :mod:`walk_env_cfg` with exactly one change: the ``base_lin_vel``
observation term is prepended to the *policy* group. The base walk task keeps
base linear velocity privileged (critic-only), so there is no slot for a state
estimator to write into; this variant gives the estimator a real injection slot
while leaving rewards, actions, commands, events, terminations, curriculum and
every algorithm hyperparameter untouched.

The added term is noiseless on purpose: it is the privileged quantity the
estimator is trained to reproduce, mirroring how the AMP and Direct-PPO teachers
in JOSE expose their privileged base state.
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from .walk_env_cfg import G1WalkEnvCfg, G1WalkPlayEnvCfg, ObservationsCfg


@configclass
class EstimatorPolicyCfg(ObsGroup):
    """Policy observations with the estimated base linear velocity in front.

    Term order defines the flattened layout, so ``base_lin_vel`` is declared
    first and every following term keeps its original relative order.
    """

    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
    projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
    joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
    last_action = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        self.history_length = 5
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class EstimatorObservationsCfg(ObservationsCfg):
    """Same observation groups as the walk task, with the extended policy group."""

    policy: EstimatorPolicyCfg = EstimatorPolicyCfg()


@configclass
class G1WalkEstimatorEnvCfg(G1WalkEnvCfg):
    observations: EstimatorObservationsCfg = EstimatorObservationsCfg()


@configclass
class G1WalkEstimatorPlayEnvCfg(G1WalkPlayEnvCfg):
    observations: EstimatorObservationsCfg = EstimatorObservationsCfg()
