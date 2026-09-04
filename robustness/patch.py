"""Install encoder noise for the whole of a training run, without editing a trainer.

``train_state_estimator.py`` is fingerprinted -- its bytes feed both the variant
identity and the dataset cache key in ``ablation_catalog.py`` -- so an
``--encoder-noise`` flag there would give every future run a new identity and cut
it loose from every result already logged. ``train_history_student.py`` is not
fingerprinted and could take a flag, but then the two trainers would have two
implementations of one idea, and the encoder-axis arms have to be degraded
*identically* or the axis is not an axis.

So both go through here: patch the environment classes before the trainer
imports anything that reads them, then hand control to the unmodified trainer.

Two patches, matching the two places encoder error has to appear (see
``robustness/noise.py`` for why either alone is not a robot):

1. ``EstimatorPolicyCfg.__post_init__`` -- the observation manager's joint terms,
   so the teacher acts on degraded encoders while it is being distilled from.
2. ``G1WalkEstimatorEnv`` accessors -- so estimators and students are trained on
   the degraded joint signal they will be evaluated with.

Locomotion only, which is the scope of the robustness section: the AMP
environments have no encoder-noise story to tell here.
"""

from __future__ import annotations


def arm_encoder_noise(scale: float) -> None:
    """Arm the patches now; apply them the moment the simulator is up.

    The environment classes cannot be imported before ``AppLauncher`` runs --
    they pull in ``pxr``, which only exists inside a live Omniverse app -- so
    patching them directly from a wrapper script fails at import. But
    ``isaaclab.app`` itself imports fine beforehand, and every trainer's first
    act is to construct an ``AppLauncher``. Wrapping that constructor gives us
    the one moment that is both after the simulator exists and before the trainer
    has built a config or an environment.
    """
    from isaaclab.app import AppLauncher

    if scale <= 0.0:
        raise ValueError("scale must be positive; the clean arm is the unpatched trainer")

    original_launcher_init = AppLauncher.__init__

    def patched_launcher_init(self, *args, **kwargs):
        original_launcher_init(self, *args, **kwargs)
        # Restore first: a trainer that builds a second launcher must not stack
        # a second set of environment patches on top of the first.
        AppLauncher.__init__ = original_launcher_init
        _install(scale)

    AppLauncher.__init__ = patched_launcher_init
    print(f"[encoder-noise] armed at scale {scale}; applies once the app is up", flush=True)


def _as_env_cfg(policy_group):
    """Present a policy observation group as the env config shape."""
    observations = type("_Observations", (), {"policy": policy_group})()
    return type("_EnvCfg", (), {"observations": observations})()


def _install(scale: float) -> None:
    """Degrade every encoder reading this process will make. Needs a live app."""
    from jose.ppo_walk.walk_estimator_env import G1WalkEstimatorEnv
    from jose.ppo_walk.walk_estimator_env_cfg import EstimatorPolicyCfg

    from .noise import EncoderCorruptor, EncoderNoiseCfg, apply_encoder_noise_cfg

    cfg = EncoderNoiseCfg(scale=scale)

    original_post_init = EstimatorPolicyCfg.__post_init__

    def patched_post_init(self):
        original_post_init(self)
        # apply_encoder_noise_cfg takes the env config and reads
        # `.observations.policy`; here `self` is already that policy group, so
        # wrap it in the two levels the function expects.
        apply_encoder_noise_cfg(_as_env_cfg(self), scale)

    EstimatorPolicyCfg.__post_init__ = patched_post_init

    original_joint_state = G1WalkEstimatorEnv.get_estimator_joint_state
    original_sensor_state = G1WalkEstimatorEnv.get_distillation_sensor_state
    # One corruptor per environment instance, built lazily: the trainer creates
    # the environment after these patches are installed, so the batch size is not
    # known yet. Keyed by id() rather than stored on the env to avoid adding an
    # attribute a fingerprinted class does not declare.
    corruptors: dict[int, EncoderCorruptor] = {}

    def corruptor_for(env, velocity):
        key = id(env)
        if key not in corruptors:
            corruptors[key] = EncoderCorruptor(
                velocity.shape[0], velocity.shape[1], velocity.device, cfg
            )
        return corruptors[key]

    def patched_joint_state(self):
        position, velocity, names = original_joint_state(self)
        position, velocity = corruptor_for(self, velocity).corrupt(position, velocity)
        return position, velocity, names

    def patched_sensor_state(self):
        state = dict(original_sensor_state(self))
        corruptor = corruptor_for(self, state["joint_velocity"])
        state["joint_position"], state["joint_velocity"] = corruptor.corrupt(
            state["joint_position"], state["joint_velocity"]
        )
        return state

    G1WalkEstimatorEnv.get_estimator_joint_state = patched_joint_state
    G1WalkEstimatorEnv.get_distillation_sensor_state = patched_sensor_state
    print(f"[encoder-noise] installed at scale {scale}: {cfg.describe()}", flush=True)
