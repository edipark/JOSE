"""Estimator joint-scope ablation: one teacher, LSTM, window 25."""

try:
    from .ablation_runner import main
    from .ablation_catalog import JOINT_SCOPE_EXPERIMENTS
except ImportError:
    from ablation_runner import main
    from ablation_catalog import JOINT_SCOPE_EXPERIMENTS


if __name__ == "__main__":
    main(JOINT_SCOPE_EXPERIMENTS, "joint_scope")
