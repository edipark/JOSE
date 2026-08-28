"""Estimator architecture ablation: one teacher, all joints, window 25."""

try:
    from .ablation_runner import main
    from .ablation_catalog import ARCHITECTURE_EXPERIMENTS
except ImportError:
    from ablation_runner import main
    from ablation_catalog import ARCHITECTURE_EXPERIMENTS


if __name__ == "__main__":
    main(ARCHITECTURE_EXPERIMENTS, "architecture")
