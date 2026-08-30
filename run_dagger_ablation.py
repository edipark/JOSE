"""DAgger-rounds ablation: how much the on-policy phase adds to the warm start."""

try:
    from .ablation_runner import main
    from .ablation_catalog import DAGGER_EXPERIMENTS
except ImportError:
    from ablation_runner import main
    from ablation_catalog import DAGGER_EXPERIMENTS


if __name__ == "__main__":
    main(DAGGER_EXPERIMENTS, "dagger")
