"""Estimator joint-scope ablation: one teacher, LSTM, window 50."""

try:
    from .ablation_runner import main
except ImportError:
    from ablation_runner import main


EXPERIMENTS = (
    ("LSTM_w50_all", "LSTM", 50, "all", 10),
    ("LSTM_w50_legs", "LSTM", 50, "legs", 10),
    ("LSTM_w50_upper", "LSTM", 50, "upper", 10),
)


if __name__ == "__main__":
    main(EXPERIMENTS, "joint_scope")
