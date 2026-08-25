"""Estimator architecture ablation: one teacher, all joints, window 50."""

try:
    from .ablation_runner import main
except ImportError:
    from ablation_runner import main


EXPERIMENTS = (
    ("LSTM_w50_all", "LSTM", 50, "all", 10),
    ("TCN_w50_all", "TCN", 50, "all", 10),
    ("HistoryMLP_w50_all", "HISTORY_MLP", 50, "all", 10),
)


if __name__ == "__main__":
    main(EXPERIMENTS, "architecture")
