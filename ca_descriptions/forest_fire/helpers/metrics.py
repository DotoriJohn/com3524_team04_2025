import os


def default_burnt_plot_path(filename: str = "burnt_percentages.png") -> str:
    """
    Return a path under the project-level metrics/ folder for saving plots.
    Respects the repository root regardless of current working directory.
    """
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    return os.path.join(root, "metrics", filename)


def default_metrics_csv_path(filename: str = "scalar_metrics.csv") -> str:
    """Return a path under the project-level metrics/ folder for saving scalar metrics."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    return os.path.join(root, "metrics", filename)


def compute_containment_efficiency(
    burnt_fraction: float, first_ignition_time: int | None, final_timestep: int
) -> float | None:
    """
    Simple containment efficiency proxy:
        average burnt fraction per timestep since first ignition.
    Lower values indicate better containment (less burned per unit time).
    """
    if first_ignition_time is None:
        return None
    duration = max(final_timestep - first_ignition_time + 1, 1)
    return (burnt_fraction / duration) * 100


def save_scalar_metrics(metrics: dict, save_path: str | None = None):
    """Persist scalar metrics to a CSV with columns metric,value."""
    if save_path is None:
        save_path = default_metrics_csv_path()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    lines = ["metric,value"]
    for key, val in metrics.items():
        lines.append(f"{key},{val}")
    with open(save_path, "w", encoding="ascii") as f:
        f.write("\n".join(lines))
    return save_path


def plot_burnt_percentages(stats, save_path=None):
    """Create a matplotlib plot from the collected burnt percentage stats (BURNT only)."""
    if not stats:
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not installed; skipping burn plot.")
        return None

    timesteps = [entry["timestep"] for entry in stats]
    terrain_keys = [k for k in stats[0].keys() if k != "timestep"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for key in terrain_keys:
        percents = [entry.get(key, 0) * 100 for entry in stats]
        ax.plot(timesteps, percents, label=key.replace("_", " ").title())

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Burnt (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Burnt progression by terrain")
    ax.legend(loc="upper left")
    ax.grid(True, linewidth=0.5, alpha=0.4)
    fig.tight_layout()

    if save_path is None:
        save_path = default_burnt_plot_path()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)

    return fig
