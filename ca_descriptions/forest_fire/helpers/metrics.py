import os


def default_burnt_plot_path(filename: str = "burnt_percentages.png") -> str:
    """
    Return a path under the project-level metrics/ folder for saving plots.
    Respects the repository root regardless of current working directory.
    """
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    return os.path.join(root, "metrics", filename)


def plot_burnt_percentages(stats, save_path=None):
    """Create a matplotlib plot from the collected burnt percentage stats."""
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
    ax.set_ylabel("Burning/Burnt (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Burn progression by terrain")
    ax.legend(loc="upper left")
    ax.grid(True, linewidth=0.5, alpha=0.4)
    fig.tight_layout()

    if save_path is None:
        save_path = default_burnt_plot_path()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)

    return fig
