import json
import csv
from pathlib import Path

import numpy as np


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return super().default(obj)


class SimpleLogger:
    """
    Minimal experiment logger.

    Saves:
    - arrays to .npz
    - metadata to .json
    - optional CSV exports
    """

    def __init__(self, base_dir="results/logs", run_name="run"):
        self.base_dir = Path(base_dir)
        self.run_name = run_name
        self.run_dir = self.base_dir / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def save_metadata(self, metadata, filename="metadata.json"):
        path = self.run_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, cls=NumpyJSONEncoder)
        return path

    def save_arrays_npz(self, filename="simulation_data.npz", **arrays):
        path = self.run_dir / filename
        np.savez(path, **arrays)
        return path

    def save_time_series_csv(
        self,
        time,
        states,
        controls=None,
        refs=None,
        filename="time_series.csv",
    ):
        path = self.run_dir / filename

        header = [
            "t",
            "x", "y", "z",
            "vx", "vy", "vz",
            "phi", "theta", "psi",
            "p", "q", "r",
        ]

        if controls is not None:
            header += ["T", "tau_x", "tau_y", "tau_z"]

        if refs is not None:
            header += ["x_ref", "y_ref", "z_ref"]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            n = len(time)
            for k in range(n):
                row = [time[k]]
                row += list(states[k])

                if controls is not None:
                    row += list(controls[k])

                if refs is not None:
                    row += list(refs[k])

                writer.writerow(row)

        return path

    def save_figure(self, fig, filename):
        path = self.run_dir / filename
        fig.savefig(path, bbox_inches="tight")
        return path

    def save_figures(self, figures, prefix="figure"):
        paths = []
        for i, fig in enumerate(figures, start=1):
            path = self.save_figure(fig, f"{prefix}_{i}.png")
            paths.append(path)
        return paths