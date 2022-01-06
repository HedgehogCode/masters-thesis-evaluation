import os
import pandas as pd
import numpy as np


def load_folder(folder, column_name):
    dfs = []
    for f in os.listdir(folder):
        df = pd.read_csv(os.path.join(folder, f))
        df[column_name] = f[:-4]  # File name without extension
        dfs.append(df)
    return pd.concat(dfs, axis=0)

def float_format(show_leading_zero=True):
    def f(v):
        if v < 1.0:
            if show_leading_zero:
                return f"{v:.4f}"
            else:
                return f"{v:.4f}"[1:]
        else:
            return f"{v:.2f}"

    return f


def mark_column_best_formatter(df, column_name, mark_max=True, num_decimals=2):
    worst_value = -np.inf if mark_max else np.inf
    value_col = df[column_name].map(lambda x: worst_value if np.isnan(x) else x)
    sorted_values = list(value_col.sort_values(ascending=not mark_max))

    def formatter(x):
        if np.isnan(x):
            # NaN
            return "---"
        elif x == sorted_values[0]:
            # Best value
            return f"\\bst{{{x:.{num_decimals}f}}}"
        elif x == sorted_values[1]:
            # Second best value
            return f"\\snd{{{x:.{num_decimals}f}}}"
        else:
            # Simple value
            return f"{x:.{num_decimals}f}"

    return formatter


def add_midrule(latex, index):
    lines = latex.splitlines()
    lines.insert(index, "\midrule")
    return "\n".join(lines)


def sort_key_for(l):
    if isinstance(l, dict):
        l = list(l.keys())
    return lambda x: [l.index(v) for v in x]


def delete_line(latex, index):
    lines = latex.splitlines()
    lines = lines[:index] + lines[(index + 1) :]
    return "\n".join(lines)