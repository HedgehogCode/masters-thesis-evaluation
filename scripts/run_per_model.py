from __future__ import print_function
import os
import sys
import re
import glob
import subprocess


def has_noise_range(name):
    groups = re.match(r".*?(\d\.\d+)?-?(\d\.\d+)", name)
    return groups[1] is not None


if __name__ == "__main__":
    idx = 1
    if sys.argv[idx] == "only_noise_range":
        only_noise_range = True
        idx += 1
    else:
        only_noise_range = False

    folder = sys.argv[idx]
    idx += 1

    script = sys.argv[idx]
    idx += 1
    script_args = sys.argv[idx:]

    model_paths = [
        p
        for p in sorted(glob.glob(os.path.join(folder, "*.h5")))
        if not only_noise_range or has_noise_range(p)
    ]
    model_names = [os.path.basename(p)[:-3] for p in model_paths]

    for mn, mp in zip(model_names, model_paths):
        script_args_m = [a.replace("{mn}", mn).replace("{mp}", mp) for a in script_args]
        arguments = ["python", script, *script_args_m]
        print(f"Running '{' '.join(arguments)}'")
        subprocess.call(arguments)
