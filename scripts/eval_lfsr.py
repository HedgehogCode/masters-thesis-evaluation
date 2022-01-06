import importlib
import os
import sys
import argparse

sys.path.append(os.path.normpath(os.path.join(__file__, "..", "..")))
import eval_utils as utils

SEED = 99


def main(args):
    method = importlib.import_module(args.method)
    superresolver, _ = method.lfsr(args.test_run, args.method_args)

    # Evaluation
    utils.eval_lfsr(
        superresolver=superresolver,
        csv_out=args.result,
        seed=SEED,
        test_run=args.test_run,
    )


def parse_args(arguments):
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "result",
        help="Path to the file where results will be written to.",
        type=argparse.FileType("w"),
    )
    parser.add_argument(
        "method",
        help="Module name of the method implementation.",
        type=str,
    )
    parser.add_argument(
        "method_args",
        help="Arguments for the method implementation (one string).",
        type=str,
        nargs="?",
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Test run: Run each task only for a few steps.",
    )
    return parser.parse_args(arguments)


if __name__ == "__main__":
    sys.exit(main(parse_args(sys.argv[1:])))
