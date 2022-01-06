import os
import sys
import argparse

import tensorflow as tf

sys.path.append(os.path.normpath(os.path.join(__file__, "..")))
from func_model_81 import define_LFattNet


def main(args):
    # Define the model
    model = define_LFattNet(None, None, list(range(9)), 0.001)

    # Load the weights
    model.load_weights(
        os.path.normpath(os.path.join(__file__, "..", "pretrain_model_9x9.hdf5"))
    )

    # Save the model to a h5 file with architecture and weights
    tf.keras.models.save_model(
        model, args.output.name, include_optimizer=False, save_format="h5"
    )


def parse_args(arguments):
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "output",
        help="Path to the output file.",
        type=argparse.FileType("w"),
    )
    return parser.parse_args(arguments)


if __name__ == "__main__":
    sys.exit(main(parse_args(sys.argv[1:])))
