# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import os
import shutil

import click
import h5py
import numpy as np

from spectre.IO.H5 import available_subfiles


def combine_h5_dat(h5files, output, wipe_nonmonotonic_times, force):
    """Combines multiple HDF5 dat files

    This executable is used for combining a series of HDF5 files, each
    containing one or more dat files, into a single HDF5 file. A typical
    use case is to join dat-containing HDF5 files from different segments
    of a simulation, with each segment containing values of the dat files
    during different time intervals.

    \f
    Arguments:
      h5files: List of H5 dat files to join
      output: Output filename. An extension '.h5' will be added if not present.
      wipe_nonmonotonic_times: If specified, wipe non-monotonically increasing
       times in favor for later times from h5 file.
      force: If specified, overwrite output file if it already exists
    """
    # Copy first input file to output file
    if not output.endswith(".h5"):
        output += ".h5"
    # If output file exists, exit unless the user specifies `--force`
    if os.path.exists(output) and not force:
        raise ValueError(f"File '{output}' exists; to overwrite, use --force")
    shutil.copy(h5files[0], output)

    # Open the output file for appending
    with h5py.File(output, "r+") as out:
        # Get a list of all dat file keys (keys ending in ".dat")
        dat_file_keys = available_subfiles(out, extension=".dat")

        # Loop over remaining input files, appending each dat file
        for input_file in h5files[1:]:
            with h5py.File(input_file, "r") as input:
                for dat_file_key in dat_file_keys:
                    if dat_file_key in input.keys():
                        data_to_append = input[dat_file_key]
                        start_size = out[dat_file_key].shape[0]
                        append_size = input[dat_file_key].shape[0]
                        out[dat_file_key].resize(
                            start_size + append_size, axis=0
                        )
                        out[dat_file_key][start_size:] = input[dat_file_key]
                    else:
                        logging.warning(
                            f"CombineH5Dat: Dat file '{dat_file_key}'"
                            f" not found in input file '{input_file}'"
                        )
                if wipe_nonmonotonic_times:
                    data = out[dat_file_key][:]
                    mask = np.zeros(len(data), dtype=bool)
                    last_time = data[-1, 0]
                    for i in range(len(data) - 2, -1, -1):
                        current_time = data[i, 0]
                        if current_time < last_time:
                            mask[i] = True
                            last_time = current_time
                    wiped_data = data[mask]
                    del out[dat_file_key]
                    out.create_dataset(dat_file_key, data=wiped_data)


@click.command(name="combine-h5-dat", help=combine_h5_dat.__doc__)
@click.argument(
    "h5files",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    nargs=-1,
    required=True,
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(
        exists=False,
        file_okay=True,
        dir_okay=False,
        writable=True,
        readable=True,
    ),
    help="Combined output filename.",
)
@click.option(
    "--wipe-nonmonotonic-times",
    "-w",
    is_flag=True,
    help="Wipe non-monotonic times in favor for later times from h5 file.",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="If the output file already exists, overwrite it.",
)
def combine_h5_dat_command(**kwargs):
    combine_h5_dat(
        kwargs["h5files"],
        kwargs["output"],
        kwargs["wipe_nonmonotonic_times"],
        kwargs["force"],
    )


if __name__ == "__main__":
    combine_h5_dat_command(help_option_names=["-h", "--help"])
