# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import re
from pathlib import Path
from typing import Optional, Union

import click
import numpy as np

import spectre.IO.H5 as spectre_h5
from spectre.IO.H5.CombineH5Dat import combine_h5_dat
from spectre.support.DirectoryStructure import PipelineStep, list_pipeline_steps
from spectre.support.Schedule import schedule, scheduler_options

logger = logging.getLogger(__name__)

CCE_INPUT_FILE_TEMPLATE = Path(__file__).parent / "Cce.yaml"


# Helper function that extracts the segment number of a path as an integer.
# Used to ensure segments are combined in the proper order.
def segment_number(path):
    match = re.search(r"Segment_(\d+)", str(path))
    if match is None:
        raise ValueError(f"Could not find segment number in {path}")
    return int(match.group(1))


def cce_input(
    bondisachs_data: Optional[Union[str, Path]] = None,
    inspiral_run_dir: Optional[Union[str, Path]] = None,
    ringdown_run_dir: Optional[Union[str, Path]] = None,
) -> dict:
    """Generate the input for the CCE pipeline.

    This input will be used to fill the '--cce-input-file-template'.
    'inspiral_run_dir' and 'ringdown_run_dir' are passed for bookeeping reasons,
    as the yaml only reads from 'bondisachs_data'.

    Arguments:
        bondisachs_data: Path to the bondisachs data file generated from a bbh
        simulation.
        inspiral_run_dir: Directory containing the segments of the inspiral run.
        ringdown_run_dir: Directory containing the segments of the ringdown run.
    """

    return {
        "BondiSachsData": bondisachs_data,
        "InspiralRunDirectory": (
            str(Path(inspiral_run_dir).resolve())
            if inspiral_run_dir is not None
            else None
        ),
        "RingdownRunDirectory": (
            str(Path(ringdown_run_dir).resolve())
            if ringdown_run_dir is not None
            else None
        ),
    }


def run_cce(
    bondisachs_data: Optional[Union[str, Path]] = None,
    inspiral_run_dir: Optional[Union[str, Path]] = None,
    ringdown_run_dir: Optional[Union[str, Path]] = None,
    extraction_radius: Optional[int] = None,
    cce_input_file_template: Union[str, Path] = CCE_INPUT_FILE_TEMPLATE,
    pipeline_dir: Optional[Union[str, Path]] = None,
    run_dir: Optional[Union[str, Path]] = None,
    segments_dir: Optional[Union[str, Path]] = None,
    **scheduler_kwargs,
):
    """Extract partial or full waveforms from a simulation.

    Point the inspiral_run_dir and ringdown_run_dir to the directories
    containing the segments of the inspiral and ringdown run, respectively, for
    a given bbh simulation for a full waveform extraction. In this case, the
    extraction radius, if left unspecified, will default to the largest radius
    available. For partial waveforms you can point bondisachs_data to the
    BondiSachs file from a given segment in either directory. Here, it's
    important that the filename of the BondiSachs data is in the form
    NameOfFileRXXXX.h5 The remaining options are forwarded to the 'schedule'
    command. See 'schedule' docs for details.

    Arguments:
        bondisachs_data: Path to the bondisachs data generated from a bbh run.
        inspiral_run_dir: Directory containing the segments of the inspiral run.
        ringdown_run_dir: Directory containing the segments of the ringdown run.
        extraction_radius: Extraction radius for CCE in units of the total mass.
        Specifying this option is only necessary when pointing into a directory
        with multiple BondiSachs files. Left unspecified, the default will be
        the largest radius available. When pointing to a single file, the file
        must be in the form NameOfFileRXXXX.h5, where the last 4 digits are the
        extraction radius.
        pipeline_dir: Directory where steps in the pipeline are created. If not
        specified, a temporary directory is used that is deleted after the
        pipeline finishes.
        run_dir: Directory where the CCE pipeline is run. If not specified, the
        pipeline is run in the current working directory.
        cce_input_file_template: Input file template for CCE. This should be a
        yaml file that defines the steps in the CCE pipeline.
    """
    logger.warning(
        "The BBH pipeline is still experimental. Please review the generated"
        " input files."
    )

    # Resolve directories
    if pipeline_dir:
        pipeline_dir = Path(pipeline_dir).resolve()
    if pipeline_dir and not run_dir:
        pipeline_steps = list_pipeline_steps(pipeline_dir)
        if pipeline_steps:
            segments_dir = pipeline_steps[-1].next(label="Cce").path
        else:
            segments_dir = PipelineStep.first(
                directory=pipeline_dir, label="Cce"
            ).path

    if not any([inspiral_run_dir, ringdown_run_dir, bondisachs_data]):
        raise ValueError(
            "No Bondi-Sachs data provided, and no inspiral or ringdown run"
            " directories provided to generate Bondi-Sachs data for CCE."
            " Specify at least one of these to run CCE."
        )
    elif bondisachs_data and any([inspiral_run_dir, ringdown_run_dir]):
        raise ValueError(
            "Cannot provide both bondisachs_data and inspiral_run_dir or"
            " ringdown_run_dir. Provide either the bondisachs_data directly, or"
            " the inspiral and/or ringdown run directories to generate the"
            " bondisachs_data for CCE."
        )
    elif bondisachs_data and not any([inspiral_run_dir, ringdown_run_dir]):
        bondisachs_data = str(Path(bondisachs_data).resolve())
        match = re.search(r"R(\d{4})\.h5$", bondisachs_data)
        if not match:
            raise ValueError(
                "The provided bondisachs_data does not end with 'RXXXX.h5'."
                " Modify the bondisachs_data filename to include the extraction"
                " radius in the format 'NameOfFileRXXXX.h5'. For example, if"
                " the extraction radius is 200, the filename should end with"
                " 'R0200.h5'."
            )
        elif match and bondisachs_data and extraction_radius:
            raise ValueError(
                "When pointing to an individual BondiSachs data file, do not"
                " specify extraction radius as an option"
            )
        extraction_radius = int(match.group(1))
    elif not bondisachs_data and any([inspiral_run_dir, ringdown_run_dir]):
        logger.info(
            "No Bondi-Sachs data provided. Combining provided inspiral and"
            " ringdown segements to generate single Bondi-Sachs data file for"
            " CCE with extraction radius specified. If no extraction radius is"
            " specified, the largest radius available will be used. This can"
            " take a couple minutes."
        )
        if not extraction_radius:
            # Grab different BondiSachs radii and save largest. Assuming radii
            # across inspiral and ringdown segments are all the same.
            extraction_radii_files = []
            if inspiral_run_dir:
                extraction_radii_files = sorted(
                    # These globs currently include BbhReduction files, but that
                    # gets ingnored in the for loop.
                    Path(inspiral_run_dir).glob(f"Segment_*/*R*.h5")
                )
            if ringdown_run_dir:
                extraction_radii_files += sorted(
                    Path(ringdown_run_dir).glob(f"Segment_*/*R*.h5")
                )
            extraction_radii = []
            for file in extraction_radii_files:
                # Returns None for filenames with incorrect formatting.
                match = re.search(r"R(\d{4})\.h5$", str(Path(file).resolve()))
                extraction_radii.append(int(match.group(1)))
            extraction_radius = np.max(extraction_radii)
        inspiral_bondi_sachs_data = []
        if inspiral_run_dir:
            inspiral_bondi_sachs_data = sorted(
                Path(inspiral_run_dir).glob(
                    f"Segment_*/BondiSachsCceR{extraction_radius:04d}.h5"
                ),
                key=segment_number,
            )
        elif not inspiral_run_dir:
            logger.info(
                "No inspiral run directory provided. Only ringdown data will be"
                " used to generate Bondi-Sachs data for CCE."
            )
        ringdown_bondi_sachs_data = []
        if ringdown_run_dir:
            ringdown_bondi_sachs_data = sorted(
                Path(ringdown_run_dir).glob(
                    f"Segment_*/BondiSachsCceR{extraction_radius:04d}.h5"
                ),
                key=segment_number,
            )
        elif not ringdown_run_dir:
            logger.info(
                "No ringdown run directory provided. Only inspiral data will be"
                " used to generate Bondi-Sachs data for CCE."
            )
        if not inspiral_bondi_sachs_data and not ringdown_bondi_sachs_data:
            raise ValueError(
                "No Bondi-Sachs data at the given extraction radius"
                f" ({extraction_radius}) in the provided inspiral or ringdown"
                " run directories. Please check the contents of the provided"
                " directories and ensure they contain Bondi-Sachs data for the"
                f" specified extraction radius ({extraction_radius})."
            )

        cce_dir = Path(segments_dir or run_dir)
        cce_dir.mkdir(parents=True, exist_ok=True)
        bondisachs_data = (
            Path(cce_dir).resolve()
            / f"combinedBondiSachsCceR{extraction_radius:04d}.h5"
        )
        combine_h5_dat(
            h5files=inspiral_bondi_sachs_data + ringdown_bondi_sachs_data,
            output=str(bondisachs_data),
            force=True,
            remove_overlapping_segments=True,
        )

    # Create a dictionary of the input parameters for the CCE pipeline. This
    # will be passed to the steps in the pipeline.
    cce_params = cce_input(
        bondisachs_data=bondisachs_data,
        inspiral_run_dir=inspiral_run_dir,
        ringdown_run_dir=ringdown_run_dir,
    )

    cce_params["BondiSachsData"] = bondisachs_data
    cce_params["InspiralRunDirectory"] = (
        str(Path(inspiral_run_dir).resolve())
        if inspiral_run_dir is not None
        else None
    )
    cce_params["RingdownRunDirectory"] = (
        str(Path(ringdown_run_dir).resolve())
        if ringdown_run_dir is not None
        else None
    )

    # Determine resource allocation
    if (
        scheduler_kwargs.get("scheduler") is not None
        and scheduler_kwargs.get("num_procs") is None
        and scheduler_kwargs.get("num_nodes") is None
    ):
        # CCE runs best on a single core
        scheduler_kwargs["num_procs"] = 1

    if scheduler_kwargs.get("num_nodes", 1) != 1 or None:
        logger.warning(
            "Forcing number of nodes to 1 for CCE, since CCE does not scale to"
            " more than 1 node."
        )
        scheduler_kwargs["num_nodes"] = 1

    # Schedule!
    return schedule(
        cce_input_file_template,
        **cce_params,
        **scheduler_kwargs,
        pipeline_dir=pipeline_dir,
        run_dir=run_dir,
        segments_dir=segments_dir,
    )


@click.command(name="run-cce", help=run_cce.__doc__)
@click.option(
    "--bondisachs-data",
    "-b",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
    help=(
        "Path to the an individual Bondi-Sachs data file of the form "
        " 'NameOfFileRXXXX.h5'. This can be combined file for a complete"
        " waveform, or, for a partial waveform, an individual Bondi-Sachs file"
        " from a segment directory. If you specify path to Bondi-Sachs data, do"
        " not specify inspiral or ringdown run directories, as these are used"
        " to generate the Bondi-Sachs data for CCE."
    ),
)
@click.option(
    "--inspiral-run-dir",
    "-i",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        path_type=Path,
    ),
    help=(
        "Path to inspiral directory to combine segments and extract waveforms."
    ),
)
@click.option(
    "--ringdown-run-dir",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        path_type=Path,
    ),
    help=(
        "Path to ringdown directory to combine segments and extract waveforms."
    ),
)
@click.option(
    "--extraction-radius",
    "--exr",
    type=click.IntRange(1, 1000),  # Radius should be positive and generally
    # not larger than 1000M, but we can adjust this if needed.
    help=(
        "Extraction radius for CCE in units of the total mass. This option"
        " should only be specified when pointing to a directory with multiple"
        " BondiSachs files at different extraction radii. Defaults to the"
        " largest extraction radius available. "
    ),
)
@click.option(
    "--cce-input-file-template",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
    default=CCE_INPUT_FILE_TEMPLATE,
    help="Input file template for CCE.",
    show_default=True,
)
@click.option(
    "--pipeline-dir",
    "-d",
    type=click.Path(
        writable=True,
        path_type=Path,
    ),
    help="Directory where steps in the pipeline are created.",
)
@scheduler_options
def run_cce_command(**kwargs):
    _rich_traceback_guard = True  # Hide traceback until here
    run_cce(**kwargs)


if __name__ == "__main__":
    run_cce_command(help_option_names=["-h", "--help"])
