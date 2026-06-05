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


# Use this to make sure segments are combined chronologically.
def segment_number(path):
    match = re.search(r"Segment_(\d+)", str(path))
    if match is None:
        raise ValueError(f"Could not find segment number in {path}")
    return int(match.group(1))


def cce_input(
    bondisachs_data: Optional[Union[str, Path]] = None,
    inspiral_run_dir: Optional[Union[str, Path]] = None,
    ringdown_run_dir: Optional[Union[str, Path]] = None,
    extraction_radius: Optional[int] = None,
) -> dict:
    """Generate the input for the CCE pipeline.

    This input will be used to fill the 'CCE_INPUT_FILE_TEMPLATE'.

    Arguments:
        bondisachs_data: Path to the bondisachs data generated from a bbh sim
        inspiral_run_dir: Directory of the inspiral run.
        ringdown_run_dir: Directory of the ringdown run.
        extraction_radius: Extraction radius for CCE in units of the total mass.
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
        "ExtractionRadius": extraction_radius,
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

    Point the inspiral_run_dir to the inspiral directory and ringdown_run_dir to
    the ringdown directory for a given bbh simulation for a full waveform
    extraction. For partial waveforms you can point bondisachs_data to the
    BondiSachs file from a given segment in either directory. Specify
    extraction_radius, otherwise the default of 200M is used. The remaining
    options are forwarded to the 'schedule' command. See 'schedule' docs for
    details.

    Arguments:
        bondisachs_data: Path to the bondisachs data generated from a bbh run.
        inspiral_run_dir: Directory of the inspiral run.
        ringdown_run_dir: Directory of the ringdown run.
        extraction_radius: Extraction radius for CCE in units of the total mass.
        pipeline_dir: Directory where steps in the pipeline are created. If not
        specified, a temporary directory is used that is deleted after the
        pipeline finishes.
        run_dir: Directory where the CCE pipeline is run. If not specified, the
        pipeline is run in the current working directory.
        cce_input_file_template: Input file template for CCE. This should be a
        yaml file that defines the steps in the CCE pipeline.
    """
    logger.warning(
        "The BBH pipeline is still experimental. Please review the "
        " generated input files."
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
        if match and bondisachs_data and extraction_radius:
            if int(match.group(1)) != extraction_radius:
                raise ValueError(
                    "The extraction radius specified does not match the"
                    " extraction radius in the bondisachs_data filename."
                )
        elif not match and not extraction_radius:
            raise ValueError(
                "The provided bondisachs_data does not end with"
                " 'RXXXX.h5', and no extraction radius is specified. Either"
                " specify the extraction radius, or modify the bondisachs_data"
                " filename to include the extraction radius in the format"
                " 'RXXXX.h5'. For example, if the extraction radius is 200, the"
                " path should end with 'R0200.h5'."
            )
        elif match and bondisachs_data and not extraction_radius:
            extraction_radius = int(match.group(1))

    elif not bondisachs_data and any([inspiral_run_dir, ringdown_run_dir]):
        logger.warning(
            "No Bondi-Sachs data provided. Combining provided inspiral and"
            " ringdown directories to combine Bondi-Sachs data for CCE with"
            " extraction radius specified. If no extraction radius is"
            " specified, the default of 200 will be used. This can take a"
            " couple minutes."
        )
        if not extraction_radius:
            extraction_radius = 200  # Somewhat arbitrary default.
        inspiral_bondi_sachs_data = []
        if inspiral_run_dir:
            inspiral_bondi_sachs_data = sorted(
                Path(inspiral_run_dir).glob(
                    f"Segment_*/BondiSachsCceR{extraction_radius:04d}.h5"
                ),
                key=segment_number,
            )
        elif not inspiral_run_dir:
            logger.warning(
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
            logger.warning(
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
        extraction_radius=extraction_radius,
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
    cce_params["ExtractionRadius"] = extraction_radius

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
        "Path to the Bondi-Sachs data generated from a bbh inspiral or"
        " ringdown. This can be combined for a complete wave form, or just from"
        " the provided directories. If you specify path to Bondi-Sachs data,"
        " do not specify inspiral or ringdown run directories, as these are"
        " used to generate the Bondi-Sachs data for CCE."
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
    type=click.IntRange(1, 1000),  # Radius should be positive and generally
    # not larger than 1000M, but we can adjust this if needed.
    help=(
        "Extraction radius for CCE in units of the total mass. Typical"
        " extraction radii for SpECTRE runs are [100, 150, 200]. "
    ),
    show_default=True,
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
