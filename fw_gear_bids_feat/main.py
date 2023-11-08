"""Main module."""

import logging
from pathlib import Path
import errorhandler
from typing import List, Tuple
from flywheel_gear_toolkit import GearToolkitContext
from fw_gear_bids_feat import feat_lower_level_analysis, feat_higher_level_analysis

log = logging.getLogger(__name__)

# Track if message gets logged with severity of error or greater
error_handler = errorhandler.ErrorHandler()

# # Also log to stderr
# stream_handler = logging.StreamHandler(stream=sys.stderr)
# log.addHandler(stream_handler)


def prepare(
        gear_options: dict,
        app_options: dict,
) -> Tuple[List[str], List[str]]:
    """Prepare everything for the algorithm run.

    It should:
     - Install FreeSurfer license (if needed)

    Same for FW and RL instances.
    Potentially, this could be BIDS-App independent?

    Args:
        gear_options (Dict): gear options
        app_options (Dict): options for the app

    Returns:
        errors (list[str]): list of generated errors
        warnings (list[str]): list of generated warnings
    """
    # pylint: disable=unused-argument
    # for now, no errors or warnings, but leave this in place to allow future methods
    # to return an error
    errors: List[str] = []
    warnings: List[str] = []

    return errors, warnings
    # pylint: enable=unused-argument


def run(gear_options: dict, app_options: dict, gear_context: GearToolkitContext) -> int:
    """Run FSL-FEAT using generic bids-derivative inputs.

    Arguments:
        gear_options: dict with gear-specific options
        app_options: dict with options for the BIDS-App

    Returns:
        run_error: any error encountered running the app. (0: no error)
    """

    # report selected config settings
    log.info("Using %s", app_options["run-level"])

    if app_options["run-level"] == "First Level Analysis":
        log.info("Using Configuration Settings: ")

        log.parent.handlers[0].setFormatter(logging.Formatter('\t\t%(message)s'))

        log.info("DropNonSteadyState: %s", str(app_options["DropNonSteadyState"]))
        if "DummyVolumes" in app_options:
            log.info("DummyVolumes: %s", str(app_options["DummyVolumes"]))
        log.info("evformat: %s", str(app_options["evformat"]))
        if "events-suffix" in app_options:
            log.info("events-suffix: %s", str(app_options["events-suffix"]))
        log.info("allow-missing-evs: %s", str(app_options["allow-missing-evs"]))
        log.info("output-name: %s", str(app_options["output-name"]))
        log.info("task-list: %s", str(app_options["task-list"]))
        log.info("Using fsf template: %s", Path(gear_options["FSF_TEMPLATE"]).name)
        log.parent.handlers[0].setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        # run!
        feat_lower_level_analysis.run(gear_options, app_options, gear_context)

    elif app_options["run-level"] == "Higher Level Analysis":
        log.warning("Ignoring configuration settings: DropNonSteadyState, DummyVolumes, evformat, events-suffix, allow-missing-evs, task-list")
        log.info("output-name: %s", str(app_options["output-name"]))
        log.info("Using fsf template: %s", Path(gear_options["FSF_TEMPLATE"]).name)

        # run!
        feat_higher_level_analysis.run(gear_options, app_options, gear_context)


