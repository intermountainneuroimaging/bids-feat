"""Parser module to parse gear config.json."""
from typing import Tuple
from zipfile import ZipFile
from flywheel_gear_toolkit import GearToolkitContext
import os
import logging
from fw_gear_bids_feat.main import execute_shell
import errorhandler

log = logging.getLogger(__name__)

# Track if message gets logged with severity of error or greater
error_handler = errorhandler.ErrorHandler()

def parse_config(
        gear_context: GearToolkitContext,
) -> Tuple[dict, dict]:
    """Parse the config and other options from the context, both gear and app options.

    Returns:
        gear_options: options for the gear
        app_options: options to pass to the app
    """
    # ##   Gear config   ## #
    errors = []

    gear_options = {
        "dry-run": gear_context.config.get("gear-dry-run"),
        "output-dir": gear_context.output_dir,
        "destination-id": gear_context.destination["id"],
        "work-dir": gear_context.work_dir,
        "client": gear_context.client,
        "environ": os.environ,
        "debug": gear_context.config.get("debug"),
        "preproc_zipfile": gear_context.get_input_path("preprocessing-pipeline-zip"),
        "FSF_TEMPLATE": gear_context.get_input_path("FSF_TEMPLATE")
    }


    # set the output dir name for the BIDS app:
    gear_options["output_analysis_id_dir"] = (
            gear_options["output-dir"] / gear_options["destination-id"]
    )

    # ##   App options:   ## #
    app_options_keys = [
        "task-list",
        "output-name",
        "confound-list",
        "DropNonSteadyState",
        "DummyVolumes",
        "multirun",
        "events-suffix"
    ]
    app_options = {key: gear_context.config.get(key) for key in app_options_keys}

    work_dir = gear_options["work-dir"]
    if work_dir:
        app_options["work-dir"] = work_dir

    # additional preprocessing input
    if gear_context.get_input_path("additional-input-one"):
        app_options["additional_input"] = True
        gear_options["additional_input_zip"] = gear_context.get_input_path("additional-input-one")
    else:
        app_options["additional_input"] = False

    # confounds file input
    if gear_context.get_input_path("additional-input-one"):
        app_options["confounds_default"] = False
        gear_options["confounds_file"] = gear_context.get_input_path("confounds-file")
    else:
        app_options["confounds_default"] = True  # look for confounds in bids-derivative folder

    # confounds file input
    if gear_context.get_input_path("event-file"):
        app_options["events-in-inputs"] = True
        gear_options["event-file"] = gear_context.get_input_path("event-file")
    else:
        app_options["events-in-inputs"] = False  # look for events in flywheel acquisition

    # log filepaths
    log.info("Inputs file path, %s", gear_options["preproc_zipfile"])
    if app_options["additional_input"]:
        log.info("Additional inputs file path, %s", gear_options["additional_input_zip"])

    # pull config settings
    gear_options["feat"] = {
        "common_command": "feat",
        "params": ""
    }

    # unzip input files
    unzip_inputs(gear_options, gear_options["preproc_zipfile"])

    if app_options["additional_input"]:
        unzip_inputs(gear_options, gear_options["additional_input_zip"])

    # if task-list is a comma seperated list, apply nifti concatenation for analysis
    if "," in app_options["task-list"]:
        app_options["task-list"] = app_options["task-list"].replace(" ","").split(",")

    # building fsf - use file mapper methods to generate bids filename and path
    destination = gear_context.client.get(gear_context.destination["id"])
    sid = gear_context.client.get(destination.parents.subject)
    sesid = gear_context.client.get(destination.parents.session)

    app_options["sid"] = sid.label
    app_options["sesid"] = sesid.label

    return gear_options, app_options


def unzip_inputs(gear_options, zip_filename):
    """
    unzip_inputs unzips the contents of zipped gear output into the working
    directory.
    Args:
        gear_options: The gear context object
            containing the 'gear_dict' dictionary attribute with key/value,
            'gear-dry-run': boolean to enact a dry run for debugging
        zip_filename (string): The file to be unzipped
    """
    rc = 0

    # use linux "unzip" methods in shell in case symbolic links exist
    log.info("Unzipping file, %s", zip_filename)
    cmd = "unzip " + zip_filename + " -d " + str(gear_options["work-dir"])
    execute_shell(cmd, cwd=gear_options["work-dir"])

    # if unzipped directory is a destination id - move all outputs one level up
    with ZipFile(zip_filename, "r") as f:
        top = [item.split('/')[0] for item in f.namelist()]

    if len(top[0]) == 24:
        # directory starts with flywheel destination id - obscure this for now...
        cmd = "mv "+top[0]+'/* . ; rm -R '+top[0]
        execute_shell(cmd, cwd=gear_options["work-dir"])

    log.info("Done unzipping.")

    if error_handler.fired:
        log.critical('Failure: exiting with code 1 due to logged errors')
        run_error = 1
        return run_error

    return rc


