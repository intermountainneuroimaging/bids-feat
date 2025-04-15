"""FEAT Lower Level Analysis Module"""

import logging
import os
import os.path as op
from pathlib import Path
import subprocess as sp
import numpy as np
import pandas as pd
import re
import shutil
from zipfile import ZIP_DEFLATED, ZipFile
import errorhandler
import nibabel as nb
from flywheel_gear_toolkit import GearToolkitContext
from nipype.interfaces.fsl.maths import ApplyMask
from nipype.interfaces.fsl import BET

from utils.command_line import exec_command
from utils.feat_html_singlefile import main as flathtml
from fw_gear_bids_feat import metadata
from fw_gear_bids_feat.support_functions import generate_command, execute_shell, searchfiles, sed_inplace, \
    locate_by_pattern, replace_line, fetch_dummy_volumes, apply_lookup, _normalize_volumes, _remove_volumes

log = logging.getLogger(__name__)

# Track if message gets logged with severity of error or greater
error_handler = errorhandler.ErrorHandler()


def run(gear_options: dict, app_options: dict, gear_context: GearToolkitContext) -> int:
    """Run module for first level analysis run option

    Arguments:
        gear_options: dict with gear-specific options
        app_options: dict with options for the BIDS-App

    Returns:
        run_error: any error encountered running the app. (0: no error)
    """

    # pseudo code...
    # 1a. check for bids-derivative confound (if no inputs are passed) and build confound file with list of confounds
    # 1b. add dummy volumes if desired (do this first?)
    # 2. generate event files
    # 3. pull relevant placeholders from fsf - populate with filemapper
    # 4. run feat,
    # 5. cleanup

    commands = []

    # check if run configuration is single run or multi run mode (concatenated)
    if app_options["multirun"] == False:

        for task in app_options["task-list"]:

            app_options["task"] = task

            # check for input consistency and select next steps
            template_checks(gear_options, app_options, gear_context)

            # generate filepaths for feat run (need this for other setup steps)
            identify_feat_paths(gear_options, app_options)

            if error_handler.fired:
                log.critical('Failure: exiting with code 1 due to logged errors')
                run_error = 1
                return run_error

            # If registration is applied... make sure the "T1w" and "T1w_brain" are used...
            prepare_registration_files(gear_options, app_options)

            # assign number of dummy scans (or none)
            app_options = identify_dummyvols(gear_options, app_options, gear_context)

            if app_options["include_confounds"]:
                # add all confounds from selected list to feat confounds (for each task in list)
                generate_confounds_file(gear_options, app_options, gear_context)

            df = pd.DataFrame({"func_file": app_options["func_file"], "funcpath": app_options["funcpath"],
                               "highres_file": app_options["highres_file"],
                               "feat_confounds_file": app_options.get("feat_confounds_file") or None,
                               "trs": app_options["trs"], "nvols": app_options["nvols"],
                               "AcqDummyVolumes": app_options["AcqDummyVolumes"]}, index=[0])

            if "summary_frame" not in app_options:
                app_options["summary_frame"] = df
            else:
                app_options["summary_frame"] = pd.concat([app_options["summary_frame"], df], ignore_index=True)

            # prepare events files (for each task in list) -- only download acq events for non multirun config
            if not app_options["events-in-inputs"]:
                download_event_files(gear_options, app_options, gear_context)

            # if drop non-steady state method is trim, trim all events first, then generate evs
            if app_options["DropNonSteadyStateMethod"] == "trim":
                trim_ev_files(gear_options, app_options)

            generate_ev_files(gear_options, app_options)

            # trim fmri if needed
            if app_options["DropNonSteadyStateMethod"] == "trim":
                app_options["func_file"] = _remove_volumes(app_options["func_file"], app_options["AcqDummyVolumes"])

            # prepare fsf design file
            generate_design_file(gear_options, app_options)

            # generate command - "stage" commands for final run
            commands.append(generate_command(gear_options, app_options))

            # if dry run - link working files to dry run directory
            if gear_options["dry-run"] == True:
                store_dry_run_files(gear_options, app_options)

            if error_handler.fired:
                log.critical('Failure: exiting with code 1 due to logged errors')
                run_error = 1
                return run_error
    else:
        # using concatenated task workflow
        if app_options["DropNonSteadyStateMethod"] == "regressor":
            log.warning("Non-steady state volumes are always trimmed in multi-run mode. Ignoring user configuration selection!")

        # check for input consistency and select next steps
        template_checks(gear_options, app_options, gear_context)

        # 1. concatenate all func files
        app_options["concat-file"] = concat_fmri(gear_options, app_options, gear_context)

        # 2. concatenate all confound files
        if app_options["confounds_file"]:
            concat_confounds(gear_options, app_options, gear_context)

        # 3. concatenate all event files
        concat_events(gear_options, app_options, gear_context)

        # prepare fsf design file
        app_options["func_file"] = app_options["concat-file"]
        generate_design_file(gear_options, app_options)

        # generate command - "stage" commands for final run
        commands.append(generate_command(gear_options, app_options))

        # if dry run - link working files to dry run directory
        if gear_options["dry-run"] == True:
            store_dry_run_files(gear_options, app_options)

        if error_handler.fired:
            log.critical('Failure: exiting with code 1 due to logged errors')
            run_error = 1
            return run_error

    # This is what it is all about
    for idx, cmd in enumerate(commands):
        stdout, stderr, run_error = exec_command(
            cmd,
            dry_run=gear_options["dry-run"],
            shell=True,
            cont_output=True,
            cwd=gear_options["work-dir"]
        )

    # zip dry run directory and move to outputs
    if gear_options["dry-run"]:
        os.chdir(gear_options["work-dir"])
        output_zipname = gear_options["output-dir"].absolute().as_posix() + "/dry_run_" + \
                         gear_options["destination-id"] + ".zip"

        # NEW method to zip working directory using 'zip --symlinks -r outzip.zip data/'
        cmd = "zip --symlinks -r " + output_zipname + " dry-run/ "
        execute_shell(cmd, cwd=str(app_options["work-dir"]))


    if not gear_options["dry-run"]:

        # move result feat directory to outputs
        featdirs = searchfiles(os.path.join(gear_options["work-dir"], "*.feat"))

        for featdir in featdirs:

            # if registration was skipped, add dummy registration
            if not app_options["highres_file"] or app_options["mumford_reg"]:
                try:
                    space = [s for s in app_options["func_file"].split("_") if "space" in s][0]
                    add_dummy_reg(featdir,space)
                except:
                    log.warning('Looks like this analysis was run in subject space AND no registration is applied... Inference across subjects is not possible.')

            # make special condition for bids pipeline
            if app_options["pipeline"] == "bids":
                app_options["pipeline"] = "bids/derivatives"

            # Create output directory
            output_analysis_id_dir = os.path.join(gear_options["destination-id"], app_options["pipeline"], "feat",
                                                  "sub-" + app_options["sid"], "ses-" + app_options["sesid"])
            Path(os.path.join(gear_options["work-dir"], output_analysis_id_dir)).mkdir(parents=True, exist_ok=True)

            log.info("Using output path %s", os.path.join(output_analysis_id_dir, os.path.basename(featdir)))

            shutil.copytree(featdir,
                            os.path.join(gear_options["work-dir"], output_analysis_id_dir,
                                         os.path.basename(featdir)),
                            dirs_exist_ok=True)

            # flatten html to single file
            flathtml(os.path.join(featdir, "report.html"))

            # make copies of design.fsf and html outside featdir before zipping
            inpath = os.path.join(featdir, "index.html")
            outpath = os.path.join(gear_options["output-dir"],
                                   os.path.basename(featdir.replace(".feat", "-")) + "report.html.zip")
            with ZipFile(outpath, "w", compression=ZIP_DEFLATED) as zf:
                zf.write(inpath, os.path.basename(inpath))

            # shutil.copy(os.path.join(featdir, "report.html.zip"), os.path.join(gear_options["output-dir"], featdir.replace(".feat","-") + "report.html.zip"))
            shutil.copy(os.path.join(featdir, "design.fsf"),
                        os.path.join(gear_options["output-dir"], os.path.basename(
                            featdir.replace(".feat", "-")) + "design.fsf"))

        cmd = "zip --q -r " + os.path.join(gear_options["output-dir"],
                                           "feat_" + str(gear_options["destination-id"])) + ".zip " + gear_options[
                  "destination-id"]
        execute_shell(cmd, dryrun=gear_options["dry-run"], cwd=gear_options["work-dir"])

        cmd = "chmod -R a+rwx " + os.path.join(gear_options["output-dir"])
        execute_shell(cmd, dryrun=gear_options["dry-run"], cwd=gear_options["output-dir"])

    else:
        shutil.copy(app_options["design_file"], os.path.join(gear_options["output-dir"], "design.fsf"))

    return run_error


def prepare_registration_files(gear_options: dict, app_options: dict):
    """
    Check for consistency of user configuration settings and template options. Define stages to run based on user input.

    Args:
        gear_options (dict): options for the gear, from config.json
        app_options (dict): options for the app, from config.json

    Returns:
        app_options (dict): updated options for the app, from config.json
    """

    design_file = gear_options["FSF_TEMPLATE"]

    # TODO check registration consistency
    # 1. check if registration should be applied...
    reghighres_yn = locate_by_pattern(design_file, r'set fmri\(reghighres_yn\) (.*)')
    regstandard_yn = locate_by_pattern(design_file, r'set fmri\(regstandard_yn\) (.*)')
    regstandard_nonlinear_yn = locate_by_pattern(design_file, r'set fmri\(regstandard_nonlinear_yn\) (.*)')

    if int(reghighres_yn[0]) == 0 and int(regstandard_yn[0]) == 0 and int(regstandard_nonlinear_yn[0]) == 0:
        # no registration will be applied - use mumford workaround after feat run
        app_options["mumford_reg"] = True
        return
    else:
        app_options["mumford_reg"] = False

    if not os.path.exists(app_options["highres_file"]):
        log.error("No file found matching high resolution input")

    # 2. If registration should be applied, make sure highres image is brain extracted... assume from bids naming
    bids_obj = get_bids_parts(app_options["highres_file"])

    if bids_obj["Modality"] == "T1w" and bids_obj["Desc"] == "brain":
        # do nothing! we have the file we need
        pass
    elif bids_obj["Modality"] == "T1w" and bids_obj["Desc"] == "preproc":

        # brain extracted image not passed... look for it...
        new_obj = bids_obj.copy()
        new_obj["Desc"] = "brain"
        new_obj["Modality"] = "mask"
        bids_mask = get_bids_parts(new_obj)
        mask_file_path = os.path.join(os.path.dirname(app_options["highres_file"]), bids_mask["Filename"])

        new_obj = bids_obj.copy()
        new_obj["Desc"] = "brain"
        new_obj["Modality"] = "T1w"
        bids_mask = get_bids_parts(new_obj)
        brain_file_path = os.path.join(os.path.dirname(app_options["highres_file"]), bids_mask["Filename"])

        # go look for new file...
        if os.path.exists(mask_file_path):
            # apply masking...
            # do stuff!
            mask = ApplyMask()
            mask.inputs.in_file = app_options["highres_file"]
            mask.inputs.mask_file = mask_file_path
            mask.inputs.out_file = brain_file_path
            log.info(mask.cmdline)
            out = mask.run()

        else:
            # create a brain mask then apply... give warning to user!
            # do stuff!
            bet_interface = BET()
            bet_interface.inputs.in_file = app_options["highres_file"]
            bet_interface.inputs.out_file = brain_file_path
            bet_interface.inputs.frac = 0.7
            bet_interface.inputs.mask = False
            bet_interface.inputs.robust = True
            log.info(bet_interface.cmdline)
            result = bet_interface.run()

        # store final highres file (brain only)
        app_options["highres_file"] = brain_file_path

    else:
        # not sure how to proceed. Log an error and exit
        log.critical("High Resolution structural image does not match any known templates... Unable to proceed. ")

    # this is a special case where the t1w head and t1w brain need to be named specifically for the program to "find" them
    if int(regstandard_nonlinear_yn[0]) == 1:
        # infer if highres file is brain extracted...
        brain_file_path = app_options["highres_file"]

        # symlink to a filename "T1w_brain.nii.gz"

        # get the full head image
        bids_obj = get_bids_parts(app_options["highres_file"])
        bids_obj["Desc"] = "preproc"
        bids_obj["Modality"] = "T1w"
        new_obj = get_bids_parts(bids_obj)
        head_file_path = os.path.join(os.path.dirname(app_options["highres_file"]), new_obj["Filename"])

        cwd = os.getcwd()
        try:
            os.chdir(os.path.dirname(brain_file_path))
            os.symlink(os.path.basename(brain_file_path),"T1w_brain.nii.gz")
            os.symlink(os.path.basename(head_file_path), "T1w.nii.gz")
            # symlink to filename "T1w.nii.gz"
        except Exception as e:
            os.chdir(cwd)
            raise(e)

        #finally...use the new symlinked files for the design template
        app_options["highres_file"] = os.path.join(os.path.dirname(app_options["highres_file"]), "T1w_brain.nii.gz")

    return


def get_extension(filename):
    "return file and file extension, use special formats common in fmi / fmriprep"

    custom_extensions = [".nii.gz", ".nii", ".dicom", ".tsv", ".csv", ".h5", ".surf.gii", ".txt", ".json"]
    pattern = r"^(.*)({})$".format("|".join(re.escape(ext) for ext in custom_extensions))
    match = re.match(pattern, filename)
    if match:
        basename, ext = match.groups()
        return basename, ext


def get_bids_parts(variable):

    # if method 1: take filename and interpret bids parts
    if isinstance(variable, str) or isinstance(variable, Path):
        filename = variable

    # if method 2: take bids_object already created and modify filename
    elif isinstance(variable, dict):
        bids_obj = variable
        name, ext = get_extension(bids_obj["Filename"])
    else:
        log.error("Unable to parse input to get_bids_parts")
        return

    if "filename" in locals():
        name, ext = get_extension(os.path.basename(filename))
        folder = Path(filename).parts[-2]
        res = dict(map(str.strip, sub.split('-', 1))
                   for sub in name.split('_') if '-' in sub)
        subject = res.get('sub')
        session = res.get('ses')
        bids_obj = {
            'Sub': subject,
            'Ses': session,
            'Task': res.get('task'),
            'Acq': res.get('acq'),
            'Ce': res.get('ce'),
            'Rec': res.get('rec'),
            'Dir': res.get('dir'),
            'Run': res.get('run'),
            'Recording': res.get('recording'),
            'Space':res.get('space'),
            'Desc': res.get('desc'),
            'Label': res.get('label'),
            'Hemi': res.get('hemi'),
            'Den': res.get('den'),
            'error_message': None,
            'Filename': None,
            'Folder': folder,
            'ignore': False,
            'Modality': name.split("_")[-1],
            'Part': None,
            'Path': f'sub-{subject}/ses-{session}/{folder}',
            'valid': True
        }

    bids_obj["Filename"] = result = '_'.join(f'{key.lower()}-{value}' for key, value in bids_obj.items() if
                                             key in ["Sub", "Ses", "Acq", "Ce", "Dir", "Rec", "Run", "Task",
                                                     "Recording", "Space", "Desc", "Label", "Hemi", "Den"] and value) + "_" + bids_obj["Modality"] + ext
    return bids_obj



def identify_dummyvols(gear_options: dict, app_options: dict, gear_context: GearToolkitContext):
    """
    build confounds file using dummy volumes entered ask config setting or from mriqc metadata. writes confounds file to work directory.
    Args:
        gear_options (dict): options for the gear, from config.json
        app_options (dict): options for the app, from config.json

    Returns:
        app_options (dict): updated options for the app, from config.json
    """
    # identify if dummy scans should be included
    app_options['AcqDummyVolumes'] = fetch_dummy_volumes("func-bold_task-" + app_options["task"], gear_context)

    # get volume count from functional path
    cmd = "fslnvols " + app_options["func_file"]
    log.debug("\n %s", cmd)
    terminal = sp.Popen(
        cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE, universal_newlines=True
    )
    stdout, stderr = terminal.communicate()
    log.debug("\n %s", stdout)
    log.debug("\n %s", stderr)

    nvols = int(stdout.strip("\n"))

    app_options['AcqNumFrames'] = nvols

    return app_options


def generate_confounds_file(gear_options: dict, app_options: dict, gear_context: GearToolkitContext):
    """
    Pull relevant columns from input confounds file for feat analysis. Stack with dummy vols if needed, and generate feat confounds file.
    Args:
        gear_options (dict): options for the gear, from config.json
        app_options (dict): options for the app, from config.json

    Returns:
        app_options (dict): updated options for the app, from config.json
    """

    log.info("Building confounds file...")

    all_confounds_df = pd.DataFrame()

    # Build Dummy Scans Censor timeseries
    dummy_scans = app_options['AcqDummyVolumes']
    nvols = app_options['AcqNumFrames']

    # log.info("Using %s volumes in confounds file...", str(nvols))

    if dummy_scans > 0 and app_options["DropNonSteadyStateMethod"] == "regressor":

        arr = np.zeros([nvols, dummy_scans])

        for idx in range(0, dummy_scans):
            arr[idx][idx] = 1

        tmpdf = pd.DataFrame(arr)

        all_confounds_df = pd.concat([all_confounds_df, tmpdf], axis=1)
        numrange = [*range(0, len(all_confounds_df.columns), 1)]
        cols = ["dummy_tr_" + str(s).zfill(2) for s in numrange]

        all_confounds_df.columns = cols
    if app_options["DropNonSteadyStateMethod"] == "trim":
        idx = dummy_scans
    else:
        idx = 0
    if app_options['confound-list']:
        # load confounds file - the pull relevant columns
        log.info("Using confounds spreadsheet: %s", str(app_options["confounds_file"]))
        data = pd.read_csv(app_options["confounds_file"], sep='\t')

        # pull relevant columns for feat
        colnames = app_options['confound-list'].replace(" ", "").split(",")
        for cc in colnames:

            # look for exact matches...
            if cc in data.columns:
                all_confounds_df = pd.concat([all_confounds_df, data[cc]], axis=1)

            # handle regular expression entries
            elif any(special_char in cc for special_char in ["*", "^", "$", "+"]):
                pattern = re.compile(cc)
                for regex_col in [s for s in data.columns if bool(re.search(pattern, s))]:
                    all_confounds_df = pd.concat([all_confounds_df, data[regex_col]], axis=1)

            else:
                log.info("WARNING: data column %s missing from confounds file.", cc)

    # assign final confounds file for task
    if not all_confounds_df.empty:
        all_confounds_df = all_confounds_df.iloc[idx:]
        # remove columns that are all 0's
        all_confounds_df = all_confounds_df.loc[:, all_confounds_df.any(axis=0)]

        log.info("Including confounds: %s", ", ".join(all_confounds_df.columns))
        all_confounds_df.to_csv(
            os.path.join(app_options["funcpath"], 'feat-confounds_' + app_options["task"] + '.txt'),
            header=False, index=False,
            sep=" ", na_rep=0, float_format='{:f}'.format)
        app_options["feat_confounds_file"] = os.path.join(app_options["funcpath"],
                                                          'feat-confounds_' + app_options["task"] + '.txt')
    else:
        app_options["feat_confounds_file"] = None
        log.warning("Confounds file will not be included in analysis.")


    return app_options


def download_event_files(gear_options: dict, app_options: dict, gear_context: GearToolkitContext):
    """
    Pull event files from flywheel acquisition. If more than one event file is uploaded, select based on "event-suffix"
    config option. If no events uploaded, log error.
    Args:
        gear_options:
        app_options:
        gear_context:

    Returns:

    """

    taskname = "func-bold_task-" + app_options["task"]
    acq, nii = metadata.find_matching_acq(taskname, gear_context)

    counter = 0

    for f in acq.files:
        if "_events" in f.name:

            # secondary check for correct suffix if provided...
            if app_options["events-suffix"] and app_options["events-suffix"] not in f.name:
                continue

            f.download(os.path.join(app_options["funcpath"], f.name))
            app_options["event-file"] = os.path.join(app_options["funcpath"], f.name)
            log.info("Using event file: %s", f.name)
            counter += 1

    if counter == 0:
        log.error("No event file located in flywheel acquisiton: %s", acq.id)

    if counter > 1:
        log.error(
            "Multiple event files in flywheel acquisition match selection criteria... not sure how to proceed")

    return app_options


def trim_ev_files(gear_options: dict, app_options: dict):
    """
    adjust local timing for event file based on number of dummy volumes.
    Args:
        gear_options:
        app_options:

    Returns:

    """

    df = pd.read_csv(app_options["event-file"], sep="\t")

    # if mode is "trim" we need to shift timing of input evs to match trimmed frames
    if app_options["DropNonSteadyStateMethod"] == "trim":
        # identify starting tr and shape
        nvols = nb.load(app_options["func_file"]).shape[3]
        tr = nb.load(app_options["func_file"]).header["pixdim"][4]
        dummyvols = app_options['AcqDummyVolumes']

        # local shift for each run...
        log.info("Shifting Onset time to match trimmed fMRI series: -%s seconds.", '{0:.2f}'.format(dummyvols * tr))
        df["onset"] = df["onset"] - dummyvols * tr

        if any(df["onset"] < 0):
            # if any events occur before new scanner start, remove entire event, otherwise offset any negative starttimes to zero
            df = df.drop(df[df["onset"] + df["duration"] < 0].index)

            # all other events starting before scanner start should be adjusted so new shorter duration is set
            df.loc[df["onset"] < 0, "duration"] = df.loc[df["onset"] < 0, ["onset", "duration"]].sum(axis=1)

            # if any timing is now before 0, set to 0
            df.loc[df["onset"] < 0, "onset"] = 0

        # write new event file with altered timing (used in generate_ev_files function)
        df.to_csv(app_options["event-file"], sep='\t', header=True, index=False)


def generate_ev_files(gear_options: dict, app_options: dict):
    """
    Method used for all fsl-feat gear methods. Event file will be passed as (1) BIDS format, (2) 3-column custom format,
     (3) 1-entry per volume format. Check first if events are passed as zip (do nothing except unzip). If tsv BIDS
     formatted, convert to 3-column custom format with standard naming convention.

    Args:
        gear_options (dict): options for the gear, from config.json
        app_options (dict): options for the app, from config.json

    Returns:
        app_options (dict): updated options for the app, from config.json
    """

    # for now, assume evs are bids format tsvs.... update this!!!

    evformat = app_options["evformat"]
    if evformat == "BIDS-Formatted":

        outpath = os.path.join(app_options["funcpath"], "events_" + app_options["task"])
        os.makedirs(outpath, exist_ok=True)

        df = pd.read_csv(app_options["event-file"], sep="\t")

        # look for empty weight values
        df.fillna(value=1, inplace=True)

        groups = df["trial_type"].unique()

        for g in groups:
            ev = df[df["trial_type"] == g]
            ev1 = ev.copy()
            if "weight" not in df.columns:
                ev1.loc[:, "weight"] = pd.Series([1 for x in range(len(df.index))])

            ev1 = ev1.drop(columns=["trial_type"])

            filename = os.path.join(outpath,
                                    os.path.basename(app_options["event-file"]).replace(".tsv", "-" + g + ".txt"))
            ev1.to_csv(filename, sep=" ", index=False, columns=["onset","duration","weight"], header=False)

            app_options["event_dir"] = outpath

    elif evformat == "FSL-1 Entry Per Volume" or evformat == "FSL-3 Column Format":
        # do nothing, use unzipped events directory for event dir
        pass

    return app_options


def identify_feat_paths(gear_options: dict, app_options: dict):
    """
    Identify all placeholders in the fsf design file. Use with filemapper to point to each filepath.
    Args:
        gear_options (dict): options for the gear, from config.json
        app_options (dict): options for the app, from config.json

    Returns:
        app_options (dict): updated options for the app, from config.json
    """

    app_options["func_file"] = None
    app_options["highres_file"] = None
    app_options["altref_file"] = None

    design_file = gear_options["FSF_TEMPLATE"]

    # check provided template matches the analysis type set in the configs
    analysis_level = locate_by_pattern(design_file, r'set fmri\(level\) (.*)')

    if analysis_level[0] != "1":
        log.error("Provided FSF template does not match analysis level from gear configuration. Exiting Now")
        return

    # apply filemapper to each file pattern and store
    if "pipeline" not in app_options:
        if os.path.isdir(os.path.join(gear_options["work-dir"], "fmriprep")):
            pipeline = "fmriprep"
        elif os.path.isdir(os.path.join(gear_options["work-dir"], "bids-hcp")):
            pipeline = "bids-hcp"
        elif len(os.walk(gear_options["work-dir"]).next()[1]) == 1:
            pipeline = os.walk(gear_options["work-dir"]).next()[1]
        else:
            log.error("Unable to interpret pipeline for analysis. Contact gear maintainer for more details.")
        app_options["pipeline"] = pipeline

    lookup_table = {"WORKDIR": str(gear_options["work-dir"]), "PIPELINE": app_options["pipeline"], "SUBJECT": app_options["sid"],
                    "SESSION": app_options["sesid"], "TASK": app_options["task"]}

    # special exception - fmriprep produces non-zeropaded run numbers - fix this only for applying lookup table here
    if app_options["pipeline"] == "fmriprep":
        # zero padding only relevent for versions less than v23
        if "preproc_gear" in gear_options:
            # check the version
            version = gear_options["preproc_gear"]["gear_info"]["version"]
            if version.split("_")[1] < "23.0.0":
                task = app_options["task"].split("_")
                for idx, prt in enumerate(task):
                    if "run" in task[idx]:
                        task[idx] = task[idx].replace("-0", "-")
                task = "_".join(task)
                lookup_table["TASK"] = task

    func_file_name = locate_by_pattern(design_file, r'set feat_files\(1\) "(.*)"')
    app_options["func_file"] = apply_lookup(func_file_name[0], lookup_table)
    app_options["funcpath"] = os.path.dirname(app_options["func_file"])
    if not searchfiles(app_options["func_file"]):
        log.error("Unable to locate functional file...exiting.")
    else:
        log.info("Using functional file: %s", app_options["func_file"])

    # check if higres structural registration is set to True
    highres_yn = locate_by_pattern(design_file, r'set fmri\(reghighres_yn\) (.*)')
    altref_yn = locate_by_pattern(design_file, r'set fmri\(alternateReference_yn\) (.*)')

    if int(highres_yn[0]):
        highres_file_name = locate_by_pattern(design_file, r'set highres_files\(1\) "(.*)"')
        app_options["highres_file"] = apply_lookup(highres_file_name[0], lookup_table)

        if not searchfiles(app_options["highres_file"]):
            log.error("Unable to locate highres file...exiting.")
        else:
            log.info("Using highres file: %s", app_options["highres_file"])
    else:
        log.info("Skipping highres registration.")

    if int(altref_yn[0]) and int(highres_yn[0]):
        altref_file_name = locate_by_pattern(design_file, r'set alt_ex_func\(1\) "(.*)"')
        app_options["altref_file"] = apply_lookup(altref_file_name[0], lookup_table)

        if not searchfiles(app_options["altref_file"]):
            log.error("Unable to locate altref file...exiting.")
        else:
            log.info("Using altref file: %s", app_options["altref_file"])
    else:
        log.info("Skipping alternative reference (SBRef) registration.")

    # check if confounds file is defined in model
    confound_yn = locate_by_pattern(design_file, r'set fmri\(confoundevs\) (.*)')

    # select confound file location
    if app_options["confounds_default"] and int(confound_yn[0]):
        # find confounds file...
        confounds_file_name = locate_by_pattern(design_file, r'set confoundev_files\(1\) "(.*)"')
        confounds_name = apply_lookup(confounds_file_name[0], lookup_table)

        if os.path.exists(confounds_name):
            app_options["confounds_file"] = confounds_name
            log.info("Using confounds file: %s", app_options["confounds_file"])
        else:
            app_options["confounds_file"] = None

        # do some error logging... if list of confound column names passed, but no spreadsheet found -> ERROR
        if not app_options["confounds_file"] and app_options["confound-list"]:
            log.error("Unable to locate confounds file...exiting.")

        # do some error logging... if no spreadsheet found and non-steady state volume removal not set -> WARNING
        if not app_options["confounds_file"] and (not app_options["DropNonSteadyState"] or app_options["DropNonSteadyStateMethod"] == "trim"):
            log.warning("No confounds file present and no confounds file will be created during processing. Resetting FEAT confound parameter to false.")
            replace_line(design_file, r'set fmri\(confoundevs\)', 'set fmri(confoundevs) 0')

    # if output name not given in config - use output name from template
    if not app_options["output-name"]:
        output_name = locate_by_pattern(design_file, r'set fmri\(outputdir\) "(.*)"')
        app_options["output-name"] = Path(output_name[0]).name

    # scan length
    app_options["nvols"] = get_fmri_length(app_options["func_file"])

    # repetition time
    cmd = "fslval " + app_options["func_file"] + " pixdim4"
    log.debug("\n %s", cmd)
    terminal = sp.Popen(
        cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE, universal_newlines=True
    )
    stdout, stderr = terminal.communicate()
    app_options["trs"] = stdout.strip("\n")

    return app_options


def get_fmri_length(filename):
    # scan length
    cmd = "fslnvols " + filename
    log.debug("\n %s", cmd)
    terminal = sp.Popen(
        cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE, universal_newlines=True
    )
    stdout, stderr = terminal.communicate()
    nvols = stdout.strip("\n")

    return nvols


def template_checks(gear_options: dict, app_options: dict, gear_context):
    """
    Check for consistency of user configuration settings and template options. Define stages to run based on user input.

    Args:
        gear_options (dict): options for the gear, from config.json
        app_options (dict): options for the app, from config.json

    Returns:
        app_options (dict): updated options for the app, from config.json
    """

    design_file = gear_options["FSF_TEMPLATE"]

    # TODO check registration consistency
    # 1. check if registration should be applied...
    reghighres_yn = locate_by_pattern(design_file, r'set fmri\(reghighres_yn\) (.*)')
    regstandard_yn = locate_by_pattern(design_file, r'set fmri\(regstandard_yn\) (.*)')
    regstandard_nonlinear_yn = locate_by_pattern(design_file, r'set fmri\(regstandard_nonlinear_yn\) (.*)')

    if int(reghighres_yn[0]) == 0 and int(regstandard_yn[0]) == 0 and int(regstandard_nonlinear_yn[0]) == 0:
        # no registration will be applied - use mumford workaround after feat run
        app_options["mumford_reg"] = True
    else:
        app_options["mumford_reg"] = False

    # 2. check confounds logic
    confound_yn = locate_by_pattern(design_file, r'set fmri\(confoundevs\) (.*)')
    confounds_file_name = locate_by_pattern(design_file, r'set confoundev_files\(1\) "(.*)"')

    # case 1: confounds of no interest should be located from list in preprocessed spreadsheet
    if int(confound_yn[0]) == 1 and app_options['confound-list'] and confounds_file_name[0]:
        app_options["include_confounds"] = True

    # case 2: confounds of no interest passed as seperate input for gear
    elif int(confound_yn[0]) == 1 and gear_context.get_input_path("confounds-file"):
        app_options["include_confounds"] = True

    # case 3: drop non-steady state method uses regressor
    elif app_options["DropNonSteadyStateMethod"] == "regressor" and app_options["DropNonSteadyState"]:
        app_options["include_confounds"] = True

    # # case 4: multirun mode adds run regress at minimum
    # elif app_options["multirun"]:
    #     app_options["include_confounds"] = True

    # otherwise skip over confounds of no interest
    else:
        app_options["include_confounds"] = False

    return app_options


def generate_design_file(gear_options: dict, app_options: dict):
    """
    Method specific to HCPPipeline preprocessed inputs. Check for correct registration method. Apply correct output directory
    name and path (name from config). Apply correct input set.

    Args:
        gear_options (dict): options for the gear, from config.json
        app_options (dict): options for the app, from config.json

    Returns:
        app_options (dict): updated options for the app, from config.json
    """

    # don't duplicate task name in fsf template (nuiscance for custom design files)
    if app_options["task"] in os.path.basename(gear_options["FSF_TEMPLATE"]):
        design_file = os.path.join(gear_options["work-dir"], os.path.basename(gear_options["FSF_TEMPLATE"]))
    else:
        design_file = os.path.join(gear_options["work-dir"],
                                   app_options["task"] + "." + os.path.basename(gear_options["FSF_TEMPLATE"]))

    app_options["design_file"] = design_file

    shutil.copy(gear_options["FSF_TEMPLATE"], design_file)

    # add sed replace in template file for:
    # 1. output name
    replace_line(design_file, r'set fmri\(outputdir\)',
                 'set fmri(outputdir) "' + os.path.join(gear_options["work-dir"], os.path.basename(
                     app_options["task"] + "." + app_options["output-name"])) + '"')

    # 2. func path
    replace_line(design_file, r'set feat_files\(1\)', 'set feat_files(1) "' + app_options["func_file"] + '"')

    # 3. total func length
    app_options["nvols"] = get_fmri_length(app_options["func_file"])
    replace_line(design_file, r'set fmri\(npts\)', 'set fmri(npts) ' + str(app_options["nvols"]))

    # 3a. repetition time (TR)
    replace_line(design_file, r'set fmri\(tr\)', 'set fmri(tr) ' + str(app_options["trs"]))

    # TODO check registration consistency

    # 4. highres/standard -- don't use highres for HCPPipeline models!
    stdname = locate_by_pattern(design_file, r'set fmri\(regstandard\) "(.*)"')
    stdname = os.path.join(os.environ["FSLDIR"], "data", "standard", os.path.basename(stdname[0]))
    replace_line(design_file, r'set fmri\(regstandard\) ', 'set fmri(regstandard) "' + stdname + '"')

    # if highres is passed, include it here
    if "highres_file" in app_options and app_options["highres_file"]:
        replace_line(design_file, r'set highres_files\(1\)',
                     'set highres_files(1) "' + app_options["highres_file"] + '"')

    # if altref is passed, include it here
    if "altref_file" in app_options and app_options["altref_file"]:
        replace_line(design_file, r'set alt_ex_func\(1\)',
                     'set alt_ex_func(1) "' + app_options["altref_file"] + '"')

    # # check confounds parser consistency...
    # confound_yn = locate_by_pattern(design_file, r'set fmri\(confoundevs\) (.*)')
    # if not int(confound_yn[0]) and "feat_confounds_file" in app_options:
    #     log.critical("Error: confounds file selected in gear options, but not set in FSF TEMPLATE")
    # elif int(confound_yn[0]) and "feat_confounds_file" not in app_options:
    #     log.critical("Error: confounds file was not selected in gear options, but is set in FSF TEMPLATE")

    # 5. if confounds: confounds path
    if app_options["include_confounds"] or "feat_confounds_file" in app_options:
        replace_line(design_file, r'set confoundev_files\(1\)',
                     'set confoundev_files(1) "' + app_options["feat_confounds_file"] + '"')
        replace_line(design_file, r'set fmri\(confoundevs\)', 'set fmri(confoundevs) 1')


    # 6. events - find events by event name in desgin file

    # locate all evtitle calls in template
    ev_numbers = locate_by_pattern(design_file, r'set fmri\(evtitle(\d+)')

    # for each ev, return name, find file pattern, it checks pass replace filename
    allfiles = []
    for num in ev_numbers:
        name = locate_by_pattern(design_file, r'set fmri\(evtitle' + num + '\) "(.*)"')
        evname = name[0]

        log.info("Located explanatory variable %s: %s", num, evname)

        evfiles = searchfiles(os.path.join(app_options["event_dir"], "*-" + evname + ".txt"), exit_on_errors=False)

        if len(evfiles) > 1:
            log.error("Problem locating event files programmatically... check event names and re-run.")
            continue

        elif evfiles[0] == '' and not app_options["allow-missing-evs"]:
            log.error("Problem locating event files programmatically... check event names and re-run.")
            continue

        elif evfiles[0] == '' and app_options["allow-missing-evs"]:
            # allow missing ev to be included - create new empty regressor
            cmd = """echo "0 0 0" > """ + os.path.join(app_options["event_dir"], "zeros.txt")

            if not os.path.exists(os.path.join(app_options["event_dir"], "zeros.txt")):
                execute_shell(cmd)
            evfiles[0] = os.path.join(app_options["event_dir"], "zeros.txt")
            replace_line(design_file, r'set fmri\(shape' + num + '\)',
                         'set fmri(shape' + num + ') 10')

        log.info("Found match... EV %s: %s", evname, evfiles[0])
        replace_line(design_file, r'set fmri\(custom' + num + '\)',
                     'set fmri(custom' + num + ') "' + evfiles[0] + '"')
        allfiles.append(evfiles[0])

    if allfiles:
        app_options["ev_files"] = allfiles

    return app_options


# ---------------------------------- #
# -- Concatenated Input Functions ---#
# ---------------------------------- #

def find_feat_file(gear_options: dict, app_options: dict, input_type = "fmri"):
    """
    Identify feat file from placeholders in the fsf design file. Use with filemapper to point to each filepath.
    Args:
        gear_options (dict): options for the gear, from config.json
        app_options (dict): options for the app, from config.json

    Returns:
        app_options (dict): updated options for the app, from config.json
    """
    app_options["func_file"] = None
    app_options["confounds_file"] = None
    app_options["highres_file"] = None

    design_file = gear_options["FSF_TEMPLATE"]

    # check provided template matches the analysis type set in the configs
    analysis_level = locate_by_pattern(design_file, r'set fmri\(level\) (.*)')

    if analysis_level[0] != "1":
        log.error("Provided FSF template does not match analysis level from gear configuration. Exiting Now")
        return

    # apply filemapper to each file pattern and store
    if os.path.isdir(os.path.join(gear_options["work-dir"], "fmriprep")):
        pipeline = "fmriprep"
    elif os.path.isdir(os.path.join(gear_options["work-dir"], "bids-hcp")):
        pipeline = "bids-hcp"
    elif len(os.walk(gear_options["work-dir"]).next()[1]) == 1:
        pipeline = os.walk(gear_options["work-dir"]).next()[1]
    else:
        log.error("Unable to interpret pipeline for analysis. Contact gear maintainer for more details.")
    app_options["pipeline"] = pipeline

    lookup_table = {"WORKDIR": str(gear_options["work-dir"]), "PIPELINE": pipeline, "SUBJECT": app_options["sid"],
                    "SESSION": app_options["sesid"], "TASK": app_options["task"]}

    # special exception - fmriprep produces non-zeropaded run numbers - fix this only for applying lookup table here
    if pipeline == "fmriprep":
        # zero padding only relevent for versions less than v23
        if "preproc_gear" in gear_options:
            # check the version
            version = gear_options["preproc_gear"]["gear_info"]["version"]
            if version.split("_")[1] < "23.0.0":
                task = app_options["task"].split("_")
                for idx, prt in enumerate(task):
                    if "run" in task[idx]:
                        task[idx] = task[idx].replace("-0", "-")
                task = "_".join(task)
                lookup_table["TASK"] = task

    func_file_name = locate_by_pattern(design_file, r'set feat_files\(1\) "(.*)"')
    app_options["func_file"] = apply_lookup(func_file_name[0], lookup_table)
    app_options["funcpath"] = os.path.dirname(app_options["func_file"])

    if not searchfiles(app_options["func_file"]):
        log.error("Unable to locate functional file...exiting.")

    app_options["include_confounds"] = False
    # check if confounds file is defined in model
    confound_yn = locate_by_pattern(design_file, r'set fmri\(confoundevs\) (.*)')
    if int(confound_yn[0]):
        app_options["include_confounds"] = True

        # select confound file location
        if app_options["confounds_default"]:
            # find confounds file...
            input_path = searchfiles(
                os.path.join(app_options["funcpath"], "*" + lookup_table["TASK"] + "*confounds_timeseries.tsv"))
            app_options["confounds_file"] = input_path[0]

            if not input_path:
                log.error("Unable to locate confounds file...exiting.")


def concat_confounds(gear_options: dict, app_options: dict, gear_context: GearToolkitContext):
    """
    Build a concatenated confounds file - look for confound path then concatenate together
    Args:
        gear_options (dict): options for the gear, from config.json
        app_options (dict): options for the app, from config.json

    Returns:
        app_options (dict): updated options for the app, from config.json
    """

    log.info("Building confounds file...")

    all_confounds_df = pd.DataFrame()

    for idx, task in enumerate(app_options["task-list"]):
        app_options["task"] = task

        # locate the feat files for each task...
        find_feat_file(gear_options, app_options)
        log.info("Using confounds file: %s", app_options["confounds_file"])

        # identify dummy volume count
        nvolumes = fetch_dummy_volumes(app_options["task"], gear_context)

        data = pd.DataFrame()

        # load confounds file - the pull relevant columns
        if app_options["confounds_file"]:
            log.info("Using confounds spreadsheet: %s", str(app_options["confounds_file"]))
            data = pd.read_csv(app_options["confounds_file"], sep='\t')

        if app_options['confound-list']:

            confounds_df = pd.DataFrame()

            # pull relevant columns for feat
            colnames = app_options['confound-list'].replace(" ", "").split(",")

            for cc in colnames:

                # look for exact matches...
                if cc in data.columns:
                    confounds_df = pd.concat([confounds_df, data[cc]], axis=1)

                # handle regular expression entries
                elif any(special_char in cc for special_char in ["*", "^", "$", "+"]):
                    pattern = re.compile(cc)
                    for regex_col in [s for s in data.columns if bool(re.search(pattern, s))]:
                        confounds_df = pd.concat([confounds_df, data[regex_col]], axis=1)
        else:
            confounds_df = data

        # trim initial rows and concatenate across runs
        confounds_df.columns = ["run_" + str(idx).zfill(2) + "_" + s for s in confounds_df.columns]

        confounds_df = confounds_df.iloc[nvolumes:]

        # add zeros buffer
        arr = np.zeros([confounds_df.shape[0], all_confounds_df.shape[1]])
        df = pd.DataFrame(arr, columns=all_confounds_df.columns)

        arr = np.zeros([all_confounds_df.shape[0], confounds_df.shape[1]])
        df2 = pd.DataFrame(arr, columns=confounds_df.columns)

        all_confounds_df = pd.concat([all_confounds_df, df], axis=0, ignore_index=True)
        confounds_df = pd.concat([df2, confounds_df], axis=0, ignore_index=True)

        # combine final confounds set
        all_confounds_df = pd.concat([all_confounds_df, confounds_df], axis=1)

    # assign final confounds file for task
    if not all_confounds_df.empty:
        log.info("Including confounds: %s", ", ".join(all_confounds_df.columns))
    all_confounds_df.to_csv(
        os.path.join(app_options["funcpath"], 'feat-confounds_concat.txt'),
        header=False, index=False,
        sep=" ", na_rep=0)
    app_options["feat_confounds_file"] = os.path.join(app_options["funcpath"], 'feat-confounds_concat.txt')

    return


def concat_fmri(gear_options: dict, app_options: dict, gear_context: GearToolkitContext):
    """
    Build a concatenated fmri file. Use task level preprocessed input files
    Args:
        gear_options (dict): options for the gear, from config.json
        app_options (dict): options for the app, from config.json

    Returns:
        app_options (dict): updated options for the app, from config.json
    """

    log.info("Building fmri file...")
    # generate concat filename
    app_options["task"] = app_options["task-list"][0]
    find_feat_file(gear_options, app_options)
    concat_filename = "concat_"+"_".join([i for i in app_options["func_file"].split("/")[-1].split("_") if all(ni not in i for ni in ["task-","run-"])])
    cmd = "fslmerge -t "+concat_filename

    for task in app_options["task-list"]:
        app_options["task"] = task

        # locate the feat files for each task...
        find_feat_file(gear_options, app_options)

        log.info("Using functional file: %s", app_options["func_file"])

        # identify dummy volume count
        nvolumes = fetch_dummy_volumes(app_options["task"], gear_context)

        # trim each feat file
        bold_file = app_options["func_file"]
        bold_file_new = _remove_volumes(bold_file, nvolumes)

        # try to apply mask - if exists...

        # normalize data
        bold_file_final = _normalize_volumes(bold_file_new)

        cmd = cmd + " " + bold_file_final

    # run concatenate command
    execute_shell(cmd, dryrun=False, cwd=app_options["work-dir"])

    # set new total fmri dims
    app_options["func_file"] = os.path.join(app_options["work-dir"], concat_filename)
    app_options["nvols"] = nb.load(app_options["func_file"]).shape[3]
    app_options["trs"] = nb.load(app_options["func_file"]).header["pixdim"][4]

    return os.path.join(app_options["work-dir"], concat_filename)


def run_regressor(gear_options: dict, app_options: dict, gear_context: GearToolkitContext):
    """
    Build run regressors in 3-Column format.
    """

    log.info("Building run regressors...")

    totaltime = 0
    totaltrs = 0
    n = 0

    events = pd.DataFrame()
    for task in app_options["task-list"]:
        app_options["task"] = task
        n += 1

        # locate the feat files for each task...
        find_feat_file(gear_options, app_options)

        # identify starting tr and shape
        nvols = nb.load(app_options["func_file"]).shape[3]
        tr = nb.load(app_options["func_file"]).header["pixdim"][4]

        # identify dummy volume count
        dummyvols = fetch_dummy_volumes(app_options["task"], gear_context)

        df = pd.DataFrame({"onset": totaltime, "duration": (nvols-dummyvols)*tr, "trial_type": "run-"+str(n).zfill(2), "weight": 1}, index = [0])

        # next iteration startime
        totaltrs = totaltrs + nvols - dummyvols
        totaltime = totaltrs * tr
        events = pd.concat([events, df], axis=0)

    events = events.sort_values(by=['onset'])
    return events


def concat_events(gear_options: dict, app_options: dict, gear_context: GearToolkitContext):
    """
    Build a concatenated events file. Use task level preprocessed input files
    Args:
        gear_options (dict): options for the gear, from config.json
        app_options (dict): options for the app, from config.json

    Returns:
        app_options (dict): updated options for the app, from config.json
    """

    log.info("Building events files...")

    totaltime = 0
    totaltrs = 0

    events = pd.DataFrame()
    for task in app_options["task-list"]:
        app_options["task"] = task

        # locate the feat files for each task...
        find_feat_file(gear_options, app_options)

        # identify starting tr and shape
        nvols = nb.load(app_options["func_file"]).shape[3]
        tr = nb.load(app_options["func_file"]).header["pixdim"][4]

        # identify dummy volume count
        dummyvols = fetch_dummy_volumes(app_options["task"], gear_context)

        # prepare events files (for each task in list) -- only download acq events for non multirun config
        if not app_options["events-in-inputs"]:
            download_event_files(gear_options, app_options, gear_context)

        df = pd.read_csv(app_options["event-file"], delim_whitespace=True)

        # local shift for each run...
        df["onset"] = df["onset"] - dummyvols * tr

        if any(df["onset"] < 0):
            # if any events occur before new scanner start, remove entire event, otherwise offset any negative starttimes to zero
            df = df.drop(df[df["onset"] + df["duration"] < 0].index)

            # all other events starting before scanner start should be adjusted so new shorter duration is set
            df.loc[df["onset"] < 0, "duration"] = df.loc[df["onset"] < 0, ["onset", "duration"]].sum(axis=1)

            # if any timing is now before 0, set to 0
            df.loc[df["onset"] < 0, "onset"] = 0

        # shift for run stacking
        df["onset"] = df["onset"] + totaltime

        # next iteration startime
        totaltrs = totaltrs + nvols - dummyvols
        totaltime = totaltrs * tr
        events = pd.concat([events, df], axis=0)

    events = events.sort_values(by=['onset'])


    # add all the run regressors
    df = run_regressor(gear_options, app_options, gear_context)
    events = pd.concat([events, df], axis=0)

    app_options["event-file"] = os.path.join(app_options["funcpath"], "concat-events.tsv")

    events.to_csv(app_options["event-file"], sep='\t', header=True, index=False, float_format='%.2f')
    app_options["task"] = "concat"

    # generate all tasks after concatenating - easier than doing it for each text file separately
    generate_ev_files(gear_options, app_options)

    return


def store_dry_run_files(gear_options: dict, app_options: dict):
    log.info("Gear run in dry-run mode. Storing all derived input files and exiting...")

    # symlink useful files
    dry_run_dir = gear_options["work-dir"] / "dry-run"
    if not os.path.exists(dry_run_dir):
        os.makedirs(dry_run_dir, exist_ok=True)

    shutil.copy(app_options["design_file"],os.path.join(dry_run_dir,os.path.basename(app_options["design_file"])))
    shutil.copy(app_options["func_file"], os.path.join(dry_run_dir, os.path.basename(app_options["func_file"])))
    if "feat_confounds_file" in app_options:
        shutil.copy(app_options["feat_confounds_file"],
                   os.path.join(dry_run_dir, os.path.basename(app_options["feat_confounds_file"])))

    if "ev_files" in app_options:
        os.makedirs(dry_run_dir / "evs", exist_ok=True)
        for f in app_options["ev_files"]:
            shutil.copy(f, os.path.join(dry_run_dir, "evs", os.path.basename(f)))
    return


def add_dummy_reg(featdir, reg_space):

    # 1. copy $FSLDIR/etc/flirtsch/ident.mat -> reg/example_func2standard.mat
    # 2. overwrite the standard.nii.gz image with the mean_func.nii.gz
    p = featdir
    if not os.path.exists(op.join(p, "reg")):
        log.info("Using Mumford Registration Workaround: %s", str(p))
        os.makedirs(op.join(p, "reg"))

        # copy placeholder identity matrix to registration folder (will apply no registration at higher level analysis
        shutil.copy(op.join(os.environ["FSLDIR"], "etc", "flirtsch", "ident.mat"),
                    op.join(p, "reg", "example_func2standard.mat"))

        # copy other stand-in reg files
        if "func" in reg_space:
            shutil.copy(op.join(p, "mean_func.nii.gz"),
                        op.join(p, "reg", "example_func2standard.nii.gz"))
            shutil.copy(op.join(p, "mean_func.nii.gz"),
                        op.join(p, "reg", "standard2example_func.nii.gz"))
            shutil.copy(op.join(p, "mean_func.nii.gz"),
                        op.join(p, "reg", "standard.nii.gz"))
        elif "MNI152NLin" in reg_space:

            shutil.copy(op.join(os.environ["FSLDIR"], "data", "standard", "MNI152_T1_2mm_brain.nii.gz"),
                        op.join(p, "reg", "standard.nii.gz"))

            # image registered in preproc shoul be in MNI152 space,
            #  ...but not the same voxel size...
            #  ...need to apply flirt first...
            cmd = op.join(os.environ["FSLDIR"],"bin","flirt") + " -in " + op.join(p, "mean_func.nii.gz") + \
                  " -ref " + "standard.nii.gz" + \
                  " -out " + "example_func2standard.nii.gz" + " -applyxfm"
            execute_shell(cmd, cwd=op.join(p, "reg"))

        else:
            return

        # create pngs for reports
        cmd = op.join(os.environ["FSLDIR"],"bin","slicer") + " example_func2standard standard -s 2 -x 0.35 sla.png " \
                                                             "-x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y " \
                                                             "0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y " \
                                                             "0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z " \
                                                             "0.55 slk.png -z 0.65 sll.png "
        execute_shell(cmd, cwd=op.join(p, "reg"))

        cmd = op.join(os.environ["FSLDIR"],"bin","pngappend") + " sla.png + slb.png + slc.png + sld.png + sle.png " \
                                                                "+ slf.png + slg.png + slh.png + sli.png + " \
                                                                "slj.png + slk.png + sll.png " \
                                                                "example_func2standard1.png "
        execute_shell(cmd, cwd=op.join(p, "reg"))

        cmd = op.join(os.environ["FSLDIR"],"bin","slicer") + " standard example_func2standard -s 2 -x 0.35 sla.png " \
                                                             "-x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y " \
                                                             "0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y " \
                                                             "0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z " \
                                                             "0.55 slk.png -z 0.65 sll.png "
        execute_shell(cmd, cwd=op.join(p, "reg"))

        cmd = op.join(os.environ["FSLDIR"],"bin","pngappend") + " sla.png + slb.png + slc.png + sld.png + sle.png " \
                                                                "+ slf.png + slg.png + slh.png + sli.png + " \
                                                                "slj.png + slk.png + sll.png " \
                                                                "example_func2standard2.png "
        execute_shell(cmd, cwd=op.join(p, "reg"))

        cmd = op.join(os.environ["FSLDIR"],"bin","pngappend") + " example_func2standard1.png - " \
                                                                "example_func2standard2.png " \
                                                                "example_func2standard.png "
        execute_shell(cmd, cwd=op.join(p, "reg"))

        cmd = "/bin/rm -f sl?.png example_func2standard2.png"
        execute_shell(cmd, cwd=op.join(p, "reg"))
