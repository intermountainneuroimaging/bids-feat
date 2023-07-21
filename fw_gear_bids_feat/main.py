"""Main module."""

import logging
import os
from pathlib import Path
import subprocess as sp
import numpy as np
import pandas as pd
import re
import shutil
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile
import errorhandler
from typing import List, Tuple, Union
import nibabel as nib
from flywheel_gear_toolkit import GearToolkitContext

from utils.command_line import exec_command
from utils.feat_html_singlefile import main as flathtml
from fw_gear_bids_feat import metadata

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

    log.info("This is the beginning of the run file")

    # pseudo code...
    # 1a. check for bids-derivative confound (if no inputs are passed) and build confound file with list of confounds
    # 1b. add dummy volumes if desired (do this first?)
    # 2. generate event files
    # 3. pull relevant placeholders from fsf - populate with filemapper
    # 4. run feat,
    # 5. cleanup

    commands=[]

    if not type(app_options["task-list"]) == list:
        app_options["task-list"] = [app_options["task-list"]]

    for task in app_options["task-list"]:

        app_options["task"] = task

        # generate filepaths for feat run (need this for other setup steps)
        identify_feat_paths(gear_options, app_options)

        # add all confounds from selected list to feat confounds (for each task in list)
        app_options = generate_confounds_file(gear_options, app_options, gear_context)

        # prepare events files (for each task in list)
        if not app_options["events-in-inputs"]:
            download_event_files(gear_options, app_options, gear_context)
        app_options = generate_ev_files(gear_options, app_options)

        if not app_options["multirun"]:
            # prepare fsf design file
            app_options = generate_design_file(gear_options, app_options)

            # generate command - "stage" commands for final run
            commands.append(generate_command(gear_options, app_options))


    # if more than one task is entered, concatenate
    if len(app_options["task-list"]) > 1 and app_options["multirun"]:
        # combine_files and run
        app_options["task"] = "combined"

        # currently do nothing else...
        log.error("Feature for multirun not yet available")


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


    if not gear_options["dry-run"]:

        # move result feat directory to outputs
        featdirs = searchfiles(os.path.join(gear_options["work-dir"], "*.feat"))

        for featdir in featdirs:
            # Create output directory
            output_analysis_id_dir = os.path.join(gear_options["destination-id"], gear_options["pipeline"], "feat","sub-" + app_options["sid"], "ses-" + app_options["sesid"])
            Path(os.path.join(gear_options["work-dir"], output_analysis_id_dir)).mkdir(parents=True, exist_ok=True)

            log.info("Using output path %s", os.path.join(output_analysis_id_dir, os.path.basename(featdir)))

            shutil.copytree(featdir, os.path.join(gear_options["work-dir"], output_analysis_id_dir, os.path.basename(featdir)),dirs_exist_ok=True)

            # flatten html to single file
            flathtml(os.path.join(featdir, "report.html"))

            # make copies of design.fsf and html outside featdir before zipping
            inpath = os.path.join(featdir, "index.html")
            outpath = os.path.join(gear_options["output-dir"], os.path.basename(featdir.replace(".feat","-")) + "report.html.zip")
            with ZipFile(outpath, "w", compression=ZIP_DEFLATED) as zf:
                zf.write(inpath, os.path.basename(inpath))

            # shutil.copy(os.path.join(featdir, "report.html.zip"), os.path.join(gear_options["output-dir"], featdir.replace(".feat","-") + "report.html.zip"))
            shutil.copy(os.path.join(featdir, "design.fsf"), os.path.join(gear_options["output-dir"], os.path.basename(featdir.replace(".feat","-")) + "design.fsf"))

        cmd = "zip -r " + os.path.join(gear_options["output-dir"], "feat_"+str(gear_options["destination-id"])) + ".zip " + gear_options["destination-id"]
        execute_shell(cmd, dryrun=gear_options["dry-run"], cwd=gear_options["work-dir"])

        cmd = "chmod -R a+rwx " + os.path.join(gear_options["output-dir"])
        execute_shell(cmd, dryrun=gear_options["dry-run"], cwd=gear_options["output-dir"])

    else:
        shutil.copy(app_options["design_file"], os.path.join(gear_options["output-dir"], "design.fsf"))

    return run_error


def generate_dummyvols_file(gear_options: dict, app_options: dict, gear_context: GearToolkitContext):
    """
    build confounds file using dummy volumes entered ask config setting or from mriqc metadata. writes confounds file to work directory.
    Args:
        gear_options (dict): options for the gear, from config.json
        app_options (dict): options for the app, from config.json

    Returns:
        app_options (dict): updated options for the app, from config.json
    """
    # identify if dummy scans should be included
    app_options['AcqDummyVolumes'] = fetch_dummy_volumes(app_options["task"], gear_context)

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

    # assign number of dummy scans (or none)
    app_options = generate_dummyvols_file(gear_options, app_options, gear_context)

    # Build Dummy Scans Censor timeseries
    dummy_scans = app_options['AcqDummyVolumes']
    nvols = app_options['AcqNumFrames']

    log.info("Using %s volumes in confounds file...", str(nvols))

    if dummy_scans > 0:

        arr = np.zeros([nvols, dummy_scans])

        for idx in range(0, dummy_scans):
            arr[idx][idx] = 1

        tmpdf = pd.DataFrame(arr)

        all_confounds_df = pd.concat([all_confounds_df, tmpdf], axis=1)

    if app_options['confound-list']:
        # load confounds file - the pull relevant columns
        log.info("Using confounds spreadsheet: %s", str(gear_options["confounds_file"]))
        data = pd.read_csv(gear_options["confounds_file"], sep='\t')

        # pull relevant columns for feat
        colnames = app_options['confound-list'].replace(" ","").split(",")
        for cc in colnames:
            if cc in data.columns:
                all_confounds_df = pd.concat([all_confounds_df, data[cc]], axis=1)
            else:
                log.info("WARNING: data column %s missing from confounds file.", cc)

    # assign final confounds file for task
    if not all_confounds_df.empty:
        all_confounds_df.to_csv(os.path.join(app_options["funcpath"], 'feat-confounds_'+app_options["task"]+'.txt'), header=False, index=False,
                                sep=" ")
        app_options["feat_confounds_file"] = os.path.join(app_options["funcpath"], 'feat-confounds_'+app_options["task"]+'.txt')

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

    taskname = app_options["task"]
    acq, nii = metadata.find_matching_acq(taskname, gear_context)

    counter = 0

    for f in acq.files:
        if "_events" in f.name and app_options["events-suffix"] in f.name:
            f.download(os.path.join(app_options["funcpath"], f.name))
            app_options["event-file"] = os.path.join(app_options["funcpath"], f.name)
            log.info("Using event file: %s", f.name)
            counter += 1

    if counter == 0:
        log.error("No event file located in flywheel acquisiton: %s", acq.id)

    if counter > 1:
        log.error("Multiple event files in flywheel acquisition match selection criteria... not sure how to proceed")

    return app_options



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

    outpath = os.path.join(app_options["funcpath"], "events_"+app_options["task"])
    os.makedirs(outpath, exist_ok=True)

    evformat = "bids"
    if evformat == "bids":
        df = pd.read_csv(app_options["event-file"], sep="\t")

        groups = df["trial_type"].unique()

        for g in groups:
            ev = df[df["trial_type"] == g]
            ev1 = ev.copy()
            ev1.loc[:, "weight"] = pd.Series([1 for x in range(len(df.index))])

            ev1 = ev1.drop(columns=["trial_type"])

            filename = os.path.join(outpath,
                                    os.path.basename(app_options["event-file"]).replace(".tsv", "-" + g + ".txt"))
            ev1.to_csv(filename, sep=" ", index=False, header=False)

    app_options["event_dir"] = outpath

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
    app_options["include_confounds"] = False

    design_file = gear_options["FSF_TEMPLATE"]

    # apply filemapper to each file pattern and store
    if os.path.isdir(os.path.join(gear_options["work-dir"],"fmriprep")):
        pipeline = "fmriprep"
    elif os.path.isdir(os.path.join(gear_options["work-dir"],"bids-hcp")):
        pipeline = "bids-hcp"
    elif len(os.walk(gear_options["work-dir"]).next()[1]) == 1:
        pipeline = os.walk(gear_options["work-dir"]).next()[1]
    else:
        log.error("Unable to interpret pipeline for analysis. Contact gear maintainer for more details.")
    gear_options["pipeline"] = pipeline

    lookup_table = {"PIPELINE": pipeline, "SUBJECT": app_options["sid"], "SESSION": app_options["sesid"], "TASK": app_options["task"]}

    # special exception - fmriprep produces non-zeropaded run numbers - fix this only for applying lookup table here
    if pipeline == "fmriprep":
        task = app_options["task"].split("_")
        for idx, prt in enumerate(task):
            if "run" in task[idx]:
                task[idx] = task[idx].replace("-0","-")
        task = "_".join(task)
        lookup_table = {"PIPELINE": pipeline, "SUBJECT": app_options["sid"], "SESSION": app_options["sesid"],
                        "TASK": task}

    func_file_name = locate_by_pattern(design_file, r'set feat_files\(1\) "(.*)"')
    app_options["func_file"] = apply_lookup(func_file_name[0], lookup_table)
    app_options["funcpath"] = os.path.dirname(app_options["func_file"])
    if not searchfiles(app_options["func_file"]):
        log.error("Unable to locate functional file...exiting.")
    else:
        log.info("Using functional file: %s", app_options["func_file"])

    # check if higres structural registration is set to True
    highres_yn = locate_by_pattern(design_file, r'set fmri\(reghighres_yn\) (.*)')
    if int(highres_yn[0]):
        highres_file_name = locate_by_pattern(design_file, r'set highres_files\(1\) "(.*)"')
        app_options["highres_file"] = apply_lookup(highres_file_name[0], lookup_table)

        if not searchfiles(app_options["highres_file"]):
            log.error("Unable to locate highres file...exiting.")
        else:
            log.info("Using highres file: %s", app_options["highres_file"])
    else:
        log.info("Skipping highres registration.")

    # check if confounds file is defined in model
    confound_yn = locate_by_pattern(design_file, r'set fmri\(confoundevs\) (.*)')
    if int(confound_yn[0]):
        app_options["include_confounds"] = True

        # select confound file location
        if app_options["confounds_default"]:
            # find confounds file...
            input_path = searchfiles(os.path.join(app_options["funcpath"], "*"+task+"*confounds_timeseries.tsv"))
            gear_options["confounds_file"] = input_path[0]

            if not input_path:
                log.error("Unable to locate confounds file...exiting.")
            else:
                log.info("Using confounds file: %s", gear_options["confounds_file"])

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

    design_file = os.path.join(gear_options["work-dir"], app_options["task"]+"."+os.path.basename(gear_options["FSF_TEMPLATE"]))
    app_options["design_file"] = design_file

    shutil.copy(gear_options["FSF_TEMPLATE"], design_file)

    # add sed replace in template file for:
    # 1. output name
    if app_options["output-name"]:
        replace_line(design_file, r'set fmri\(outputdir\)', 'set fmri(outputdir) "' + os.path.join(gear_options["work-dir"], os.path.basename(app_options["task"]+"."+app_options["output-name"])) + '"')

    # 2. func path
    replace_line(design_file, r'set feat_files\(1\)', 'set feat_files(1) "' + app_options["func_file"] + '"')

    # 3. total func length??
    cmd = "fslnvols " + app_options["func_file"]
    log.debug("\n %s", cmd)
    terminal = sp.Popen(
        cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE, universal_newlines=True
    )
    stdout, stderr = terminal.communicate()
    nvols = stdout.strip("\n")
    replace_line(design_file, r'set fmri\(npts\)', 'set fmri(npts) ' + nvols)

    # 3a. repetition time (TR)
    cmd = "fslval " + app_options["func_file"] +" pixdim4"
    log.debug("\n %s", cmd)
    terminal = sp.Popen(
        cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE, universal_newlines=True
    )
    stdout, stderr = terminal.communicate()
    trs = stdout.strip("\n")
    replace_line(design_file, r'set fmri\(tr\)', 'set fmri(tr) ' + trs)

    # TODO check registration consistency

    # 4. highres/standard -- don't use highres for HCPPipeline models!
    stdname = locate_by_pattern(design_file, r'set fmri\(regstandard\) "(.*)"')
    stdname = os.path.join(os.environ["FSLDIR"], "data", "standard", os.path.basename(stdname[0]))
    replace_line(design_file, r'set fmri\(regstandard\) ', 'set fmri(regstandard) "' + stdname + '"')

    # if highres is passed, include it here
    if app_options["highres_file"]:
        replace_line(design_file, r'set highres_files\(1\)', 'set feat_files(1) "' + app_options["highres_file"] + '"')

    # check confounds parser consistency...
    confound_yn = locate_by_pattern(design_file, r'set fmri\(confoundevs\) (.*)')
    if not confound_yn[0] and "feat_confounds_file" in app_options:
        log.critical("Error: confounds file selected in gear options, but not set in FSF TEMPLATE")
    elif confound_yn[0] and "feat_confounds_file" not in app_options:
        log.critical("Error: confounds file was not selected in gear options, but is set in FSF TEMPLATE")

    # 5. if confounds: confounds path
    if "feat_confounds_file" in app_options:
        replace_line(design_file, r'set confoundev_files\(1\)',
                     'set confoundev_files(1) "' + app_options["feat_confounds_file"] + '"')

    # 6. events - find events by event name in desgin file

    # locate all evtitle calls in template
    ev_numbers = locate_by_pattern(design_file, r'set fmri\(evtitle(\d+)')

    # for each ev, return name, find file pattern, it checks pass replace filename
    allfiles = []
    for num in ev_numbers:
        name = locate_by_pattern(design_file, r'set fmri\(evtitle' + num + '\) "(.*)"')
        evname = name[0]

        log.info("Located explanatory variable %s: %s", num, evname)

        evfiles = searchfiles(os.path.join(app_options["event_dir"], "*" + evname + "*"))

        if len(evfiles) > 1 or evfiles[0] == '':
            log.error("Problem locating event files programmatically... check event names and re-run.")
        else:
            log.info("Found match... EV %s: %s", evname, evfiles[0])
            replace_line(design_file, r'set fmri\(custom' + num + '\)',
                         'set fmri(custom' + num + ') "' + evfiles[0] + '"')
            allfiles.append(evfiles[0])

    if allfiles:
        app_options["ev_files"] = allfiles

    return app_options


def concat_nifti(gear_options: dict, app_options: dict):

    zerocenter_list = []; tmean_list = [];

    with tempfile.TemporaryDirectory(dir=gear_options["work-dir"]) as tmpdir:

        for task in app_options["task-list"]:

            funcpath = searchfiles(os.path.join(gear_options["work-dir"], "**", "MNINonLinear", "Results",
                                                "*" + task + "*"))
            funcpath = funcpath[0]

            if app_options["icafix"]:
                funcfile = searchfiles(os.path.join(funcpath, "*clean.nii.gz"))
                infile = funcfile[0]
            else:
                funcfile = searchfiles(os.path.join(funcpath, "*_bold.nii.gz"))
                infile = funcfile[0]

            # 1. create a noise image
            img = nib.load(infile)
            noise = np.random.randn(img.shape[0], img.shape[1], img.shape[2], app_options["dummy-scans"])
            nim = nib.Nifti1Image(noise.astype('f'), img.affine, img.header)
            noise_fname = os.path.join(tmpdir, 'noise.nii.gz')
            nib.save(nim, noise_fname)

            # 2. from orginal image, create a trimmed series
            trim_fname = os.path.join(tmpdir, 'trimmed.nii.gz')
            cmd = "fslroi " + infile + " " + trim_fname + " " + str(app_options["dummy-scans"]) + " -1"
            execute_shell(cmd, gear_options["dry-run"])

            # 3. using trimmed file, compute temporal mean
            tmean_fname = os.path.join(tmpdir, Path(infile).name.replace(".nii.gz", "_meanfunc.nii.gz"))
            cmd = "fslmaths " + trim_fname + " -Tmean " + tmean_fname
            execute_shell(cmd, gear_options["dry-run"])

            tmean_list.append(tmean_fname)

            # 4 remove temporal mean from trimmed datset
            demeaned_fname = os.path.join(tmpdir, 'trimmed_zerocenter.nii.gz')
            cmd = "fslmaths " + trim_fname + " -sub " + tmean_fname + " " + demeaned_fname + " -odt float"
            execute_shell(cmd, gear_options["dry-run"])

            # 5. concatenate adjusted noise model and trimmed timeseries
            output_zerocenter = os.path.join(tmpdir,
                                             Path(infile).name.replace(".nii.gz", "_withnoise.nii.gz"))
            cmd = "fslmerge -t " + output_zerocenter + " " + noise_fname + " " + demeaned_fname
            execute_shell(cmd, gear_options["dry-run"])

            zerocenter_list.append(output_zerocenter)

        # get across-trials temporal mean
        itr_fname = os.path.join(tmpdir, "rm.tmp.nii.gz")
        tmean_fname = os.path.join(tmpdir, "rm.mean_func.nii.gz")
        cmd = "fslmerge -t " + itr_fname + " " + " ".join(tmean_list)
        execute_shell(cmd, gear_options["dry-run"])

        cmd = "fslmaths " + itr_fname + " -Tmean " + tmean_fname
        execute_shell(cmd, gear_options["dry-run"])

        # --- finally concatenate all runs ---

        # use fslmerge (requires fsl in system path) to concatenate list of input images
        itr_fname = os.path.join(tmpdir, "rm.tmp2.nii.gz")
        cmd = "fslmerge -t " + itr_fname + " " + " ".join(zerocenter_list)
        execute_shell(cmd, gear_options["dry-run"])

        # and add temporal mean back to concatenated dataset
        pardir = searchfiles(os.path.join(gear_options["work-dir"], "**", "MNINonLinear", "Results"))
        os.makedirs(os.path.join(pardir[0],"concatenated"),exist_ok=True)

        final_output = os.path.join(pardir[0], "concatenated", "concatenated_withnoise.nii.gz")
        cmd = "fslmaths " + itr_fname + " -add " + tmean_fname + " " + final_output
        execute_shell(cmd, gear_options["dry-run"])

        app_options["func_file"] = final_output
        app_options["funcpath"] = os.path.join(pardir[0], "concatenated")

        return app_options

def concat_motion(gear_options: dict, app_options: dict):
    files = []
    for task in app_options["task-list"]:
        motion = searchfiles(os.path.join(gear_options["work-dir"], "**", "MNINonLinear", "Results",
                                            "*" + task + "*","Movement_Regressors.txt"))

        files.append(motion[0])
    log.info("Concatenating Motion files: \n%s",'\n'.join(files))

    cmd = 'cat '+" ".join(files) + ' > ' + os.path.join(app_options["funcpath"], "Movement_Regressors.txt")
    log.debug("\n %s", cmd)
    terminal = sp.Popen(
        cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE, universal_newlines=True
    )
    stdout, stderr = terminal.communicate()
    log.debug("\n %s", stdout)
    log.debug("\n %s", stderr)

    return app_options



def generate_command(
        gear_options: dict,
        app_options: dict,
) -> List[str]:
    """Build the main command line command to run.

    This method should be the same for FW and XNAT instances. It is also BIDS-App
    generic.

    Args:
        gear_options (dict): options for the gear, from config.json
        app_options (dict): options for the app, from config.json
    Returns:
        cmd (list of str): command to execute
    """

    cmd = []
    cmd.append(gear_options["feat"]["common_command"])
    cmd.append(app_options["design_file"])

    return cmd


def execute_shell(cmd, dryrun=False, cwd=os.getcwd()):
    log.info("\n %s", cmd)
    if not dryrun:
        terminal = sp.Popen(
            cmd,
            shell=True,
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            universal_newlines=True,
            cwd=cwd
        )
        stdout, stderr = terminal.communicate()
        returnCode = terminal.poll()
        log.debug("\n %s", stdout)
        log.debug("\n %s", stderr)

        if returnCode > 0:
            log.error("Error. \n%s\n%s", stdout, stderr)
        return returnCode


def searchfiles(path, dryrun=False) -> list[str]:
    cmd = "ls -d " + path

    log.debug("\n %s", cmd)

    if not dryrun:
        terminal = sp.Popen(
            cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE, universal_newlines=True
        )
        stdout, stderr = terminal.communicate()
        returnCode = terminal.poll()
        log.debug("\n %s", stdout)
        log.debug("\n %s", stderr)

        files = stdout.strip("\n").split("\n")

        if returnCode > 0:
            log.error("Error. \n%s\n%s", stdout, stderr)

        return files


def sed_inplace(filename, pattern, repl):
    """
    Perform the pure-Python equivalent of in-place `sed` substitution: e.g.,
    `sed -i -e 's/'${pattern}'/'${repl}' "${filename}"`.
    """
    # For efficiency, precompile the passed regular expression.
    pattern_compiled = re.compile(pattern)

    # For portability, NamedTemporaryFile() defaults to mode "w+b" (i.e., binary
    # writing with updating). This is usually a good thing. In this case,
    # however, binary writing imposes non-trivial encoding constraints trivially
    # resolved by switching to text writing. Let's do that.
    with tempfile.NamedTemporaryFile(dir=os.getcwd(), mode='w', delete=False) as tmp_file:
        with open(filename) as src_file:
            for line in src_file:
                tmp_file.write(pattern_compiled.sub(repl, line))

    # Overwrite the original file with the munged temporary file in a
    # manner preserving file attributes (e.g., permissions).
    shutil.copystat(filename, tmp_file.name)
    shutil.move(tmp_file.name, filename)


def locate_by_pattern(filename, pattern):
    """
    Locates all instances that meet pattern and returns value from file.
    Args:
        filename: text file
        pattern: regex

    Returns:

    """
    # For efficiency, precompile the passed regular expression.
    pattern_compiled = re.compile(pattern)
    arr = []
    with open(filename) as src_file:
        for line in src_file:
            num = re.findall(pattern_compiled, line)
            if num:
                arr.append(num[0])

    return arr


def replace_line(filename, pattern, repl):
    """
        Perform the pure-Python equivalent of in-place `sed` substitution: e.g.,
        `sed -i -e 's/'${pattern}'/'${repl}' "${filename}"`.
        """
    # For efficiency, precompile the passed regular expression.
    pattern_compiled = re.compile(pattern)

    # For portability, NamedTemporaryFile() defaults to mode "w+b" (i.e., binary
    # writing with updating). This is usually a good thing. In this case,
    # however, binary writing imposes non-trivial encoding constraints trivially
    # resolved by switching to text writing. Let's do that.
    with tempfile.NamedTemporaryFile(dir=os.getcwd(), mode='w', delete=False) as tmp_file:
        with open(filename) as src_file:
            for line in src_file:
                if re.findall(pattern_compiled, line):
                    tmp_file.write(repl)
                else:
                    tmp_file.write(line)

    # Overwrite the original file with the munged temporary file in a
    # manner preserving file attributes (e.g., permissions).
    shutil.copystat(filename, tmp_file.name)
    shutil.move(tmp_file.name, filename)



def fetch_dummy_volumes(taskname, context):
    # Function generates number of dummy volumes from config or mriqc stored IQMs
    if context.config["DropNonSteadyState"] is False:
        return 0

    acq, f = metadata.find_matching_acq(taskname, context)

    if "DummyVolumes" in context.config:
        log.info("Extracting dummy volumes from acquisition: %s", acq.label)
        log.info("Set by user....Using %s dummy volumes", context.config['DummyVolumes'])
        return context.config['DummyVolumes']

    if f:
        IQMs = f.info["IQM"]
        log.info("Extracting dummy volumes from acquisition: %s", acq.label)
        if "dummy_trs_custom" in IQMs:
            log.info("Set by mriqc....Using %s dummy volumes", IQMs["dummy_trs_custom"])
            return IQMs["dummy_trs_custom"]
        else:
            log.info("Set by mriqc....Using %s dummy volumes", IQMs["dummy_trs"])
            return IQMs["dummy_trs"]

    # if we reach this point there is a problem! return error and exit
    log.error("Option to drop non-steady state volumes selected, no value passed or could be interpreted from session metadata. Quitting...")


def apply_lookup(text, lookup_table):
    if '{' in text and '}' in text:
        for lookup in lookup_table:
            text = text.replace('{' + lookup + '}', lookup_table[lookup])
    return text