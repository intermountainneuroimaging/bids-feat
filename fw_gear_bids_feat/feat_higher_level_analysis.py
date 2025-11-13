"""FEAT Higher Level Analysis Module"""

import logging
import os
import os.path as op
from pathlib import Path
import pandas as pd
import numpy as np
import shutil
from zipfile import ZIP_DEFLATED, ZipFile
import errorhandler
from flywheel_gear_toolkit import GearToolkitContext

from utils.command_line import exec_command
from utils.feat_html_singlefile import main as flathtml
from fw_gear_bids_feat.support_functions import generate_command, execute_shell, searchfiles, sed_inplace, locate_by_pattern, replace_line, fetch_dummy_volumes, apply_lookup
log = logging.getLogger(__name__)

# Track if message gets logged with severity of error or greater
error_handler = errorhandler.ErrorHandler()

def run(gear_options: dict, app_options: dict, gear_context: GearToolkitContext) -> int:
    """Run module for higher level analysis run option

    Arguments:
        gear_options: dict with gear-specific options
        app_options: dict with options for the BIDS-App

    Returns:
        run_error: any error encountered running the app. (0: no error)
    """

    # pseudo code...
    # 1. find paths for each input outlined in template...
    # 2. check for registration folders... assume identity reg if none included?
    # 3. run?
    # 4. cleanup and zip

    commands = []

    # start from working directory for all commands...
    os.chdir(gear_options["work-dir"])

    # generate filepaths for feat run (need this for other setup steps)
    identify_feat_paths(gear_options, app_options)

    # prepare fsf design file
    generate_design_file(gear_options, app_options)

    # add dummy registration folder if needed
    if (app_options["lower_level_registration"] == "none-MNI152NLin6Asym") or (app_options["lower_level_registration"] == "none-func"):
        add_dummy_reg(gear_options, app_options)

    # if missing evs were allowed in first stage - check for excluded copes
    if gear_options['preproc_gear'].job.config["config"]['allow-missing-evs']:
        single_cope_design_files = setup_higher_level_analysis(app_options["design_file"])
        for design_file in single_cope_design_files:
            app_options["design_file"] = design_file
            # generate command - "stage" commands for final run
            commands.append(generate_command(gear_options, app_options))
    else:
        commands.append(generate_command(gear_options, app_options))

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
        featdirs = searchfiles(os.path.join(gear_options["work-dir"], "*.gfeat"))

        for featdir in featdirs:
            # Create output directory
            if app_options["sesid"]:
                output_analysis_id_dir = os.path.join(gear_options["destination-id"], app_options["pipeline"], "feat",
                                                      "sub-" + app_options["sid"], "ses-" + app_options["sesid"])
            elif app_options["sid"]:
                output_analysis_id_dir = os.path.join(gear_options["destination-id"], app_options["pipeline"], "feat",
                                                      "sub-" + app_options["sid"])
            else:
                output_analysis_id_dir = os.path.join(gear_options["destination-id"], app_options["pipeline"], "feat"
                                                      )
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
                                   os.path.basename(featdir.replace(".gfeat", "-")) + "report.html.zip")
            with ZipFile(outpath, "w", compression=ZIP_DEFLATED) as zf:
                zf.write(inpath, os.path.basename(inpath))

            # shutil.copy(os.path.join(featdir, "report.html.zip"), os.path.join(gear_options["output-dir"], featdir.replace(".feat","-") + "report.html.zip"))
            shutil.copy(os.path.join(featdir, "design.fsf"),
                        os.path.join(gear_options["output-dir"], os.path.basename(
                            featdir.replace(".gfeat", "-")) + "design.fsf"))

        cmd = "zip -q -r " + os.path.join(gear_options["output-dir"],
                                       "gfeat_" + str(gear_options["destination-id"])) + ".zip " + gear_options[
                  "destination-id"]
        execute_shell(cmd, dryrun=gear_options["dry-run"], cwd=gear_options["work-dir"])

        cmd = "chmod -R a+rwx " + os.path.join(gear_options["output-dir"])
        execute_shell(cmd, dryrun=gear_options["dry-run"], cwd=gear_options["output-dir"])

    else:
        shutil.copy(app_options["design_file"], os.path.join(gear_options["output-dir"], "design.fsf"))

    return run_error


def setup_higher_level_analysis(design_file):
    """
    Function checks all lower level directories for missing events, identifies copes that can be used in higher level
    analysis and creates single cope .gfeat design files with lower level models containing relevant copes
    Args:
        design_file: design file (should contain all relevant paths)

    Returns: list of single cope design files - use for analysis

    """
    # pull all lower level cope directories being used for analysis
    lower_feat = pd.DataFrame(locate_by_pattern(design_file, r'set feat_files\((\d+)\) "(.*)"'),
                              columns=["num", "path"])
    problem_contrasts = []
    for idx, row in lower_feat.iterrows():
        problem_contrasts.append(get_problem_contrasts(row["path"]))
        lower_level_contrasts = get_lower_contrasts(row["path"])

    # after putting all relevant input paths... make sure analysis is single cope
    lower_cope_key = pd.DataFrame(locate_by_pattern(design_file, r'set fmri\(copeinput.(\d+)\) (.*)'),
                                  columns=["num", "value"])

    # list of design files
    out_design_files = []

    for idx, row in lower_cope_key.iterrows():
        # set all lower level copes to 0
        lower_cope_key["value"] = 0
        # next set lower level cope of interest to 1
        lower_cope_key.loc["value", idx] = 1
        # pull cope name for labels
        conname = lower_level_contrasts["name"][idx]
        output_name_onecope = locate_by_pattern(design_file, r'set fmri\(outputdir\) "(.*)"')[0] + "." + str(
            idx + 1).zfill(2) + "." + conname
        design_file_onecope = design_file.replace('.fsf', "." + str(idx + 1).zfill(2) + "." + conname + ".fsf")

        # assign new values...
        shutil.copy(design_file, design_file_onecope)

        replace_line(design_file_onecope, r'set fmri\(outputdir\)',
                     'set fmri(outputdir) "' + output_name_onecope + '"\n')

        for i, r in lower_cope_key.iterrows():
            if idx == i:
                replace_line(design_file_onecope, r'set fmri\(copeinput.' + str(r["num"]) + '\)',
                             'set fmri(copeinput.' + str(r["num"]) + ') 1\n')
            else:
                replace_line(design_file_onecope, r'set fmri\(copeinput.' + str(r["num"]) + '\)',
                             'set fmri(copeinput.' + str(r["num"]) + ') 0\n')
            # print(r'set fmri\(copeinput.' + str(r["num"]) + '\)')
        # pull all the ev assignments...
        higher_ev_key = pd.DataFrame(locate_by_pattern(design_file, r'set fmri\(evg(\d+).(\d+)\) (.*)'),
                                     columns=["input", "ev", "value"])

        # set zero for all problem contrasts
        for i, r in lower_feat.iterrows():
            # get probblem contransts for input
            itr_prob_contrasts = problem_contrasts[i]
            # print(itr_prob_contrasts)
            if conname in itr_prob_contrasts:
                higher_ev_key["value"][higher_ev_key["input"] == r[0]] = 0

        # check there is at least one lower level with results... if no valid lower level results - do not run higher level analysis
        if all(higher_ev_key["value"] == 0):
            print(conname + " cannot be used in secondary analysis")
            os.remove(design_file_onecope)
            continue

        # apply value changes based on problem contrasts in lower level
        for i, r in higher_ev_key.iterrows():
            replace_line(design_file_onecope, r'set fmri\(evg' + str(r["input"]) + "." + str(r["ev"]) + '\)',
                         'set fmri(evg' + str(r["input"]) + "." + str(r["ev"]) + ') ' + str(r["value"]) + '\n')

        out_design_files.append(design_file_onecope)

    return out_design_files


def identify_feat_paths(gear_options: dict, app_options: dict):
    """
    Identify all placeholders in the fsf design file. Use with filemapper to point to each filepath.
    Args:
        gear_options (dict): options for the gear, from config.json
        app_options (dict): options for the app, from config.json

    Returns:
        app_options (dict): updated options for the app, from config.json
    """

    app_options["input_feat_directories"] = []

    design_file = gear_options["FSF_TEMPLATE"]

    # check provided template matches the analysis type set in the configs
    analysis_level = locate_by_pattern(design_file, r'set fmri\(level\) (.*)')

    if analysis_level[0] != "2":
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
                    "SESSION": app_options["sesid"]}

    # gather number of inputs:
    input_numbers = locate_by_pattern(design_file, r'set feat_files\((\d+)')

    for num in input_numbers:
        filename = locate_by_pattern(design_file, r'set feat_files\(' + num + '\) "(.*)"')
        inputfile = apply_lookup(filename[0], lookup_table)

        if not searchfiles(inputfile):
            log.error("Unable to locate functional file...exiting.")
        else:
            log.info("Using feat directory: %s", inputfile)
        app_options["input_feat_directories"].append(inputfile)

    # check registration (lower level)
    lower_design_file = op.join(app_options["input_feat_directories"][0], "design.fsf")

    app_options["lower_level_registration"] = "unknown"
    highres_yn = locate_by_pattern(lower_design_file, r'set fmri\(reghighres_yn\) (.*)')
    regstandard_yn = locate_by_pattern(lower_design_file, r'set fmri\(regstandard_yn\) (.*)')
    lower_func_file_name = locate_by_pattern(lower_design_file, r'set feat_files\(1\) "(.*)"')
    if not int(highres_yn[0]) and not int(regstandard_yn[0]):
        # no registration is applied - check if input is in MNINonLinear space (same as FSL tempalte)
        if "MNI152NLin6Asym" in lower_func_file_name[0]:
            # we can apply mumford registration hack after analysis is complete
            app_options["lower_level_registration"] = "none-MNI152NLin6Asym"
        elif "space-func" in lower_func_file_name[0]:
            # we can apply mumford registration hack after analysis is complete
            app_options["lower_level_registration"] = "none-func"
        else:
            app_options["lower_level_registration"] = "none-other"

    # if output name not given in config - use output name from template
    if not app_options["output-name"]:
        output_name = locate_by_pattern(design_file, r'set fmri\(outputdir\) "(.*)"')
        app_options["output-name"] = Path(output_name[0]).name

    return app_options


def generate_design_file(gear_options: dict, app_options: dict):
    """
    Use paths and configuration settings to fill template for feat run

    Args:
        gear_options (dict): options for the gear, from config.json
        app_options (dict): options for the app, from config.json

    Returns:
        app_options (dict): updated options for the app, from config.json
    """

    design_file = os.path.join(gear_options["work-dir"], os.path.basename(gear_options["FSF_TEMPLATE"]))
    app_options["design_file"] = design_file

    shutil.copy(gear_options["FSF_TEMPLATE"], design_file)

    replace_line(design_file, r'set fmri\(outputdir\)',
                 'set fmri(outputdir) "' + os.path.join(gear_options["work-dir"], os.path.basename(app_options["output-name"])) + '"')

    # 2. lower level feat paths
    for idx, x in enumerate(app_options["input_feat_directories"]):
        replace_line(design_file, r'set feat_files\(' + str(idx+1) + '\)',
                     'set feat_files(' + str(idx+1) + ') "' + x + '"')

    return app_options


def add_dummy_reg(gear_options: dict, app_options: dict):

    # 1. copy $FSLDIR/etc/flirtsch/ident.mat -> reg/example_func2standard.mat
    # 2. overwrite the standard.nii.gz image with the mean_func.nii.gz

    for p in app_options["input_feat_directories"]:
        if not os.path.exists(op.join(p, "reg")):
            os.makedirs(op.join(p, "reg"))

            # copy placeholder identity matrix to registration folder (will apply no registration at higher level analysis
            shutil.copy(op.join(os.environ["FSLDIR"], "etc", "flirtsch", "ident.mat"),
                        op.join(p, "reg", "example_func2standard.mat"))

            # copy other stand-in reg files
            if app_options["lower_level_registration"] == "none-func":
                shutil.copy(op.join(p, "mean_func.nii.gz"),
                            op.join(p, "reg", "example_func2standard.nii.gz"))
                shutil.copy(op.join(p, "mean_func.nii.gz"),
                            op.join(p, "reg", "standard2example_func.nii.gz"))
                shutil.copy(op.join(p, "mean_func.nii.gz"),
                            op.join(p, "reg", "standard.nii.gz"))
            elif app_options["lower_level_registration"] == "none-MNI152NLin6Asym":

                shutil.copy(op.join(os.environ["FSLDIR"], "data", "standard", "MNI152_T1_2mm_brain.nii.gz"),
                            op.join(p, "reg", "standard.nii.gz"))

                # image registered in preproc shoul be in MNI152 space,
                #  ...but not the same voxel size...
                #  ...need to apply flirt first...
                cmd = op.join(os.environ["FSLDIR"],"bin","flirt") + " -in " + op.join(p, "mean_func.nii.gz") + \
                      " -ref " + "standard.nii.gz" + \
                      " -out " + "example_func2standard.nii.gz" + " -applyxfm"
                execute_shell(cmd, dryrun=gear_options["dry-run"], cwd=op.join(p, "reg"))

            else:
                continue

            # create pngs for reports
            cmd = op.join(os.environ["FSLDIR"],"bin","slicer") + " example_func2standard standard -s 2 -x 0.35 sla.png " \
                                                                 "-x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y " \
                                                                 "0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y " \
                                                                 "0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z " \
                                                                 "0.55 slk.png -z 0.65 sll.png "
            execute_shell(cmd, dryrun=gear_options["dry-run"], cwd=op.join(p, "reg"))

            cmd = op.join(os.environ["FSLDIR"],"bin","pngappend") + " sla.png + slb.png + slc.png + sld.png + sle.png " \
                                                                    "+ slf.png + slg.png + slh.png + sli.png + " \
                                                                    "slj.png + slk.png + sll.png " \
                                                                    "example_func2standard1.png "
            execute_shell(cmd, dryrun=gear_options["dry-run"], cwd=op.join(p, "reg"))

            cmd = op.join(os.environ["FSLDIR"],"bin","slicer") + " standard example_func2standard -s 2 -x 0.35 sla.png " \
                                                                 "-x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y " \
                                                                 "0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y " \
                                                                 "0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z " \
                                                                 "0.55 slk.png -z 0.65 sll.png "
            execute_shell(cmd, dryrun=gear_options["dry-run"], cwd=op.join(p, "reg"))

            cmd = op.join(os.environ["FSLDIR"],"bin","pngappend") + " sla.png + slb.png + slc.png + sld.png + sle.png " \
                                                                    "+ slf.png + slg.png + slh.png + sli.png + " \
                                                                    "slj.png + slk.png + sll.png " \
                                                                    "example_func2standard2.png "
            execute_shell(cmd, dryrun=gear_options["dry-run"], cwd=op.join(p, "reg"))

            cmd = op.join(os.environ["FSLDIR"],"bin","pngappend") + " example_func2standard1.png - " \
                                                                    "example_func2standard2.png " \
                                                                    "example_func2standard.png "
            execute_shell(cmd, dryrun=gear_options["dry-run"], cwd=op.join(p, "reg"))

            cmd = "/bin/rm -f sl?.png example_func2standard2.png"
            execute_shell(cmd, dryrun=gear_options["dry-run"], cwd=op.join(p, "reg"))


def get_problem_contrasts(lower_feat_dir):
    """
    Return list of contrasts that were computed using zero frame EVs these should not be included in higher level analyses.
    Args:
        lower_feat_dir: path to .feat directory

    Returns: list of problem copes

    """
    design_file = os.path.join(lower_feat_dir, "design.fsf")
    conmat = os.path.join(lower_feat_dir, "design.con")
    frfmat = os.path.join(lower_feat_dir, "design.frf")

    # locate all evtitle calls in template
    ev_key = locate_by_pattern(design_file, r'set fmri\(evtitle(\d+)\) "(.*)"')
    ev_files = locate_by_pattern(design_file, r'set fmri\(custom(\d+)\) "(.*)"')
    ev_shape = locate_by_pattern(design_file, r'set fmri\(shape(\d+)\) (.*)')

    # check for empty evs
    empty_ev_check_1 = [k[1] == '10' for k in ev_shape]
    empty_ev_check_2 = ["zeros.txt" == os.path.basename(k[1]) for k in ev_files]

    if not empty_ev_check_1 == empty_ev_check_2:
        # log.error("Checks for lower level missing evs not sucessful. Not sure how to proceed. Exiting")
        print("Checks for lower level missing evs not sucessful. Not sure how to proceed. Exiting")

    # read in contrasts matrix (add relevant labels)
    arr = np.loadtxt(forward_csv(open(conmat), "/Matrix"), skiprows=1)
    lower_level_contrasts = pd.DataFrame(locate_by_pattern(conmat, r'[+-/]ContrastName(\d+)[ \t\n\r\f\v](.*)'),
                                         columns=["num", "name"])
    lower_level_contrasts["name"] = [s.rstrip() for s in lower_level_contrasts["name"]]
    evs = pd.read_csv(frfmat, header=None, dtype=str)

    # assign labels to con matrix
    contrasts = pd.DataFrame(arr, index=list(lower_level_contrasts["name"]), columns=list(evs[0]))

    # find impacted contrasts...
    ev_key_df = pd.DataFrame(ev_key, columns=["num", "name"])
    copes = []
    for (columnName, columnData) in contrasts.items():
        if int(columnName) <= len(empty_ev_check_1):
            flag = empty_ev_check_1[int(columnName) - 1]
            # if ev is empty - check for "unreliable" contrasts
            if flag:
                copes.extend(list(columnData[columnData > 0].index))

    copes = pd.Series(copes).drop_duplicates().tolist()

    mask = [c in copes for c in lower_level_contrasts["name"]]
    problem_copes = lower_level_contrasts[mask]

    return list(problem_copes["name"])


def get_lower_contrasts(lower_feat_dir):
    conmat = os.path.join(lower_feat_dir, "design.con")
    lower_level_contrasts = pd.DataFrame(locate_by_pattern(conmat, r'[+-/]ContrastName(\d+)[ \t\n\r\f\v](.*)'),
                                         columns=["num", "name"])
    lower_level_contrasts["name"] = [s.rstrip() for s in lower_level_contrasts["name"]]
    return lower_level_contrasts


def forward_csv(f, prefix):
    pos = 0
    while True:
        line = f.readline()
        if not line or line.startswith(prefix):
            f.seek(pos)
            return f
        pos += len(line.encode('utf-8'))

