"""FEAT Higher Level Analysis Module"""

import logging
import os
import os.path as op
from pathlib import Path
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

    # generate filepaths for feat run (need this for other setup steps)
    identify_feat_paths(gear_options, app_options)

    # prepare fsf design file
    generate_design_file(gear_options, app_options)

    # add dummy registration folder if needed
    if (app_options["lower_level_registration"] == "none-MNI152NLin6Asym") or (app_options["lower_level_registration"] == "none-func"):
        add_dummy_reg(gear_options, app_options)

    # generate command - "stage" commands for final run
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

