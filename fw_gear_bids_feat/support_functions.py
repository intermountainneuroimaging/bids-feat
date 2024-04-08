import logging
import os
import os.path as op
from pathlib import Path
import subprocess as sp
import re
import shutil
import tempfile
import errorhandler
from typing import List
import nibabel as nb
import numpy as np
from nipype.utils.filemanip import fname_presuffix
from fw_gear_bids_feat import metadata

from nipype.interfaces.fsl.maths import MeanImage, DilateImage, MathsCommand
from nipype.interfaces.fsl import BET
from nipype.interfaces.fsl.utils import ImageStats

log = logging.getLogger(__name__)

# Track if message gets logged with severity of error or greater
error_handler = errorhandler.ErrorHandler()

# -----------------------------------------------
# Support functions
# -----------------------------------------------

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

        if stderr:
            log.warning("Error. \n%s\n%s", stdout, stderr)
            returnCode = 1
        return returnCode


def searchfiles(path, dryrun=False, exit_on_errors=True) -> list[str]:
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

        if returnCode > 0 and exit_on_errors:
            log.error("Error. \n%s\n%s", stdout, stderr)

        if returnCode > 0 and not exit_on_errors:
            log.warning("Warning. \n%s\n%s", stdout, stderr)

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
    log.error(
        "Option to drop non-steady state volumes selected, no value passed or could be interpreted from session metadata. Quitting...")


def apply_lookup(text, lookup_table):
    if '{' in text and '}' in text:
        for lookup in lookup_table:
            text = text.replace('{' + lookup + '}', lookup_table[lookup])
    return text


def _remove_volumes(bold_file,n_volumes):
    if n_volumes == 0:
        return bold_file

    out = fname_presuffix(bold_file, suffix='_cut')
    nb.load(bold_file).slicer[..., n_volumes:].to_filename(out)
    return out


def _remove_timepoints(motion_file,n_volumes):
    arr = np.loadtxt(motion_file, ndmin=2)
    arr = arr[n_volumes:,...]

    filename, file_extension = os.path.splitext(motion_file)
    motion_file_new = motion_file.replace(file_extension,"_cut"+file_extension)
    np.savetxt(motion_file_new, arr, delimiter='\t')
    return motion_file_new


def _add_volumes(bold_file, bold_cut_file, n_volumes):
    """prepend n_volumes from bold_file onto bold_cut_file"""
    bold_img = nb.load(bold_file)
    bold_data = bold_img.get_fdata()
    bold_cut_img = nb.load(bold_cut_file)
    bold_cut_data = bold_cut_img.get_fdata()

    # assign everything from n_volumes forward to bold_cut_data
    bold_data[..., n_volumes:] = bold_cut_data

    # assume all values less than 1 should be 0 (rounding)
    bold_data[ bold_data < 1] = 0

    out = bold_cut_file.replace("_cut","")
    bold_img.__class__(bold_data, bold_img.affine, bold_img.header).to_filename(out)
    log.info("Trimmed nifti file saved: %s", out)
    return out


def _normalize_volumes(bold_file):
    out = fname_presuffix(bold_file, suffix='_psc')
    # with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmpdir:
    tmpdir='/flywheel/v0/work'
    tmean = MeanImage()
    tmean.inputs.in_file = bold_file
    tmean.inputs.out_file = op.join(tmpdir, "mean_func")
    log.info(tmean.cmdline)
    res = tmean.run()

    bet = BET()
    bet.inputs.in_file = op.join(tmpdir, "mean_func.nii.gz")
    bet.inputs.frac = 0.3
    bet.inputs.out_file = op.join(tmpdir, "mask")
    bet.inputs.no_output = True
    bet.inputs.mask = True
    log.info(bet.cmdline)
    res = bet.run()

    maths = MathsCommand()
    maths.inputs.in_file = bold_file
    maths.inputs.out_file = op.join(tmpdir, "bold_thres.nii.gz")
    maths.inputs.args = " -mul "+op.join(tmpdir,"mask_mask.nii.gz")
    log.info(maths.cmdline)
    res = maths.run()

    fslstats = ImageStats()
    fslstats.inputs.in_file = op.join(tmpdir, "bold_thres.nii.gz")
    fslstats.inputs.op_string = "-P 50"
    log.info(fslstats.cmdline)
    res = fslstats.run()

    value = 10000 / res.outputs.out_stat

    dil = DilateImage()
    dil.inputs.in_file = op.join(tmpdir, "mask_mask.nii.gz")
    dil.inputs.operation = "max"
    dil.inputs.out_file = op.join(tmpdir, "mask.nii.gz")
    log.info(dil.cmdline)
    res = dil.run()

    maths = MathsCommand()
    maths.inputs.in_file = bold_file
    maths.inputs.args = "-mul "+str(value)
    maths.inputs.out_file = out
    log.info(maths.cmdline)
    res = maths.run()

    log.info("Normalized by global median nifti file saved: %s", out)

    return out
