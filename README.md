

# bids-feat
(HPC Compatible) FSL's FEAT (FMRI Expert Analysis Tool). As implemented in this Gear, FEAT first level analysis will act on any generic bids compliant pipeline outputs. Voxelwise activation analyses for a single (or multiple) task will be generated. A template FSF design file is required, and all processing steps indicated in the design file will be followed.

## Overview
This gear should be run on preprocessed datasets that are in a BIDS derivative format (BIDS derivative info here). Two example compliant preprocessing flywheel gears are: (1) bids-fmriprep and (2) bids-hcp v.^1.2.5_4.3.0_inc1.5.1. The gear leverages the expected output format of a BIDS derivative dataset with a required gear input "FSF_TEMPLATE" to assign fmri preprocessed images, high resolution T1 images, confounds, and event files within a FEAT first level analysis. All design choices are set within the template file including registration steps, filtering and smoothing options, explanatroy variable models, and more. The bids-feat gear is capable of running a full FEAT 1st level analysis. Information leveraged from the flywheel database include the subject and session id as well as stored event files if no event files are directly passed as gear inputs. Multiple runs can also be included as long as all runs use the same template design file (see further instructions below on running multiple acquisitions in the gear).


## Important Notes
Many assumptions are made when generating the job design file. Please closely inspect the design file generated from the gear in "dry run" mode before proceeding. If any files appear incorrectly mapped, make adjustments to the template design file before proceeding. Review the Troubleshooting section for more information on identifying and correcting issues. Most importantly, make sure naming is consistent between your acquisiton label in flywheel, bids-feat configuration setting "task-list", and task naming in the BIDS derivative dataset used as gear input. In addition, event naming is critical, wherein the event naming used in a BIDS formatted event file or file name contain the explanatory variable name used in the template design file. Please see "Building a Template Design File below for examples.




## Required inputs

`preprocessing-pipeline-zip`: Select preprocessing output directory zip. Preprocessing outputs must be in bids derivative format. Example compatible pipelines: fmriprep, bids-hcp.

`FSF_TEMPLATE`: FSL DESIGN FILE that will be used as the template for all analyses. Record all common processing decisions in this file, for example slice-timing correction, intensity normization, EV naming and design.


## Optional inputs

`additional-input-one`: (Optional) Additional preprocessing output directory. Preprocessing outputs must be in bids derivative format. 

`confounds-file`: (Optional) Additional input used as confound timeseries in FSL FEAT. If no input is passed, default confounds file is used from bids derivative directory '*counfound_timeseries.tsv'.

`event-file`: Explanatory variable (EVs) custom text files. Identify in config options the event files type (BIDS-Formatted|FSL-3 Column Format|FSL-1 Entry Per Volume). If not event file is passed, events will be downloaded from flywheel acquisition.

## Configuration 

`task-list`: Comma seperated list of tasks may be selected for FEAT 1st level analysis. Task name must be consistent with naming used in preprocessing package (e.g task-abc_run-01, task-abc_run-02).

`events-suffix`: suffix used to select correct events file from bids curated dataset. Events may be pulled directly from acquisition container if no event file is passed as input.

`output-name`: [NAME].feat directory name. If left blank, output name will be drawn from the fsf template file.

`confound-list`: Comma seperated list of components to be included in feat glm confounds. Confound timeseries will be pulled from confound file in inputs if passed, otherwise defaults to using bids derivative file '*counfound_timeseries.tsv'. Example entry: rot_x, rot_y, rot_z, trans_x, trans_y, trans_z. If left blank no confounds will be included in feat analysis.

`DropNonSteadyState`: set whether or not to remove XX number of inital non-steady state volumes. If no value is passed in 'DummyVolumes', non-steady state number is taken from mriqc IQMs, if neither are defined, an error will be returned.


`DummyVolumes`: Number of dummy volumes to ignore at the beginning of scan. Leave blank if you want to use the non-steady state volumes recorded in mriqc IQMs.

`multirun`: select to concatenate all tasks in 'task-list' before running feat analysis. Otherwise each run will be run independently.

`evformat`: (Default: BIDS-Formatted) Select type of file where events are stored. Selected format should match format provided in the events input file, or files downloaded directly from flywheel acquisition. If format passed is not FSL compatible, update format to match FSL requirements before running analysis. Options: BIDS-Formatted|FSL-3 Column Format|FSL-1 Entry Per Volume.

`gear-log-level`: Gear Log verbosity level (ERROR|WARNING|INFO|DEBUG)

`gear-dry-run`: Do everything except actually executing gear

`gear-writable-dir`: Gears expect to be able to write temporary files in /flywheel/v0/.  If this location is not writable (such as when running in Singularity), this path will be used instead.  The gear creates a large number of files so this disk space should be fast and local.

`slurm-cpu`: [SLURM] How many cpu-cores to request per command/task. This is used for the underlying '--cpus-per-task' option. If not running on HPC, then this flag is ignored

`slurm-ram`: [SLURM] How much RAM to request. This is used for the underlying '--mem-per-cpu' option. If not running on HPC, then this flag is ignored

`slurm-ntasks`: [SLURM] Total number of tasks/commands across all nodes (not equivalent to neuroimaging tasks). Using a value greater than 1 for code that has not been parallelized will not improve performance (and may break things).

`slurm-nodes`: [SLURM] How many HPC nodes to run on

`slurm-partition`: [SLURM] Blanca, Alpine, or Summit partitions can be entered

`slurm-qos`: [SLURM] For Blanca the QOS has a different meaning, ie blanca-ics vs blanca-ibg, etc. For Alpine and Summit, the QOS should be set to normal if running a job for 1 day or less, and set to long if running a job with a maximum walltime of 7 days

`slurm-account`: [SLURM] For Blanca the ACCOUNT should be set to the sub-account of choice (e.g. blanca-ics-rray). For Alpine, the account should be set to ucb-general, or the specialized account granted by RC: ucb278_asc1

`slurm-time`: [SLURM] Maximum walltime requested after which your job will be cancelled if it hasn't finished. Default to 1 day


## Building a Template Design File (*.fsf)
The flywheel gear is built around the use of a template FEAT design.fsf file to generate a new 1st level feat model. Three parts of the design file are critical to organize correctly to ensure the Flywheel gear can correctly interpret the intended feat design.

- **FEAT configurations.** The design file should outline all configurations / feat settings such as temporal filtering, registration method, etc.
- **FEAT input file paths.** Placeholders for each input file should be added to the design file. The Flywheel gear uses a generic "Lookup" table to replace placeholder arugments in the filepath. Currently the following lookup table placeholders are recognized:   
  >   `PIPELINE`   # name of the preprocessing parent directory (e.g. fmriprep)  
  > `SUBJECT`      # subject label in flywheel (attached to current analysis, e.g. 001)  
  > `SESSION`      # session label in flywheel (attached to current analysis, e.g. S1)  
  > `TASK`         # task name passed in task list  
  > `WORKDIR`      # placeholder for the work directory where analysis is run 

    Putting it all together, the file paths should look something like:  
`{WORKDIR}/{PIPELINE}/sub-{SUBJECT}/ses-{SESSION}/func/sub-{SUBJECT}_ses-{SESSION}_task-{TASK}_space-MNI152NLin6Asym_desc-preproc_bold.nii.gz`  

**Important Notes:**  
The file paths including the placeholders must follow the template shown above exactly, each placeholder variable must be written with {} and in all upper case. If additional lookup table variables are needed, please contact the gear developers.  

- **FEAT explanatory variables.** Sections for each explanatory variable must be included in the template design.fsf. Importantly, the label of each explanatory variable in the template file should match the label used in the events file. For example, if using a BIDS-Formatted event file, your inputs should look like...  

**`func-bold_task-experiment_run-01_desc-block_events.tsv`**

| Onset | Duration | trial_type |  
|-------|----------|------------|
| 10    | 3        | conditionA |
| 22    | 5        | conditionB |
| 35    | 3        | conditionA |
| 65    | 3        | conditionA |

**Code Snippet** `design.fsf` 

```
# EV 1 title
set fmri(evtitle1) "conditionA"

# Basic waveform shape (EV 1)
# 0 : Square
# 1 : Sinusoid
# 2 : Custom (1 entry per volume)
# 3 : Custom (3 column format)
# 4 : Interaction
# 10 : Empty (all zeros)
set fmri(shape1) 3

# Convolution (EV 1)
# 0 : None
# 1 : Gaussian
# 2 : Gamma
# 3 : Double-Gamma HRF
# 4 : Gamma basis functions
# 5 : Sine basis functions
# 6 : FIR basis functions
# 8 : Alternate Double-Gamma
set fmri(convolve1) 3

# Convolve phase (EV 1)
set fmri(convolve_phase1) 0

# Apply temporal filtering (EV 1)
set fmri(tempfilt_yn1) 1

# Add temporal derivative (EV 1)
set fmri(deriv_yn1) 1

# Custom EV file (EV 1)
set fmri(custom1) "ses-01_task-experiment_run-01_desc-block_events-conditionA.txt"

# Orthogonalise EV 1 wrt EV 0
set fmri(ortho1.0) 0

# Orthogonalise EV 1 wrt EV 1
set fmri(ortho1.1) 0

# Orthogonalise EV 1 wrt EV 2
set fmri(ortho1.2) 0
```

As you can see from the example above, the conditions labeled in the *events.tsv file should exactly match the explanatory variable name within the design.fsf template.

### Tips for Success:
The best way to create a template design file is to first generate an example feat model on your local computer. Within the FSL FEAT user interface, make all the selections desired for the first level analysis. Point to example input files. Add all the desired explanatory variables, and contrasts. Once you are satisfied you have created the correct feat model design, safe the design configuration from the user interface. Open the "design.fsf" file in a text editor. Locate all the file paths inserted in the file, and update these paths with the lookup table placeholders where necessary. Save and close the final design template.  

## Building Event Files
The flywheel bids-feat gear is designed to act on three types of event files: 
1. BIDS-Formatted
2. FSL-3 Column Format
3. FSL-1 Entry Per Volume  

Familiarize your self with the format of each file type by exploring the FSL FEAT documentation <link> and BIDS specifications <link>. Examples of each file type are also included in the "examples" directory in this project. Be sure that the template design file configuration matches the type of input event files selected for the analysis. Be aware, BIDS-Formatted event files (see example above) will be automatically converted to FSL-3 Column Format for the FEAT analysis. 

## Multiple Acquisitions
While this gear is written to act only on a single session using a single design template, the analysis can be run on multiple acquisitions. When the same design file can be applied to multiple acquisitions (e.g. run-01, run-02) and the event conditions are the same across runs (e.g. conditionA and conditionB modeled in both runs), a user can run all first level analyses in the same flywheel job.

`task-list` configuration is used to indicate if multiple acquisitions should be included in the flywheel job, passed as a comma seperated list.

If the user wishes to **concatenate** multiple runs into a single time series before first level analysis, this can also be achieved by setting `multirun` configuration to `true`. If a user wishes to use the `multirun` option, they **must** pass the events file as an external input to the gear where the concatenated events are time adjusted and correctly labeled for each run. The input nifti timeseries and confounds file will be generated automatically with the flywheel gear.

## Troubleshooting

Use `gear-dry-run` to generate all the necessary inputs and design file but do not run the analysis. Check the design file contains all relevant inputs and settings. Test the design file locally by changing or removing the absolute file path in the design file. Watch for FEAT errors in the report.


