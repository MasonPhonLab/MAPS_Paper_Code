These are the scripts associated with data extraction, model training, and model evaluation for the Mason-Alberta Phonetic Segmenter (MAPS) paper. Little has been done to change the scripts from when they were run since, as every seassoned programmer knows, even small modifications can cause unexpected deviations from the results in the paper of record. The directory structure of the scripts may need to be manipulated to work with where they intend to extract data to, as they have been organized here by script type, rather than where they were run.

The TIMIT and Buckeye corpora will need to be acquired before being able to use the scripts in this repository. Additionally, if using TIMIT data directly from the Language Data Consortium download, the audio files will need to be converted from NIST to RIFF. [This StackOverflow page](https://stackoverflow.com/questions/47370167/change-huge-amount-of-data-from-nist-to-riff-wav-file) has some useful instruction on how to do so.

# 0. Data preparation

The data preparation invovles both Praat and Python scripts. The Python libraries being called are fairly stable and hopefully will not require version specificity.

It is important to note that s04 in the Buckeye data was held out for validation data through a different manner as the test set, so some attention should be paid to either get the file structure to work for the `00-data.py` script, or the speaker's phrases can be manually added to a validation folder outside of the Python script.

## 0.1. Extracting phrases from the Buckeye corpus

1. Run the `praat_scripts/buckeye_textgrids.praat` script, changing directories as needed, to convert the Buckeye transcription files to TextGrid files.

2. Run the `praat_scripts/extract_buckeye_phrases.praat` script, changing directories as needed, to extract phrases from the Buckeye recordings.

## 0.2. Extracting MFCC+delta+delta_delta features

Run the `00-data.py` script, changing directories as needed. Provided the directory structure is correct, the extraction should run without error.

# 1. Training the crisp acoustic models

The crips models can be trained using the `py_scripts/01-crisp_models.py` script. Provided correct directory structure, the models should train without additional user interaction. A directory titled `timbuck_trained_models_repetitions` must be created before the resultant models can be saved.

* For each model, a version of the model will be saved after each epoch
* After each epoch, the associated training and validation metrics will be recorded in `real_seed_crisp_3blstm_128units_bs64_rd{}_res.txt`, a tab-separated file, and where the `{}` will be replaced with the round number (or model number) of training
* Each `real_seed_crisp_3blstm_128units_bs64_rd{}_res.txt` file is appended to and not overwritten to curtail data loss; ensure previous rows are deleted if not needed

# 2. Training the sparse acoustic models

The so-called sparse models can be trained using the `py_scripts/02-sparse_models.py` script. **Before running this script**, the user should verify that the best epoch of each of the trained crisp models match the `dict` of best epochs in the script. While we have attempted to select seeds for the random number generators to make the results deterministic and have generally observed deterministic behavior of the scripts, there still seems to be occasional non-deterministic behavior, which has been a difficulty throughout the history of training neural networks.

Otherwise, as with the crisp models, provided the correct directory structure, they should train without additional user interaction.

* The `recreate_labels` and `recreate_val_labels` variables at the top of the script can be set to `False` if the tagging-version of the labels has already been created from a previous run and does not need to be made over again
* For each model, a version of the model will be saved after each epoch
* After each epoch, the associated training and validation metrics will be recorded in `full_real_seed_multiposthoc_3blstm_128units_bs64_rd{}_res.txt`, tab-separated file, and where the `{}` will be replaced with the round number (or model number) of training
* Each `full_real_seed_multiposthoc_3blstm_128units_bs64_rd{}_res.txt` file is appended to and not overwritten to curtail data loss; ensure previous rows are deleted if not needed

# 3. Creating TextGrids for model evaluation

TextGrids are created using the `03-test_aligner.py` script. As-written, the script will need to be run six times with small modification in-between. The 6 runs correspond to each pairwise combination of the data set (train, val, or test) and model type (crisp, sparse). The flags for this are the top of the script. The design is intedned as a way to force checkpoints into the evaluation so that only a subpart of the evaluation needs to be re-run in the event of an exception, hardware failure, or unexpected reboot (often caused by rogue Windows Update installations).

The `OVERWRITE` flag can be used to decide whether to resume interrupted iterations through the evaluation loop. When set to `False`, the system will only write TextGrid files that don't already exist and will skip the generation for files that already exist. When set to `True`, no files will be skipped.

This script calls out to a `Julia` script to perform the actual alignment process. This script is located at `jl_scripts/dtw_align.jl`.

# 4. Test set evaluation

To calculate network performance metrics for on the test set, run the `04-eval_test.py` script. Two files will be written, `crisp_test_res.txt` and `sparse_test_res.txt`. These files will be analyzed in top-level `R` scripts.

# 5. Calculating boundary errors

The errors for the boundaries are calculated using the `05-boundary_error.py` script. It will require there to be folders with the names "train", "val", and "test" within the same directory. Similar to the `03-test_aligner.py` sciprt, this script will need run multiple times. However, it combines the evaluation of the sparse and crisp models, so it only needs to be run 3 times, changing the flags for the train set, val set, and test set as needed. It also has an `OVERWRITE` flag that can be set as in `03-test_aligner.py`.

# 6. Metric calculations and plot generation

There are 4 `R` scripts that need to be run. Two are at the top-level. The first is `average_metrics.R`, which will calculate averages of the evaluation metrics and associated standard errors. The second is `objective_plots.R`, which will generate the plots of training performance over each epoch.

The other two `R` scripts are nested within the `py_scripts` folder. The `py_scripts/boundary_eval_res/boundary_eval.R` script will create the tables associated with the boundary error thresholds and the CDF plots. There are two associated `Julia` scripts as well that will be called by the `R` script to generate the plots used in the paper, but the script will also generate the plots with base R graphics. This script will also read in MFA results as needed.

The second final `R` script is in `py_scripts/mfa_boundary_eval_res/mfa_eval_res.R`. It will generate the table rows for the MFA evaluations and comparisons.
