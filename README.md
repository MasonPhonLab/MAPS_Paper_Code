These are the scripts associated with data extraction, model training, and model evaluation for the APhL Aligner paper. Little has been done to change the scripts from when they were run since, as every seassoned programmer knows, even small modifications can cause unexpected deviations from the results in the paper of record. The directory structure of the scripts may need to be manipulated to work with where they intend to extract data to, as they have been organized here by script type, rather than where they were run.

The TIMIT and Buckeye corpora will need to be acquired before being able to use the scripts in this repository. Additionally, if using TIMIT data directly from the Language Data Consortium download, the audio files will need to be converted from NIST to RIFF. [This StackOverflow page](https://stackoverflow.com/questions/47370167/change-huge-amount-of-data-from-nist-to-riff-wav-file) has some useful instruction on how to do so.

# 1. Data preparation

The data preparation invovles both Praat and Python scripts. The Python libraries being called are fairly stable and hopefully will not require version specificity.

It is important to note that s04 in the Buckeye data was held out for validation data through a different manner as the test set, so some attention should be paid to either get the file structure to work for the `00-data.py` script, or the speaker's phrases can be manually added to a validation folder outside of the Python script.

## 1.1. Extracting phrases from the Buckeye corpus

1. Run the `praat_scripts/buckeye_textgrids.praat` script, changing directories as needed, to convert the Buckeye transcription files to TextGrid files.

2. Run the `praat_scripts/extract_buckeye_phrases.praat` script, changing directories as needed, to extract phrases from the Buckeye recordings.

## 1.2. Extracting MFCC+delta+delta_delta features

Run the `00-data.py` script, changing directories as needed. Provided the directory structure is correct, the extraction should run without error.

# 2. Training the crisp acoustic models

The crips models can be trained using the `py_scripts/01-crisp_models.py` script. Provided correct directory structure, the models should train without additional user interaction. A directory titled `timbuck_trained_models_repetitions` must be created before the resultant models can be saved.

* For each model, a version of the model will be saved after each epoch
* After each epoch, the associated training and validation metrics will be recorded in `real_seed_crisp_3blstm_128units_bs64_rd{}_res.txt`, a tab-separated file, and where the `{}` will be replaced with the round number (or model number) of training
* Each `real_seed_crisp_3blstm_128units_bs64_rd{}_res.txt` file is appended to and not overwritten to curtail data loss; ensure previous rows are deleted if not needed

# 3. Training the sparse acoustic models

The so-called sparse models can be trained using the `py_scripts/02-sparse_models.py` script. **Before running this script**, the user should verify that the best epoch of each of the trained crisp models match the `dict` of best epochs in the script. While we have attempted to select seeds for the random number generators to make the results deterministic and have generally observed deterministic behavior of the scripts, there still seems to be occasional non-deterministic behavior, which has been a difficulty throughout the history of training neural networks.

Otherwise, as with the crisp models, provided the correct directory structure, they should train without additional user interaction.

* The `recreate_labels` and `recreate_val_labels` variables at the top of the script can be set to `False` if the tagging-version of the labels has already been created from a previous run and does not need to be made over again
* For each model, a version of the model will be saved after each epoch
* After each epoch, the associated training and validation metrics will be recorded in `full_real_seed_multiposthoc_3blstm_128units_bs64_rd{}_res.txt`, tab-separated file, and where the `{}` will be replaced with the round number (or model number) of training
* Each `full_real_seed_multiposthoc_3blstm_128units_bs64_rd{}_res.txt` file is appended to and not overwritten to curtail data loss; ensure previous rows are deleted if not needed