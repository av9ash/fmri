"""
************* Method Description ******************
Author: Avinash Patil
Description: Data pre processing for feeding it to ML models
Input: List of paths to data folder for each subject
Output: Data Matrix and Label Matrix
"""

import os
import numpy as np
import scrubber


def get_raw_data(data_dir, idx, cut):
    # if len(subject)<=1:
    #
    # else
    subject = data_dir.split('/')[2]
    print("Loading data for: ", subject)

    os.chdir(data_dir)
    cwd = os.getcwd()

    meta = {}
    with open(cwd + "\\meta.data") as f:
        for line in f:
            (key, val) = line.rstrip('\n').split(':')
            meta[key] = val

    trials = []
    info_files = []

    for path, sub_dirs, files in os.walk(cwd):
        for name in sub_dirs:
            cwd = os.path.join(path, name)
            trial_data = open(cwd + "\\data.csv").read()
            # get data by lines but drop last line as it is empty
            trial_data = trial_data.split("\n")[:-1]

            # trial_data = np.array([i.split(",") for i in trial_data])
            trial_data = np.array([i.split(",") for i in trial_data])
            # trial_data.append(open(cwd + "\\info.data").read())
            # trials.append(trial_data)

            if cut:
                trials.append(trial_data[:, :4633])
            else:
                trials.append(trial_data)

            info = {}
            with open(cwd + "\\info.data") as f:
                for line in f:
                    (key, val) = line.rstrip('\n').split(':')
                    info[key] = val

            info['trial'] = idx + len(trials) - 1
            info['subject'] = subject
            info_files.append(info)

    print("Done!")
    return info_files, trials, meta


def get_all_subjects_data(subjects):
    cwd = os.getcwd()
    trials_data = []
    info = []
    meta = []
    for subject in subjects:
        info_files, trial_data, meta_files = get_raw_data(subject, len(trials_data), True)
        trials_data.extend(trial_data)
        info.extend(info_files)
        meta.extend(meta_files)
        os.chdir(cwd)

    return scrubber.clean_data(trials_data, info)


def get_subject_data(subject):
    cwd = os.getcwd()
    trials_data = []
    info = []
    meta = []

    info_files, trial_data, meta_files = get_raw_data(subject, len(trials_data), False)
    trials_data.extend(trial_data)
    info.extend(info_files)
    meta.extend(meta_files)
    os.chdir(cwd)

    return scrubber.clean_data2(trials_data, info)