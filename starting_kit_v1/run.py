#!/usr/bin/env python
# I/O defaults
import argparse
import time
import os
from sys import path
import datetime
import gc
from sklearn.linear_model import LogisticRegression
# import logging
# logging.basicConfig(level=logging.INFO)
##############

#############################
# ChaLearn AutoML2 challenge #
#############################

# Usage: python program_dir/run.py input_dir output_dir program_dir

# program_dir is the directory of this program

#
# The input directory input_dir contains 5 subdirectories named by dataset,
# including:
# 	dataname/dataname_feat.type        -- the feature type "Numerical", "Binary", or "Categorical" (Note: if this file is abscent, get the feature type from the dataname.info file)
# 	dataname/dataname_public.info      -- parameters of the data and task, including metric and time_budget
# 	dataname/dataname_test.data        -- training, validation and test data (solutions/target values are given for training data only)
# 	dataname/dataname_train.data
# 	dataname/dataname_train.solution
# 	dataname/dataname_valid.data
#
# The output directory will receive the predicted values (no subdirectories):
# 	dataname_valid.predict
# 	dataname_test.predict
# We have 2 test sets named "valid" and "test", please provide predictions for both.
#
# We implemented 2 classes:
#
# 1) DATA LOADING:
#    ------------
# Use/modify
#                  D = DataManager(basename, input_dir, ...)
# to load and preprocess data.
#     Missing values --
#       Our default method for replacing missing values is trivial: they are replaced by 0.
#       We also add extra indicator features where missing values occurred. This doubles the number of features.
#     Categorical variables --
#       The location of potential Categorical variable is indicated in D.feat_type.
#       NOTHING special is done about them in this sample code.
#     Feature selection --
#       We only implemented an ad hoc feature selection filter efficient for the
#       dorothea dataset to show that performance improves significantly
#       with that filter. It takes effect only for binary classification problems with sparse
#       matrices as input and unbalanced classes.
#
# 2) LEARNING MACHINE:
#    ----------------
# Use/modify
#                 M = MyAutoML(D.info, ...)
# to create a model.
#     Number of base estimators --
#       Our models are ensembles. Adding more estimators may improve their accuracy.
#       Use M.model.n_estimators = num
#     Training --
#       M.fit(D.data['X_train'], D.data['Y_train'])
#       Fit the parameters and hyper-parameters (all inclusive!)
#       What we implemented hard-codes hyper-parameters, you probably want to
#       optimize them. Also, we made a somewhat arbitrary choice of models in
#       for the various types of data, just to give some baseline results.
#       You probably want to do better model selection and/or add your own models.
#     Testing --
#       Y_valid = M.predict(D.data['X_valid'])
#       Y_test = M.predict(D.data['X_test'])
#
# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS".
# ISABELLE GUYON, CHALEARN, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRIGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS.
# IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL,
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS,
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE.
#
# Main contributors: Isabelle Guyon and Arthur Pesah, March-October 2014
# Lukasz Romaszko April 2015
# Originally inspired by code code: Ben Hamner, Kaggle, March 2013
# Modified by Ivan Judson and Christophe Poulain, Microsoft, December 2013
# Last modifications Isabelle Guyon, November 2017

# =========================== BEGIN USER OPTIONS ==============================
# Verbose mode:
##############
# Recommended to keep verbose = True: shows various progression messages
verbose = True  # outputs messages to stdout and stderr for debug purposes

# Debug level:
##############
# 0: run the code normally, using the time budget of the tasks
# 1: run the code normally, but limits the time to max_time
# 2: run everything, but do not train, generate random outputs in max_time
# 3: stop before the loop on datasets
# 4: just list the directories and program version
debug_mode = 0

# Time budget
#############
# Maximum time of training in seconds PER DATASET (there are 5 datasets).
# The code should keep track of time spent and NOT exceed the time limit
# in the dataset "info" file, stored in D.info['time_budget'], see code below.
# If debug >=1, you can decrease the maximum time (in sec) with this variable:
max_time = 1200

# Maximum number of cycles, number of samples, and estimators
#############################################################
# Your training algorithm may be fast, so you may want to limit anyways the
# number of points on your learning curve (this is on a log scale, so each
# point uses twice as many time than the previous one.)
# The original code was modified to do only a small "time probing" followed
# by one single cycle. We can now also give a maximum number of estimators
# (base learners).
max_cycle = 1
max_estimators = 10
max_samples = float('Inf')

# If true, the previous output directory is not overwritten, it changes name
save_previous_results = False
# Use default location for the input and output data:
# If no arguments to run.py are provided, this is where the data will be found
# and the results written to. Change the root_dir to your local directory.
root_dir = "../"
default_input_dir = root_dir + "public_dat"
default_output_dir = root_dir + "AutoML2_sample_result_submission"
default_program_dir = os.getcwd()  # root_dir + "starting_k"

# =============================================================================
# =========================== END USER OPTIONS ================================
# =============================================================================

# Version of the sample code
version = 5

# General purpose functions

overall_start = time.time()         # <== Mark starting time

the_date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")

# =========================== BEGIN PROGRAM ================================


def time_to_predict(D):
    # time to predict
    if D.info['train_num'] > 1000:
        incre = (D.info.get('valid_num', 0) + D.info['test_num']) / float(D.info['train_num'])
        d = 0.03 * incre
        if d > 0.5:
            d = 0.5
    else:
        d = 0
    return 1 - d


if __name__ == "__main__" and debug_mode < 4:
    # Check whether everything went well (no time exceeded)
    execution_success = True

    # INPUT/OUTPUT: Get input and output directory names
    # if len(argv)==1: # Use the default input and output directories if no arguments are provided
    #     input_dir = default_input_dir
    #     output_dir = default_output_dir
    #     program_dir = default_program_dir
    # else:
    #     input_dir = os.path.abspath(argv[1])
    #     output_dir = os.path.abspath(argv[2])
    #     program_dir = os.path.abspath(argv[3])

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, help="Path to input data, defult is '../public_dat'",
                        nargs='?', default=default_input_dir, const=default_input_dir)
    parser.add_argument("--output_dir", type=str, help="Path to save results, default is '../AutoML2_sample_result_submission'",
                        nargs='?', default=default_output_dir, const=default_output_dir)
    parser.add_argument("--program_dir", type=str, help="Path to codes you would like to submit, default is the current directory",
                        nargs='?', default=default_program_dir, const=default_program_dir)
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    program_dir = args.program_dir

    if verbose:
        print("Using input_dir: " + input_dir)
        print("Using output_dir: " + output_dir)
        print("Using program_dir: " + program_dir)

    # Our libraries
    path.append(program_dir + "/lib/")
    path.append(input_dir)
    import data_io                        # general purpose input/output functions
    from data_io import vprint            # print only in verbose mode
    from data_manager import DataManager  # load/save data and get info about them
    from EvoDAG.model import EvoDAGE
    from EvoDAG.model import Model
    from SparseArray import SparseArray
    import numpy as np

    def read_data(fname):
        L = None
        with open(fname) as fpt:
            for r, l in enumerate(fpt.readlines()):
                x = l.strip().split(' ')
                if L is None:
                    L = [list() for _ in x]
                for c, v in enumerate(x):
                    if v == 0:
                        continue
                    L[c].append([r, float(v)])
        return Model.convert_features([SparseArray.index_data(x, r + 1) for x in L])

    if debug_mode >= 4:  # Show library version and directory structure
        data_io.show_dir(".")

    # Move old results and create a new output directory (useful if you run locally)
    if save_previous_results:
        data_io.mvdir(output_dir, output_dir+'_'+the_date)
    data_io.mkdir(output_dir)

    # INVENTORY DATA (and sort dataset names alphabetically)
    datanames = data_io.inventory_data(input_dir)
    # Overwrite the "natural" order

    # DEBUG MODE: Show dataset list and STOP
    if debug_mode >= 3:
        data_io.show_version()
        data_io.show_io(input_dir, output_dir)
        print('\n****** Ingestion program version ' + str(version) + ' ******\n\n' + '========== DATASETS ==========\n')
        data_io.write_list(datanames)
        datanames = []  # Do not proceed with learning and testing

    # MAIN LOOP OVER DATASETS:
    overall_time_budget = 0
    time_left_over = 0
    for basename in datanames:  # Loop over datasets
        vprint(verbose,  "\n========== Ingestion program version " + str(version) + " ==========\n")
        vprint(verbose,  "************************************************")
        vprint(verbose,  "******** Processing dataset " + basename.capitalize() + " ********")
        vprint(verbose,  "************************************************")
        tmp_valid = os.path.join(program_dir, 'output', basename + '_valid.predict')
        if os.path.isfile(tmp_valid):
            os.link(tmp_valid, os.path.join(output_dir, basename + '_valid.predict'))
        tmp_test = os.path.join(program_dir, 'output', basename + '_test.predict')
        if os.path.isfile(tmp_test):
            os.link(tmp_test, os.path.join(output_dir, basename + '_test.predict'))
            vprint(verbose,  "[+] Results saved using cache")
            continue

        # ======== Learning on a time budget:
        # Keep track of time not to exceed your time budget. Time spent to inventory data neglected.
        start = time.time()

        # ======== Creating a data object with data, informations about it
        vprint(verbose,  "========= Reading training set ==========")
        D = DataManager(basename, input_dir, replace_missing=True,
                        filter_features=True, max_samples=max_samples, verbose=verbose)
        train_fname = os.path.join(input_dir, basename, basename + '_train.data')
        label_fname = os.path.join(input_dir, basename, basename + '_train.solution')
        valid_fname = os.path.join(input_dir, basename, basename + '_valid.data')
        test_fname = os.path.join(input_dir, basename, basename + '_test.data')
        X = read_data(train_fname)
        y = np.array([float(x.strip()) for x in open(label_fname).readlines()])

        # ======== Keeping track of time
        if debug_mode < 1:
            time_budget = D.info['time_budget']        # <== HERE IS THE TIME BUDGET!
        else:
            time_budget = max_time
        time_budget = float(time_budget)
        overall_time_budget = overall_time_budget + time_budget
        vprint(verbose,  "[+] Cumulated time budget (all tasks so far)  %5.2f sec" % (overall_time_budget))
        # We do not add the time left over form previous dataset: time_budget += time_left_over
        vprint(verbose,  "[+] Time budget for this task %5.2f sec" % time_budget)
        time_spent = time.time() - start
        vprint(verbose,  "[+] Remaining time after reading data %5.2f sec" % (time_budget-time_spent))

        # ========= Creating a model, knowing its assigned task from D.info['task'].
        # The model can also select its hyper-parameters based on other elements of info.
        vprint(verbose,  "======== Creating model ==========")
        time_budget = 3600
        M = EvoDAGE(n_jobs=16, classifier=True, time_limit=time_budget, fitness_function='macro-F1', Tanh=False)
        print(M)

        # ========= Iterating over learning cycles and keeping track of time
        time_spent = time.time() - start
        vprint(verbose,  "[+] Remaining time after building model %5.2f sec" % (time_budget-time_spent))
        if time_spent >= time_budget:
            vprint(verbose,  "[-] Sorry, time budget exceeded, skipping this task")
            execution_success = False
            continue
        time_predict_value = time_to_predict(D)
        time_budget = time_budget - time_spent  # Remove time spent so far
        start = time.time()                     # Reset the counter
        time_spent = 0                          # Initialize time spent learning
        M.time_limit = time_budget * time_predict_value * 0.9
        vprint(verbose,  "[+] Time budget to train the model %5.2f sec" % M._time_limit)
        Xtest = None
        if D.info['test_num'] < 1000:
            Xtest = np.array([x.hy.full_array() for x in read_data(test_fname)]).T
        M.fit(X, y, test_set=Xtest)
        # log_reg = LogisticRegression(random_state=0, class_weight='balanced')
        # log_reg.fit(M.raw_decision_function(X), y)
        vprint(verbose, "=========== " + basename.capitalize() + " Training cycle " + " ================")
        vprint(verbose, "[+] Fitting success, time spent so far %5.2f sec" % (time.time() - start))
        vprint(verbose, "[+] Size of trained model  %5.2f bytes" % data_io.total_size(M))
        # Make predictions
        # -----------------
        if os.path.isfile(valid_fname):
            Y_valid = M.predict_proba(read_data(valid_fname))[:, 1]
            # Y_valid = log_reg.predict_proba(M.raw_decision_function(read_data(valid_fname)))[:, 1]
        else:
            Y_valid = None
        if Xtest is None:
            Xtest = read_data(test_fname)
        Y_test = M.predict_proba(Xtest)[:, 1]
        # Y_test = log_reg.predict_proba(M.raw_decision_function(read_data(test_fname)))[:, 1]
        vprint(verbose,  "[+] Prediction success, time spent so far %5.2f sec" % (time.time() - start))
        # Write results
        # -------------
        filename_valid = basename + '_valid.predict'
        filename_test = basename + '_test.predict'
        if Y_valid is not None:
            data_io.write(os.path.join(output_dir, filename_valid), Y_valid)
        data_io.write(os.path.join(output_dir, filename_test), Y_test)
        vprint(verbose,  "[+] Results saved, time spent so far %5.2f sec" % (time.time() - start))
        time_spent = time.time() - start
        time_left_over = time_budget - time_spent
        vprint(verbose,  "[+] End cycle, time left %5.2f sec" % time_left_over)
        if time_left_over <= 0:
            execution_success = False
        X = None
        gc.collect()
    overall_time_spent = time.time() - overall_start
    if execution_success:
        vprint(verbose,  "[+] Done")
        vprint(verbose,  "[+] Overall time spent %5.2f sec " % overall_time_spent + "::  Overall time budget %5.2f sec" % overall_time_budget)
    else:
        vprint(verbose,  "[-] Done, but some tasks aborted because time limit exceeded")
        vprint(verbose,  "[-] Overall time spent %5.2f sec " % overall_time_spent + " > Overall time budget %5.2f sec" % overall_time_budget)
