# coding=utf-8
import os
import random
import sys
import os
import scipy
import scipy.io
import json
import sklearn
import numpy as np
import time

import models
import preprocessing
import real_data_loaders
import simulated_data_generators
import splitters
import copy


CONFIGS_SET = {"channels_config", "datasets_config", "splitters_config", "models_config", "preprocess_config", "synthetic_config"}
JOB_FIELDS_SET = {"data", "channels", "targets", "preprocessing", "model", "splitter"}
JOB_PARAMETERS_SEPARATOR = "__"


def to_result_filename(data, channels, targets, preprocessing, model, splitter):
    return JOB_PARAMETERS_SEPARATOR.join([data, channels, preprocessing, model, splitter])



def from_result_filename(filename):
    assert(len(filename.split(JOB_PARAMETERS_SEPARATOR)) == len(JOB_FIELDS_SET))
    dataset, channels, preprocessing, model, cv = filename.split(JOB_PARAMETERS_SEPARATOR)
    return dataset, channels, preprocessing, model, testing


def load_json(path):
    with open(path) as file_stream:
        data = json.load(file_stream)
    return data


def load_single_config(path):
    config = load_json(path)
    config = {config_row["name"]: config_row for config_row in config}
    return config


def load_configs(configs_folder):
    configs = {}
    for config_name in CONFIGS_LIST:
        configs[config_name] = load_single_config(f"{configs_folder}/{config_name}.json")
    return configs


def load_single_job(job, configs):
    job["data"] = configs["datasets_config"][job["data"]]
    job["channels"] = configs["channels_config"][job["channels"]]
    job["splitter"] = configs["splitters_config"][job["splitter"]]
    job["model"] = configs["models_config"][job["model"]]
    job["preprocess"] = configs["preprocess_config"][job["preprocess"]]
    return job


def check_and_load_ref_functions(job):
    loader_name = job["data"]["loader"]
    assert hasattr(real_data_loaders, loader_name)
    job["data"]["loader_reference"] = getattr(real_data_loaders, loader_name)
    
    model_base_class_name = job["model"]["model_base_class"]
    assert hasattr(models, model_base_class_name)
    job["model"]["model_base_class_reference"] = getattr(real_data_loaders, model_base_class_name)
    
    splitter_name = job["splitter"]["name"]
    assert hasattr(splitters, splitter_name)
    job["splitter"]["splitter_method_reference"] = getattr(splitters, splitter_name)
    
    for index, preprecessing_stage in enumerate(job["preprocessing"]["preprocessing_pipeline"]):
        preprocessing_stage_method_name = preprecessing_stage['function_name']
        assert hasattr(preprocessing, preprocessing_stage_method_name)
        job["preprocessing"]["preprocessing_pipeline"][index]["function_reference"] = getattr(preprocessing, preprocessing_stage_method_name)
    return job


def load_and_check_jobs(jobs_filepath, configs):
    jobs = load_json(jobs_filepath)
    for index, job in enumerate(jobs):
        job["original_job"] = copy.deepcopy(job)
        assert JOB_FIELDS_SET == set(job.keys())
        output_filename = to_result_filename(**job)
        jobs[index]["output_filename"] = output_filename
        jobs[index]["output_file_path"] = f'{output_folder}/{output_filename}'
        jobs[index] = load_and_check_single_job(jobs[index])
        assert os.path.exists(jobs[index]['data']['path'])
        jobs[index] = check_and_load_ref_functions(jobs[index])
    return jobs


def process(job):
    start_time = time.time()
    output_file_path = job["output_file_path"]
    with open(output_file_path, 'w') as outfile:
        outfile.write("")

    loader = job["data"]["loader_reference"]
    filepath = job["data"]["path"]
    frequency = job["data"]["frequency"]
    X, Y = loader(filepath)
    
    channels = job["data"]["channels"]
    X = X[:, channels]
    
    targets = job["data"]["targets"]
    Y = Y[:, targets]

    data_preprocessor = job["data"]["loader_reference"]
    for preprecessing_stage in job["preprocessing"]["preprocessing_pipeline"]:
        data_preprocessor = preprecessing_stage["function_reference"]
        X, Y, frequency = data_preprocessor(X=X, Y=Y, frequency=frequency, **preprecessing_stage["kwargs"])

    output = {
        "job": job["original_job"],
        "results": []
    }
    output["job"]["start_time"] = int(start_time)

    splitter = job["splitter"]["splitter_method_reference"]
    for X_train, Y_train, X_test, Y_test in splitter(X, Y):
        start_subjob_time = time.time()
        assert len(Y_train) == len(Y_test) == 2
        assert(Y_train.shape[1:] == Y_test.shape[1:])
        assert len(X_train) == len(X_test)
        assert len(X_train) >= 2
        assert len(X_test) >= 2
        assert(X_train.shape[1:] == X_test.shape[1:])

        input_shape = X_train.shape[1:]
        output_shape = Y_train.shape[1:]

        bench_model = bench_model_class(input_shape=input_shape, output_shape=output_shape, frequency=frequency, **model_config["kwargs"])
        bench_model.fit(X_train, Y_train, X_test, Y_test)

        Y_train_predicted = bench_model.predict(X_train)
        Y_test_predicted = bench_model.predict(X_test)

        Y_train_sliced = bench_model.slice_target(Y_train)
        Y_test_sliced = bench_model.slice_target(X_test)

        assert len(Y_train_predicted) == len(Y_train_sliced) == len(Y_test_predicted) == len(Y_test_sliced) == 2
        assert Y_train_predicted.shape == Y_train_sliced.shape
        assert Y_test_predicted.shape == Y_test_sliced.shape

        end_subjob_time = time.time()
        result = {
            "Y_train_predicted": Y_predicted.tolist(),
            "Y_train_sliced": Y_train_sliced.tolist(),
            "Y_test_predicted": Y_test_predicted.tolist(),
            "Y_test_sliced": Y_test_sliced.tolist(),
            "execution_time": float(end_subjob_time - start_subjob_time)
        }
        output["results"].append(result)
    end_time = time.time()
    output["job"]["execution_time"] = float(end_time - start_time)
    with open(output_file_path, 'w') as outfile:
        json.dump(output, outfile)


def update_jobs(jobs):
    new_jobs = []
    for job in jobs:
        if not os.path.exists(job["output_file_path"]):
            new_jobs.append(job)
    random.shuffle(new_jobs)
    return new_jobs


def runner(output_folder, jobs_filepath, configs_folder):
    configs = load_configs(configs_folder)
    jobs = load_and_check_jobs(jobs_filepath, configs)

    while len(jobs) > 0:
        jobs = update_jobs(jobs)
        if len(jobs) == 0:
            sys.stderr.write('All Jobs Are DONE!\n')
            return
        else:
            sys.stderr.write("Jobs left: {}\n\n".format(len(jobs)))
        job_to_process = jobs[0]
        output_filename = job_to_process["output_filename"]
        sys.stderr.write("Process start: {}\n".format(output_filename))
        process.process(job_to_process)
        sys.stderr.write("Process finished: {}\n".format(output_filename))
