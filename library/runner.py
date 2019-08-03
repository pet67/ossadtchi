import copy
import json
import os
import random
import sys
import time

import numpy as np

from . import models
from . import preprocessing
from . import real_data_loaders
# from . import simulated_data_generators # Not ready yet
from . import splitters


CONFIGS_SET = {
    "channels_config", "targets_config", "datasets_config",
    "splitters_config", "models_config", "preprocess_config", "synthetic_config"
}
JOB_FIELDS_SET = {"data", "channels", "targets", "preprocess", "model", "splitter"}
JOB_PARAMETERS_SEPARATOR = "__"


def to_result_filename(data, channels, targets, preprocess, model, splitter):
    filename = JOB_PARAMETERS_SEPARATOR.join([data, channels, targets, preprocess, model, splitter])
    assert len(filename.split(JOB_PARAMETERS_SEPARATOR)) == len(JOB_FIELDS_SET)
    assert len(filename) <= 255, f"Too long filename {len(filename)}"
    return filename


def from_result_filename(filename):
    assert len(filename.split(JOB_PARAMETERS_SEPARATOR)) == len(JOB_FIELDS_SET), filename
    data, channels, targets, preprocess, model, splitter = filename.split(JOB_PARAMETERS_SEPARATOR)
    return data, channels, targets, preprocess, model, splitter


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
    for config_name in CONFIGS_SET:
        configs[config_name] = load_single_config(f"{configs_folder}/{config_name}.json")
    return configs


def load_single_job(job, configs):
    job["data"] = configs["datasets_config"][job["data"]]
    job["channels"] = configs["channels_config"][job["channels"]]
    job["targets"] = configs["targets_config"][job["targets"]]
    job["splitter"] = configs["splitters_config"][job["splitter"]]
    job["model"] = configs["models_config"][job["model"]]
    job["preprocess"] = configs["preprocess_config"][job["preprocess"]]
    return job


def check_and_load_ref_functions(job):
    loader_name = job["data"]["data_loader"]
    assert hasattr(real_data_loaders, loader_name)
    job["data"]["loader_reference"] = getattr(real_data_loaders, loader_name)

    model_base_class_name = job["model"]["model_base_class"]
    assert hasattr(models, model_base_class_name)
    job["model"]["model_base_class_reference"] = getattr(models, model_base_class_name)

    splitter_name = job["splitter"]["name"]
    assert hasattr(splitters, splitter_name)
    job["splitter"]["splitter_method_reference"] = getattr(splitters, splitter_name)

    for index, preprecessing_stage in enumerate(job["preprocess"]["preprocessing_pipeline"]):
        preprocessing_stage_method_name = preprecessing_stage['function_name']
        assert hasattr(preprocessing, preprocessing_stage_method_name)
        job["preprocess"]["preprocessing_pipeline"][index]["function_reference"] = getattr(preprocessing, preprocessing_stage_method_name)
    return job


def load_and_check_jobs(jobs_filepath, configs, output_folder):
    jobs = load_json(jobs_filepath)
    for index, job in enumerate(jobs):
        assert JOB_FIELDS_SET == set(job.keys()), f"Wrong fields diff: {JOB_FIELDS_SET ^ set(job.keys())}"
        output_filename = to_result_filename(**job)
        jobs[index]["output_filename"] = output_filename
        job["original_job"] = copy.deepcopy(job)
        jobs[index]["output_file_path"] = f'{output_folder}/{output_filename}'
        jobs[index] = load_single_job(job, configs)
        assert os.path.exists(jobs[index]['data']['path'])
        jobs[index] = check_and_load_ref_functions(jobs[index])
    return jobs


def get_correlations_list(Y_sliced, Y_predicted):
    correlations_list = []
    for i in range(Y_sliced.shape[1]):
        correlation = np.corrcoef(Y_sliced[:, i], Y_predicted[:, i], rowvar=False)[0, 1]
        correlations_list.append(float(correlation))
    return correlations_list


def process(job):
    start_time = time.time()
    output_file_path = job["output_file_path"]
    with open(output_file_path, 'w') as outfile:
        outfile.write("")

    loader = job["data"]["loader_reference"]
    filepath = job["data"]["path"]
    frequency = job["data"]["frequency"]
    X, Y = loader(filepath)

    channels = job["channels"]["channels"]
    X = X[:, channels].reshape((-1, len(channels)))

    targets = job["targets"]["channels"]
    Y = Y[:, targets].reshape((-1, len(targets)))

    data_preprocessor = job["data"]["loader_reference"]
    for preprecessing_stage in job["preprocess"]["preprocessing_pipeline"]:
        data_preprocessor = preprecessing_stage["function_reference"]
        X, Y, frequency = data_preprocessor(X=X, Y=Y, frequency=frequency, **preprecessing_stage["kwargs"])

    output = {
        "job": job["original_job"],
        "results": []
    }
    output["job"]["start_time"] = int(start_time)

    splitter = job["splitter"]["splitter_method_reference"]
    for X_train, Y_train, X_val, Y_val, X_test, Y_test in splitter(X, Y):
        start_subjob_time = time.time()
        assert len(Y_train.shape) == len(Y_test.shape) == 2, \
            f"Expected get 2 demensional Y got: Y_train {Y_train.shape}, Y_test {Y_test.shape}"
        assert(Y_train.shape[1:] == Y_test.shape[1:])
        assert len(X_train) >= 2
        assert len(X_val) >= 2
        assert len(X_test) >= 2
        assert(X_train.shape[1:] == X_val.shape[1:] == X_test.shape[1:])

        input_shape = X_train.shape[1:]
        output_shape = Y_train.shape[1:]

        bench_model_class = job["model"]["model_base_class_reference"]
        bench_model = bench_model_class(input_shape=input_shape, output_shape=output_shape, frequency=frequency, **job["model"]["kwargs"])
        bench_model.fit(copy.deepcopy(X_train), copy.deepcopy(Y_train), copy.deepcopy(X_val), copy.deepcopy(Y_val))

        Y_train_predicted = bench_model.predict(copy.deepcopy(X_train))
        Y_val_predicted = bench_model.predict(copy.deepcopy(X_val))
        Y_test_predicted = bench_model.predict(copy.deepcopy(X_test))

        Y_train_sliced = bench_model.slice_target(copy.deepcopy(Y_train))
        Y_val_sliced = bench_model.slice_target(copy.deepcopy(Y_val_predicted))
        Y_test_sliced = bench_model.slice_target(copy.deepcopy(Y_test))

        assert len(Y_train_predicted.shape) == len(Y_train_sliced.shape) == len(Y_val_predicted.shape) == len(Y_val_sliced.shape) == len(Y_test_predicted.shape) == len(Y_test_sliced.shape) == 2, \
            f"Expected get 2 demensional Y got: Y_train_predicted {Y_train_predicted.shape}, Y_train_sliced {Y_train_sliced.shape}, Y_test_predicted {Y_test_predicted.shape}, Y_test_sliced {Y_test_sliced.shape}"
        assert Y_train_predicted.shape == Y_train_sliced.shape, \
            f'Y_train_predicted.shape {Y_train_predicted.shape} != Y_train_sliced.shape {Y_train_sliced.shape}'
        assert Y_val_predicted.shape == Y_val_sliced.shape, \
            f'Y_val_predicted.shape {Y_val_predicted.shape} != Y_val_sliced.shape {Y_val_sliced.shape}'
        assert Y_test_predicted.shape == Y_test_sliced.shape, \
            f'Y_test_predicted.shape {Y_test_predicted.shape} != Y_test_sliced.shape {Y_test_sliced.shape}'

        end_subjob_time = time.time()
        result = {
            # "Y_train_predicted": Y_train_predicted.tolist(),
            # "Y_train_sliced": Y_train_sliced.tolist(),
            # "Y_test_predicted": Y_test_predicted.tolist(),
            # "Y_test_sliced": Y_test_sliced.tolist(),
            "train_correlation": get_correlations_list(Y_train_sliced, Y_train_predicted),
            "val_correlation": get_correlations_list(Y_val_sliced, Y_val_predicted),
            "test_correlation": get_correlations_list(Y_test_sliced, Y_test_predicted),
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
    jobs = load_and_check_jobs(jobs_filepath, configs, output_folder)
    sys.stderr.write('Configs successfully loaded\n')
    if len(jobs) == 0:
        sys.stderr.write('No undone jobs found\n')
    while True:
        jobs = update_jobs(jobs)
        if len(jobs) == 0:
            sys.stderr.write('All Jobs Are DONE!\n')
            return
        else:
            sys.stderr.write("Jobs left: {}\n\n".format(len(jobs)))
        job_to_process = jobs[0]
        output_filename = job_to_process["output_filename"]
        sys.stderr.write("Process start: {}\n".format(output_filename))
        process(job_to_process)
        sys.stderr.write("Process finished: {}\n".format(output_filename))
