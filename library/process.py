# coding=utf-8
import os
import scipy
import scipy.io
import json
import sklearn
import numpy as np
import time

import models
import preprocessing
import data_loaders



def generate_train_val_samples(all_samples, folds):
    assert(len(all_samples.shape) == 1)
    assert(folds > 0)
    train_part = int(1.0 / 2 * len(all_samples))
    train_val_samples = []
    train_samples = all_samples[:train_part]
    val_samples = all_samples[train_part:]
    train_val_samples.append((train_samples, val_samples))
    return train_val_samples


def get_tools(job):
    return (lambda x: x, lambda x: x, lambda x: x, lambda x: x, lambda x: x)
    

def process(job, configs):
    start_time = time.time()
    output_file_path = job["output_file_path"]
    with open(output_file_path, 'w') as outfile:
        outfile.write("")
    data_loader, channels, targets, data_preprocessor, bench_model_class, testing_processor = get_tools(job)
    X, Y = data_loader(job)
    X = X[:, channels]
    Y = Y[:, targets]
    X, Y = data_preprocessor(X, Y, job)
    output = {
        "job": job,
        "results": []
    }
    output["job"]["start_time"] = int(start_time)
    for X_train, Y_train, X_test, Y_test in testing_processor(X, Y):
        start_subjob_time = time.time()
        assert len(Y_train) == len(Y_test) == 2
        assert(Y_train.shape[1:] == Y_test.shape[1:])
        assert len(X_train) == len(X_test)
        assert len(X_train) >= 2
        assert len(X_test) >= 2
        assert(X_train.shape[1:] == X_test.shape[1:])

        input_shape = X_train.shape[1:]
        output_shape = Y_train.shape[1:]
        
        model_config = job["model"]
        kwargs = model_config["kwargs"]
        kwargs["input_shape"] = input_shape
        kwargs["output_shape"] = output_shape
        
        assert hasattr(model_config["data"], real_data_loaders) or \
               hasattr(model_config["model_base_class"], simulated_data_generators)
        bench_model_class = getattr(model_config["model_base_class"], models)
        bench_model = bench_model_class(**kwargs)
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
