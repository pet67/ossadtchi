# coding=utf-8
import os
import scipy
import scipy.io
import json
import sklearn
import sklearn.linear_model
import sklearn.metrics
import numpy as np
import time

import tensorflow as tf
from tensorflow.python.client import device_lib
config = tf.ConfigProto()  # noqa
config.gpu_options.allow_growth = True  # noqa
sess = tf.Session(config=config)  # noqa

import keras
from keras.layers import Input
from keras.models import Model
from keras import optimizers

from . import preprocessing
from . import nets


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
    data_loader, channels, targets, data_preprocessor, model_instance, testing_processor = get_tools(job)
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

        model_instance.fit(X_train, Y_train)
        Y_train_predicted = model_instance.predict(X_train)
        Y_test_predicted = model_instance.predict(X_test)

        Y_train_sliced = model_instance.slice(Y_train)
        Y_test_sliced = model_instance.slice(X_test)

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
