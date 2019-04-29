# coding=utf-8
import os
import random
import sys

from . import process


def to_result_filename(dataset, channels, preprocessing, model, cv):
    return "{dataset}:{channels}:{preprocessing}:{model}:{cv}"


def from_result_filename(filename):
    assert(len(filename.split(":")) == 5)
    dataset, channels, preprocessing, model, cv = filename.split(":")
    return dataset, channels, preprocessing, model, cv


def update_jobs(jobs, jobs_retry_count, output_folder):
    new_jobs = []
    for job in jobs:
        if jobs_retry_count[job] > 0 and not os.path.exists("{}/{}".format(output_folder, job)):
            new_jobs.append(job)
    random.shuffle(new_jobs)
    return new_jobs


def runner(input_files, output_folder, model, config):
    jobs = []
    for file in input_files:
        if model != "net_all":
            for channel in range(config["number_of_channels"]):
                jobs.append(to_result_filename(file, channel, model))
        else:
            jobs.append(to_result_filename(file, -1, model))
    jobs_retry_count = {}
    for job in jobs:
        jobs_retry_count[job] = 1

    while len(jobs) > 0:
        jobs = update_jobs(jobs, jobs_retry_count, output_folder)
        sys.stderr.write("Jobs left: {}\n\n".format(len(jobs)))
        if len(jobs) == 0:
            sys.stderr.write('All Jobs Are DONE!\n')
            return
        job_to_process = jobs[0]
        jobs_retry_count[job_to_process] -= 1
        original_file, channel, model_from_job = to_original_file_and_channel_and_model(job_to_process)
        assert(model == model_from_job)
        output_file = output_folder + "/" + job_to_process
        sys.stderr.write("Process start: {}\n".format(job_to_process))
        process.process(original_file, channel, output_file, model, config)
        sys.stderr.write("Process finished: {}\n".format(job_to_process))
