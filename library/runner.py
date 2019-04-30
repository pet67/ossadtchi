# coding=utf-8
import os
import random
import sys

from . import process


JOB_PARAMETERS_SEPARATOR = "__"


def is_job_valid(job):
    requered_fields = ["data", "channels", "targets", "preprocessing", "model", "testing"]
    for field in requered_fields:
        assert field in job
    for field in job:
        assert field in requered_fields
    return True


def to_result_filename(data, channels, targets, preprocessing, model, testing):
    return JOB_PARAMETERS_SEPARATOR.join([data, channels, preprocessing, model, testing])


def from_result_filename(filename):
    assert(len(filename.split(JOB_PARAMETERS_SEPARATOR)) == 6)
    dataset, channels, preprocessing, model, cv = filename.split(JOB_PARAMETERS_SEPARATOR)
    return dataset, channels, preprocessing, model, testing


def update_jobs(jobs):
    new_jobs = []
    for job in jobs:
        if not os.path.exists(job["output_file_path"]):
            new_jobs.append(job)
    random.shuffle(new_jobs)
    return new_jobs


def runner(output_folder, jobs, configs):
    for index, job in enumerate(jobs):
        is_job_valid(job)
        output_filename = to_result_filename(**job)
        jobs[index]["output_filename"] = output_filename
        jobs[index]["output_file_path"] = f'{output_folder}/{output_filename}'

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
        process.process(job_to_process, configs)
        sys.stderr.write("Process finished: {}\n".format(output_filename))
