# coding=utf-8
import argparse
import sys
import os
import os.path

import library
import library.runner


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)

    subparsers = parser.add_subparsers(dest='cmd')

    runner_parser = subparsers.add_parser('runner')
    runner_parser.add_argument('-j', '--jobs_file', required=True)
    runner_parser.add_argument('-o', '--output_folder', required=True)

    clear_parser = subparsers.add_parser('clear')

    parsed_args = parser.parse_args()

    for attr in ["jobs_file", "output_folder"]:
        if hasattr(parsed_args, attr):
            original_path = getattr(parsed_args, attr)
            setattr(parsed_args, attr, os.path.abspath(original_path))

    assert 'CUDA_VISIBLE_DEVICES' in os.environ, 'CUDA_VISIBLE_DEVICES should be specified'

    return parsed_args


def load_json(path):
    with open(path) as file_stream:
        data = json.load(file_stream)
    return data


def load_config(path):
    config = load_json(f"configs/{config_name}.json")
    config = {config_row["name"]: config_row for config_row in config}
    return config


def load_configs():
    configs = {}
    for config_name in ["channels_config", "datasets_config", "models_config", "preprocess_config", "synthetic_config"]:
        configs[config_name] = load_config(config_name)
    return configs


def load_jobs(jobs_file):
    jobs = load_json(jobs_file)
    return jobs

if __name__ == '__main__':
    parsed_args = parse_args()
    configs = load_configs()
    if parsed_args.cmd == "runner":
        jobs = load_jobs(parsed_args.jobs_file)
        library.runner.runner(parsed_args.output_folder , jobs, configs)
    elif parsed_args.cmd == "clear":
        pass
    else:
        raise(ValueError)
