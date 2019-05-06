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
    runner_parser.add_argument('-j', '--jobs_file', default="jobs.json", required=False)
    runner_parser.add_argument('-c', '--configs_folder', default="configs", required=False)
    runner_parser.add_argument('-o', '--output_folder', default="results", required=False)


    clear_parser = subparsers.add_parser('clear')

    parsed_args = parser.parse_args()

    for attr in ["jobs_file", "output_folder", "configs_folder"]:
        if hasattr(parsed_args, attr):
            original_path = getattr(parsed_args, attr)
            abs_path = os.path.abspath(original_path)
            setattr(parsed_args, attr, abs_path)
            assert os.path.exists(abs_path)

    assert 'CUDA_VISIBLE_DEVICES' in os.environ, 'CUDA_VISIBLE_DEVICES should be specified'

    return parsed_args


if __name__ == '__main__':
    parsed_args = parse_args()
    if parsed_args.cmd == "runner":
        library.runner.runner(parsed_args.output_folder, parsed_args.jobs_file, parsed_args.configs_folder)
    elif parsed_args.cmd == "clear":
        pass
    else:
        raise(ValueError)
