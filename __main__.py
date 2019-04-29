# coding=utf-8
import argparse
import sys
import os
import os.path

import library
import library.runner

from keras.layers import Input


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)

    subparsers = parser.add_subparsers(dest='cmd')

    runner_parser = subparsers.add_parser('runner')
    runner_parser.add_argument('-j', '--jobs_file', required=True)
    runner_parser.add_argument('-o', '--output_folder', required=True, help='folder to log results')

    clear_parser = subparsers.add_parser('clear')

    parsed_args = parser.parse_args()

    for attr in ["files_file", "output_folder", "final_file"]:
        if hasattr(parsed_args, attr):
            original_path = getattr(parsed_args, attr)
            setattr(parsed_args, attr, os.path.abspath(original_path))

    assert 'CUDA_VISIBLE_DEVICES' in os.environ, 'CUDA_VISIBLE_DEVICES should be specified'

    return parsed_args


def get_files(files_file):
    files_for_process = []
    with open(files_file) as input_file:
        for row in input_file:
            path = os.path.abspath(row.strip())
            assert os.path.exists(path), "Input file {} should exist".format(path)
            files_for_process.append(path)
    return files_for_process


if __name__ == '__main__':
    parsed_args = parse_args()
    if parsed_args.cmd == "runner":
        files_for_process = get_files(parsed_args.files_file)
        library.runner.runner(files_for_process, parsed_args.output_folder, parsed_args.model, CONFIG)
    elif parsed_args.cmd == "clear":
        pass
    else:
        raise(ValueError)
