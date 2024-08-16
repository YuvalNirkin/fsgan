""" Batch processing utility. """

import os
import argparse
from glob import glob
import inspect
from itertools import product
import traceback
import logging
from fsgan.utils.obj_factory import partial_obj_factory


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('source', metavar='STR', nargs='+',
                    help='sources input')
parser.add_argument('-t', '--target', metavar='STR', nargs='*',
                    help='targets input')
parser.add_argument('-o', '--output', metavar='DIR',
                    help='output directory')
parser.add_argument('-fo', '--func_obj', default='fsgan.utils.batch.echo', metavar='OBJ',
                    help='function object including kwargs')
parser.add_argument('-p', '--postfix', metavar='POSTFIX',
                    help='input files postfix')
parser.add_argument('-op', '--out_postfix', metavar='POSTFIX',
                    help='output files postfix')
parser.add_argument('-i', '--indices',
                    help='python style indices (e.g 0:10')
parser.add_argument('-se', '--skip_existing', action='store_true',
                    help='skip existing output file our directory')
parser.add_argument('-ro', '--reverse_output', action='store_true',
                    help='reverse the output name to be <target>_<source>')
parser.add_argument('-io', '--ignore_output', action='store_true',
                    help='avoid specifying an output parameter for the function object')


def main(source, target=None, output=None, func_obj=None, postfix=None, out_postfix=None, indices=None,
         skip_existing=False, reverse_output=False, ignore_output=False):
    out_postfix = postfix if out_postfix is None else out_postfix
    out_postfix = '' if out_postfix is None else out_postfix
    ignore_output = True if output is None else ignore_output

    # Parse input paths
    source_paths = parse_paths(source, postfix)
    target_paths = parse_paths(target, postfix)
    assert len(source_paths) > 0, 'Found 0 source paths'
    assert target_paths is None or len(target_paths) > 0, 'Found 0 target paths'

    if target_paths is None:
        input_paths = source_paths
    else:
        input_paths = list(product(source_paths, target_paths))
        input_paths = [(p1, p2) for p1, p2 in input_paths if os.path.basename(p1) != os.path.basename(p2)]
    input_paths = eval('input_paths[%s]' % indices) if indices is not None else input_paths

    # Get function object instance
    partial_func_obj = partial_obj_factory(func_obj)
    func_obj = partial_func_obj() if inspect.isclass(partial_func_obj.func) else partial_func_obj

    # For each input path
    for i, curr_input in enumerate(input_paths):
        if isinstance(curr_input, (list, tuple)):
            if reverse_output:
                out_vid_name = os.path.splitext(os.path.basename(curr_input[1]))[0] + '_' + \
                               os.path.splitext(os.path.basename(curr_input[0]))[0] + out_postfix
            else:
                out_vid_name = os.path.splitext(os.path.basename(curr_input[0]))[0] + '_' + \
                               os.path.splitext(os.path.basename(curr_input[1]))[0] + out_postfix
        else:
            out_vid_name = os.path.splitext(os.path.basename(curr_input))[0]
            curr_input = [curr_input]
        out_vid_path = os.path.join(output, out_vid_name) if output is not None else None
        if skip_existing and os.path.exists(out_vid_path):
            print('[%d/%d] Skipping "%s"' % (i + 1, len(input_paths), out_vid_name))
            continue

        print('[%d/%d] Processing "%s"...' % (i + 1, len(input_paths), out_vid_name))
        try:
            func_obj(*curr_input) if ignore_output else func_obj(*curr_input, out_vid_path)
        except Exception as e:
            logging.error(traceback.format_exc())


def parse_paths(inputs, postfix=None):
    postfix = '' if postfix is None else postfix
    if inputs is None:
        return None
    input_paths = []
    i = 0
    while i < len(inputs):
        if os.path.isfile(inputs[i]):
            ext = os.path.splitext(inputs[i])[1]
            if ext == '.txt':
                # Found a list file with absolute paths
                with open(inputs[i], 'r') as f:
                    file_abs_paths = f.read().splitlines()
                input_paths += file_abs_paths
            else:
                input_paths.append(inputs[i])
        elif os.path.isdir(inputs[i]):
            if (i + 1) < len(inputs) and os.path.splitext(inputs[i + 1])[1] == '.txt':
                # Found root directory and list file pair
                file_list_path = inputs[i + 1] if os.path.exists(inputs[i + 1]) \
                    else os.path.join(inputs[i], inputs[i + 1])
                assert os.path.isfile(file_list_path), f'List file does not exist: "{inputs[i + 1]}"'
                with open(file_list_path, 'r') as f:
                    file_rel_paths = f.read().splitlines()
                input_paths += [os.path.join(inputs[i], p) for p in file_rel_paths]
                i += 1
            else:
                # Found a directory
                # Parse the files in the directory
                input_paths += glob(os.path.join(inputs[i], '*' + postfix))
        elif any(c in inputs[i] for c in ['*']):
            input_paths += glob(inputs[i])

        i += 1

    return input_paths


def echo(*args, **kwargs):
    print('Received the following input:')
    print(f'args = {args}')
    print(f'kwargs = {kwargs}')


if __name__ == "__main__":
    main(**vars(parser.parse_args()))
