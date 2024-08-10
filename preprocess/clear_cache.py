import os
import argparse
from glob import glob


parser = argparse.ArgumentParser(os.path.splitext(os.path.basename(__file__))[0])
parser.add_argument('input', metavar='VIDEO',
                    help='path to input sequence video')
parser.add_argument('-o', '--output', metavar='PATH',
                    help='output video path')
parser.add_argument('-ep', '--except_postfix', default=('_dsfd.pkl',), nargs='+', metavar='POSTFIX',
                    help='cache postfixes not to delete')
default = parser.get_default


def main(input, output=default('output'), except_postfix=default('except_postfix')):
    except_postfix = tuple(except_postfix)

    # Validation
    assert os.path.isfile(input), f'Input path "{input}" does not exist'

    # Parse cache files
    cache_dir = os.path.splitext(input)[0]
    cache_files = glob(os.path.join(cache_dir, '*'))

    # Warning and exit
    if not os.path.isdir(cache_dir):
        print(f'Warning: cache dir "{cache_dir}" does not exist')
        return
    if any([os.path.isdir(f) for f in cache_files]):
        print(f'Warning: "{cache_dir}" is not a cache directory')
        return

    # For each cache file
    delete_cache_dir = True
    for cache_file in cache_files:
        if cache_file.endswith(except_postfix):
            delete_cache_dir = False
            continue
        os.remove(cache_file)

    # Delete cache directory if it is empty
    if delete_cache_dir:
        os.rmdir(cache_dir)


if __name__ == "__main__":
    main(**vars(parser.parse_args()))
