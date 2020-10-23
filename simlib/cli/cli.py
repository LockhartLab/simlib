"""
simlib module -top trj.psf -dcd trj.dcd
"""

import argparse


# Arguments
parser = argparse.ArgumentParser(description='command-line interface to simlib')
parser.add_argument('module', help='simlib module to call')
args = parser.parse_args()

print(args)

if __name__ == '__main__':
    pass




