"""
setup.py
written in Python3
author: C. Lockhart <chris@lockhartlab.org>
"""

import numpy as np
from setuptools import setup
import yaml


# Read version
with open('version.yml', 'r') as f:
    version_data = yaml.safe_load(f.read())

# Convert the version_data to a string
version = '.'.join([str(version_data[key]) for key in ['major', 'minor', 'patch']])

# Read in requirements.txt
requirements = np.loadtxt('requirements.txt', dtype='str').tolist()

# Setup
setup(
    name='simlib',
    version=version,
    author='C. Lockhart',
    author_email='chris@lockhartlab.org',
    description='A toolkit for molecular dynamics simulations',
    long_description='A toolkit for molecular dynamics simulations',
    long_description_content_type='text/x-rst',
    url="https://www.lockhartlab.org",
    packages=[
        'simlib',
        'simlib.analysis',
        'simlib.geometry',
        'simlib.io',
        'simlib.misc',
    ],
    install_requires=requirements,
    include_package_data=True,
    zip_safe=True
)
