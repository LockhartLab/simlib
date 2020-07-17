"""
deploy_to_master.py
written in Python3
author: C. Lockhart <chris@lockhartlab.org>
"""

# from _scripts.increment_version import *
from gitpipe import Git
import yaml

# Connect to git repository
git = Git()

# We must be on master branch
# branch = git.get_branch()
# assert branch == 'master', branch

# Increment version; print out string
# version = increment_version()
# print('package version: {}\n'.format(version))

with open('version.yml', 'r') as f:
    version = yaml.safe_load(f.read())
version = '.'.join([str(version[key]) for key in ['major', 'minor', 'patch']])

# Connect to git repository, tag, add files, commit, push
# git.add('-A')
# git.commit('[deploy] test successful')
git.tag('v' + version)
# git.push(remote='origin', branch='master', options='--tags')
