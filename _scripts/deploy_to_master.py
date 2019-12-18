"""
deploy_to_master.py
written in Python3
author: C. Lockhart <chris@lockhartlab.org>
"""

from _scripts.increment_version import *
from izzy.io import Git

# Connect to git repository
git = Git()

# We must be on master branch
branch = git.get_branch()
assert branch == 'master', branch

# Increment version; print out string
version = increment_version()
print('package version: {}\n'.format(version))

# Connect to git repository, tag, add files, commit, push
# git.tag('v' + version)
git.add('-A')
git.commit(input('Commit message: '))
git.tag('v' + version)
git.push(remote='origin', branch='master', options='--tags')

