"""
deploy_to_dev.py
written in Python3
author: C. Lockhart <chris@lockhartlab.org>
"""

from izzy.io import Git

# Connect to git repository
git = Git()

# We must be on master branch
branch = git.get_branch()
assert branch == 'dev', branch

# Connect to git repository, tag, add files, commit, push
git.add('-A')
git.commit(input('Commit message: '))
git.push(remote='origin', branch='dev')
