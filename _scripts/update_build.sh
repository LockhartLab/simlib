#!/bin/bash

# checkout master branch
git checkout -qf master

# configure git
git config --global user.name "travis"
git config --global user.email "travis"

# increment version
python _scripts/increment_version.py

# build documentation
(
  cd docs || exit
  rm -r build/
  make html
)

# upload to git without triggering TravisCI again
cat version.yml
git add version.yml
git add simlib/_version.py
git add docs/build/*
git add docs/source/api/generated/*
git commit -m "updating doc and version [ci skip]"
git push -fq https://doclockh:"${GIT_API_KEY}"@github.com/"${TRAVIS_REPO_SLUG}".git master
