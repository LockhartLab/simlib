
import yaml

# read in version
with open('version.yml', 'r') as f:
    data = yaml.safe_load(f.read())

# update patch
data['patch'] += 1

# output version
with open('version.yml', 'w') as f:
    yaml.safe_dump(data, f)

# create version string
version = '.'.join([str(data[key]) for key in ['major', 'minor', 'patch']])

# write version string to simlib/_version.py
with open('simlib/_version.py', 'w') as f:
    f.write("__version__ = '{}'\n".format(version))
