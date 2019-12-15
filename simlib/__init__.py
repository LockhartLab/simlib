import os

from .analysis import *
from .geometry import *
from .io import *
from .misc import *
from ._version import __version__


# Contents
__all__ = [
    'analysis',
    'geometry',
    'io',
    'misc',
    '__version__'
]

# Add include path
include_dir = os.path.abspath(__file__ + '/../../include')
