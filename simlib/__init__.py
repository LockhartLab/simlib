
import os

from . import analysis
from . import geometry

from .analysis import *
from .geometry import *
from .io import *
from .misc import *
from .version import __version__


# Contents
__all__ = [
    'analysis',
    'io',
    'include_dir',
    'misc',
    '__version__'
]

__all__.extend(geometry.__all__)

# Add include path
include_dir = os.path.abspath(__file__ + '/../../include')
