
import os

from . import analysis
from . import external
from . import geometry
from . import viz

from .analysis import *
from .external import *
from .geometry import *
from .io import *
from .misc import *
from .version import __version__
from .viz import *

# Contents
__all__ = [
    'io',
    'include_dir',
    'misc',
    '__version__'
]

__all__.extend(analysis.__all__)
__all__.extend(external.__all__)
__all__.extend(geometry.__all__)
__all__.extend(viz.__all__)

# Add include path
include_dir = os.path.abspath(__file__ + '/../../include')
