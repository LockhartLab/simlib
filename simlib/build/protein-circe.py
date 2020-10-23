"""
protein.py

description: The purpose of this module is to build protein structures from known coordinates.
language: Python 3.x
author: C. Lockhart <chris@lockhartlab.org>


"""

from simlib.io import read_pdb

import numpy as np
import os.path

# TODO move this to root
include_dir = os.path.abspath(__file__ + '../../../../include')


class Builder(object):
    def __init__(self):
        self._xyz = None

    @property
    def shape(self):
        return self._xyz.shape

    @property
    def xyz(self):
        return self._xyz


class ProteinBuilder(Builder):
    def __init__(self):
        # Initialize builder
        Builder.__init__(self)

    def __add__(self, other):
        return self.add_residue(other)

    def add_residue(self, residue, phi=0., psi=0., omega=0.):
        # Get the path to the residue file in include_dir
        residue_file = os.path.join(include_dir, 'protein', 'build', residue.upper() + '.pdb')
        if not os.path.exists(residue_file):
            raise AttributeError('PDB file %s does not exist' % residue_file)

        # Read the residue file as PDB to get structure
        structure = read_pdb(residue_file)
        if structure.n_structures != 1:
            raise AttributeError('expecting PDB file %s with 1 structure' % residue_file)
        structure = structure.get_structure(0)

        # Add the coordinates to the existing structure
        if self._xyz is None:
            self._xyz = structure.xyz[0]

        else:
            # First, place nitrogen in backbone

            # Second, place C-alpha 

            # Finally, add the new coordinates to the total coordinates
            self._xyz = np.vstack([self._xyz, structure.xyz[0]])

        return self


def _dihedral_angle(xyz1, xyz2, xyz3, xyz4):
    vec0 = xyz2 - xyz1
    vec1 = xyz3 - xyz2
    vec2 = xyz4 - xyz3

    vec0 = vec0 / np.linalg.norm(vec0)
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)

    norm0 = np.cross(vec0, vec1)
    norm1 = np.cross(vec1, vec2)

    angle = np.dot(norm0, norm1)

    return angle


def _place_atom(a, b, c, d=np.random.rand(3), r=1., alpha=0., phi=0.):
    while True:
        i = np.random.choice(3)

        if i == 0:
            r_ = np.sqrt(np.sum(np.square(c - d)))
            dr = r_ - r


        elif i == 1:
            pass

        elif i == 2:
            pass


build = ProteinBuilder()
build += 'ALA'
build += 'ALA'
print(build.xyz)
# build += 'ALA'
# build += 'NME'
# build.to_pdb('dialanine.pdb')


if __name__ == '__main__':
    pass
