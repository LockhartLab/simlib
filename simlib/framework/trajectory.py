"""
trajectory.py
written in Python3
author: C. Lockhart <chris@lockhartlab.org>
"""

from .topology import Topology

import numpy as np
import pandas as pd
# from typelike import IntLike


# Trajectory class
class Trajectory(object):
    """
    `Trajectory` stores instantaneous data for a `Topology`, i.e., atomic coordinates, box information, velocity, etc.
    """

    # Initialize instance of Trajectory
    def __init__(self, xyz, box=None, topology=None):
        """


        Parameters
        ----------
        xyz : ArrayLike, three-dimensional
            Cartesian coordinates for all atoms and all structures. First axis must represent the structure index,
            second axis must represent the atom, and the third index will represent x, y, or z.
        topology : Topology
        """

        self._xyz = xyz
        self._topology = topology

    # Length of trajectory
    def __len__(self):
        """
        The length of the trajectory is the number of structures.

        Returns
        -------
        int
            Length of trajectory
        """

        return self.n_structures

    # Get item
    def __getitem__(self, item):
        """

        Parameters
        ----------
        item

        Returns
        -------

        """

        # TODO create an IntLike object for typelike
        if isinstance(item, (int, np.int, np.int64)):
            result = self.get_structure(item)

        # If we still don't know what to do, try to get from topology
        else:
            result = self.topology[item]

        return result

    # Check topology
    def _check_topology(self):
        """
        Check that the topology is correctly set
        """

        # First off, must be Topology instance
        if not isinstance(self._topology, Topology):
            raise AttributeError('topology is not correct class (%s)' % type(self._topology))

        # Second, number of atoms must match
        if self.n_atoms != self._topology.n_atoms:
            raise AttributeError('number of atoms in topology and trajectory do not match ({0} vs {1})'
                                 .format(self.n_atoms, self._topology.n_atoms))

    # Apply
    # TODO please optimize this. Use vectorize, map, parallelization
    def apply(self, function):
        result = []
        for i in np.arange(self.n_structures):
            structure = self.get_structure(i)
            result.append(function(structure))
        return result

    # Get atoms
    def get_atoms(self, index):
        return self.xyz[:, index, :]

    # Get structure
    def get_structure(self, index):
        index = np.array([index]).ravel()  # this cuks
        structure = Trajectory(self.xyz[index, :, :].reshape(len(index), self.n_atoms, self.n_dim),
                               topology=self.topology)
        if self.n_atoms != structure.n_atoms:
            raise AttributeError('number of atoms do not match ({0} vs {1})'.format(self.n_atoms, structure.n_atoms))
        return structure

    # Number of atoms
    @property
    def n_atoms(self):
        """
        Number of atoms in the `Trajectory`.

        Returns
        -------
        int
            Number of atoms
        """

        return self.shape[1]

    # Number of dimensions
    @property
    def n_dim(self):
        """
        Number of dimensions.

        Returns
        -------
        int
            Number of dimensions
        """

        return self.shape[2]

    # Number of structures
    @property
    def n_structures(self):
        """
        Number of structures in the `Trajectory`.

        Returns
        -------
        int
            Number of structures
        """

        return self.shape[0]

    # Query
    def query(self, text, only_index=False):
        """
        Query the set `Topology` and return matching indices

        Parameters
        ----------
        text
        only_index

        Returns
        -------

        """

        # Query the topology to get pertinent data
        topology = self.topology.query(text)

        # Extract indices
        index = topology['index']

        #
        if only_index:
            result = index

        else:
            result = Trajectory(self.get_atoms(index).reshape(self.n_structures, len(index), self.n_dim),
                                topology=topology)

        # Return result
        return result

    # Shape
    @property
    def shape(self):
        """
        Shape of `Trajectory`

        Returns
        -------
        tuple
            (Number of structures, number of atoms, dimensionality)
        """

        return self._xyz.shape

    # Convert to pandas DataFrame
    def to_dataframe(self):
        """
        Convert `Trajectory` to pandas.DataFrame instance

        Returns
        -------
        pandas.DataFrame
            Trajectory represented as pandas DataFrame
        """

        # Prepare the data as DataFrame
        result = pd.DataFrame({
            'index': np.tile(np.arange(self.n_atoms), self.n_structures),
            'structure_id': np.repeat(np.arange(self.n_structures), self.n_atoms),
            'x': self._xyz[:, :, 0].ravel(),
            'y': self._xyz[:, :, 1].ravel(),
            'z': self._xyz[:, :, 2].ravel(),
        })

        # Return
        return result

    # Show
    def show(self):
        pass

    # Save as PDB
    def to_pdb(self, filename):
        """
        Convert `Trajectory` to PDB.

        Parameters
        ----------
        filename : str
            Name of PDB to write.
        """

        # Convert topology and trajectory to DataFrame
        topology = self.topology.to_dataframe()
        trajectory = self.to_dataframe()

        # Merge trajectory and topology
        data = trajectory.merge(topology, how='inner', on='index').drop(columns='index')

        # Sort by structure_id and then atom_id
        data = data.sort_values(['structure_id', 'atom_id'])

        # Add ATOM record
        data['record'] = 'ATOM'

        # TODO this should also be done for residue id probably
        # Change base-0 to base-1 for atom_id
        if data['atom_id'].min() == 0:
            data['atom_id'] += 1

        # Format atom names
        i = data['atom'].str.len() == 1
        data.loc[i, 'atom'] = ' ' + data.loc[i, 'atom'] + '  '

        i = data['atom'].str.len() == 2
        data.loc[i, 'atom'] = ' ' + data.loc[i, 'atom'] + ' '

        # Open up the buffer
        with open(filename, 'w') as buffer:
            # Select every structure and write out
            # TODO please make this more efficient
            for structure in data['structure_id'].unique():
                is_structure = data['structure_id'] == structure

                # Select pertinent rows and columns
                structure = data.loc[is_structure, ['record', 'atom_id', 'atom', 'residue', 'chain', 'residue_id',
                                                    'x', 'y', 'z', 'alpha', 'beta', 'segment', 'element']]

                # Write out PDB file
                np.savetxt(
                    buffer,
                    structure,
                    fmt='%-6s%5i %4s%4s%2s%4i%12.3f%8.3f%8.3f%6.2f%6.2f%9s%2s',
                    header='CRYST1    0.000    0.000    0.000  90.00  90.00  90.00 P 1           1',
                    footer='END\n',
                    comments=''
                )

    # Convert to MDAnalysis Universe
    # mdanalysis.org
    def to_universe(self):
        pass

    # Get topology
    @property
    def topology(self):
        """
        Get the `Topology` instance

        Returns
        -------
        Topology
            `Topology` instance associated with this `Trajectory`.
        """

        # Check the topology
        self._check_topology()

        # Return
        return self._topology

    # View
    # https://github.com/arose/nglview
    def view(self):
        pass

    # Get XYZ coordinates
    @property
    def xyz(self):
        """
        Get x, y, and z coordinates.

        Returns
        -------
        numpy.ndarray
            Cartesian coordinates.
        """

        return self._xyz
