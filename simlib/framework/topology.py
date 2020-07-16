"""
topology.py
written in Python3
author: C. Lockhart <chris@lockhartlab.org>
"""

from .errata import pivot

from glovebox import GloveBox
import numpy as np
import os
import pandas as pd


# Topology class
class Topology:
    """
    A `Topology` is an object that stores metadata for a `Trajectory`, such as atomic relationships

    "Atomic" can be used loosely here because it applies to proteins, proteins in water baths, proteins in lipid
    environments, ligands in water, etc.

    The `Topology` instance generally follows PDB format [1] in that in contains,
      - atom_id
      - atom
      - residue
      - chain
      - residue_id
      - alpha
      - beta
      - segment
      - element
    """

    # Required columns
    columns = [
        'index',
        'atom_id',
        'atom',
        'residue',
        'chain',
        'residue_id',
        'alpha',
        'beta',
        'segment',
        'element',
    ]

    # Initialize class instance
    def __init__(self, data=None):
        """
        Initialize class instance
        """

        # Add index to data
        if isinstance(data, pd.DataFrame) and 'index' not in data.columns:
            data['index'] = np.arange(len(data))

        # TODO can this be protected?
        # Save data or create blank DataFrame
        self._data = data if data is not None else pd.DataFrame(columns=self.columns)

    # Add atom to topology
    def __add__(self, other):
        pass

    # Get attribute
    def __getattr__(self, item):
        if item not in self._data:
            raise AttributeError('%s not in Structure' % item)

        return self._data[item].values

    # Get atom from Structure by index
    def __getitem__(self, item):
        """


        Parameters
        ----------
        item

        Returns
        -------

        """

        # First, try to see if we can ping the DataFrame
        if isinstance(item, str):
            return self._data[item].values

        # Make sure that item a valid atom_id
        if item not in self._data['atom_id']:
           raise AttributeError('%s not a valid atom_id' % item)

        # Create copy of the row
        data = self._data[self._data['atom_id'] == item].copy()

        # Sanity
        if len(data) != 1:
            raise AttributeError('multiple atom_id %s found in Structure' % item)

        # Return (as Series)
        return data.iloc[0, :]

    # Length of structure (returns number of atoms)
    def __len__(self):
        """
        Number of atoms in the Structure instance.

        Returns
        -------
        int
            Number of atoms in the Structure instance.
        """

        return self.n_atoms

    # Add new atoms f
    def add_atoms(self, **kwargs):
        pass

    # Apply
    def apply(self, function):
        return function(self)

    # TODO is this too specific? Evaluate if Structure should be more generalized
    # Compute secondary structure
    # def compute_secondary_structure(self, recompute=False, executable='stride'):
    #     # If secondary_structure is not in the Structure, or recompute is true, compute
    #     if 'secondary_structure' not in self._data or recompute:
    #         # TODO in lieu of directly porting stride to Python, how can this be sped up?
    #         # Write out PDB of structure to glovebox
    #         gb = GloveBox('simlib-stride', persist=True)
    #         temp_pdb = os.path.join(gb.path, 'temp.pdb')
    #         self.to_pdb(temp_pdb)
    #
    #         # Run STRIDE
    #         secondary_structure = stride(temp_pdb, executable=executable)[['residue_id', 'secondary_structure']]
    #
    #         # Clean glovebox
    #         gb.clean()
    #
    #         # Drop secondary_structure if it's already in self._data
    #         self._data = self._data.drop(columns='secondary_structure', errors='ignore')
    #
    #         # Merge STRIDE results with Structure to store
    #         self._data = self._data.merge(secondary_structure, how='inner', on='residue_id', validate='m:1')
    #
    #     # Return
    #     return self._data[['residue_id', 'secondary_structure']].drop_duplicates()

    # Copy
    def copy(self):
        """
        Create a copy of `Topology` instance.

        Returns
        -------
        simlib.Topology
            Copy of instance.
        """

        return Topology(self._data.copy())

    # Number of atoms
    @property
    def n_atoms(self):
        """
        Get the number of atoms in the `Topology`.

        Returns
        -------
        int
            Number of atoms
        """

        return len(self._data)

    # Pivot
    def pivot(self, index, columns=None, values=None, aggfunc='mean'):
        return pivot(self._data, index=index, columns=columns, values=values, aggfunc=aggfunc)

    # TODO should this return a Structure or Trajectory? TBD
    # TODO what if there were an intermediate object, like a Query class that knew how to handle subindexing / setting
    # Query
    def query(self, text):
        # TODO make this so it's customizable
        # Run macros
        text = text.lower().replace('peptide', 'residue in ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", ' +
                                    '"HIS", "HSD", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", ' +
                                    '"SER", "THR", "TRP", "TYR", "VAL"]')

        # Get indices and parsed data
        indices = self._data.query(text).index.values
        data = self._data.loc[self._data.index.isin(indices), :].copy()

        # Return
        return Topology(data)

    # TODO extend this to another trajectory
    # Compute RMSD
    def rmsd(self, structure):
        pass

    # Write to CSV
    def to_csv(self, path, topology=False):
        pass

    # Convert to pandas DataFrame
    def to_dataframe(self):
        """
        Convert to pandas.DataFrame

        Returns
        -------
        pandas.DataFrame
            `Topology` as pandas.DataFrame
        """

        return self._data.copy()

    # Convert to trajectory
    def to_trajectory(self, structure_id=None):
        # Create and/or update structure_id if necessary
        if 'structure_id' not in self._data:
            structure_id = 0
        if structure_id is not None:
            self._data['structure_id'] = structure_id

        # Return Trajectory
        return Trajectory(self._data)

    # Get coordinates
    @property
    def xyz(self):
        """

        Returns
        -------

        """

        return self._data[['x', 'y', 'z']].values
