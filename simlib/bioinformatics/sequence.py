"""
sequence.py

>>> abeta = Protein('YEVHHQKLVFFAEDVGSNKGAIIGLMVGGVV')
"""

from typelike import ArrayLike


class Sequence:
    def __init__(self):
        pass


class Protein(Sequence):
    """
    Construct for dealing with protein sequence.

    """

    def __init__(self, residues):
        # Convert to list if not already list
        super().__init__()
        if not isinstance(residues, ArrayLike):
            residues = list(residues)

        # Make sure we're in the right format
        letter_to_code = {
            'A': 'ALA',
            'R': 'ARG',
            'N': 'ASN',
            'D': 'ASP',
            'C': 'CYS',
            'Q': 'GLN',
            'E': 'GLU',
            'G': 'GLY',
            'H': 'HIS',
            'I': 'ILE',
            'L': 'LEU',
            'K': 'LYS',
            'M': 'MET',
            'F': 'PHE',
            'P': 'PRO',
            'S': 'SER',
            'T': 'THR',
            'W': 'TRP',
            'Y': 'TYR',
            'V': 'VAL'
        }
        self.residues = [letter_to_code.get(residue, residue) for residue in residues]

    def to_str(self):
        pass
