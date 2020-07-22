"""
r1N.py

author: C. Lockhart <chris@lockhartlab.org>
"""

def r1N(a, selection='atom_name == CA'):
    """

    Parameters
    ----------
    a

    Returns
    -------

    """

    residues = a.topology['residue']
    residue0 = residues.min()
    residue1 = residues.max()

    a.query('atom_name == "CA"')
    pass
