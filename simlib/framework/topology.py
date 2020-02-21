
# Topology class
class Topology:
    """
    A `Topology` is an object that stores how atoms are arranged

    The `Toplogy` instance follows CHARMM [1] in that in contains,
      - atom_id
      - atom_name
      - atom_type
      - residue
      - chain
      - segment_name
    """

    columns = [
        'atom_id',
        'atom_name',
        'atom_type',
        'residue',
        'chain',
        'segment_name'
    ]

    # Initialize class instance
    def __init__(self):
        """
        Initalize class instance
        """

        # Create empty DataFrame to store data
        self.data = pd.DataFrame(columns=self.columns)

    # Add atoms
    def add_atoms(self, **kwargs):
        """

        Parameters
        ----------
        kwargs

        Returns
        -------

        """

        _add_atoms(self.data, self.columns, kwargs)


# Helper function to add atoms
def _add_atoms(data, columns, kwargs):
    # If atom_id is not in kwargs, add it
    if 'atom_id' not in kwargs:
        # Get starting atom position
        i = 0 if len(data) == 0 else data['atom_id'].max()

        # Get number of elements to add
        n = 0
        for column in columns:
            if column in kwargs:
                n = len(kwargs[column])
                break

        # Increment
        kwargs['atom_id'] = np.arange(i, i + n)

    # Create dictionary to store new data
    new_data = {}

    # Go through all required columns and add to dictionary if present
    for column in columns:
        if column in kwargs:
            new_data[column] = kwargs[column]

    # Convert new_data to DataFrame
    new_data = pd.DataFrame(new_data)

    # Return merged data
    return pd.concat([data, new_data])